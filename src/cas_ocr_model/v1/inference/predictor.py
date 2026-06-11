"""模型加载与端到端验证码预测（equal_symbol / operator / digit）。"""

from __future__ import annotations

import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ..configs.defaults import (
    equal_symbol_key_end,
    equal_symbol_key_start,
    key_point_chs,
    key_point_symbol,
    thresh,
)
from ..configs.model import get_pth_name
from ..configs.paths import pth_save_dir_path
from ..data_modules.device import get_recommended_device
from ..helpers.image import split_image_by_ratio
from ..models.operator_enum import (
    OperatorEnum,
    calculate_operator,
    get_operator_type_by_int,
    get_operator_type_str,
)
from ..models.resnet import init_model

INPUT_SIZE = 224
DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
    ]
)


def get_default_transform() -> transforms.Compose:
    return DEFAULT_TRANSFORM


def load_models(
    device: torch.device,
    pth_dir: str = pth_save_dir_path,
) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    """加载等号/运算符/数字三个最新模型权重。"""
    from ..configs.defaults import (
        model_digit_type,
        model_equal_symbol_type,
        model_operator_type,
    )

    model_equal_symbol = init_model(model_equal_symbol_type, 2, pretrained=False)
    model_operator = init_model(model_operator_type, 6, pretrained=False)
    model_digit = init_model(model_digit_type, 10, pretrained=False)

    model_equal_symbol = model_equal_symbol.to(device)
    model_operator = model_operator.to(device)
    model_digit = model_digit.to(device)

    for model, label, model_type in [
        (model_equal_symbol, "equal_symbol", model_equal_symbol_type),
        (model_operator, "operator", model_operator_type),
        (model_digit, "digit", model_digit_type),
    ]:
        pth_file = os.path.join(pth_dir, get_pth_name(model_type, label, "latest"))
        model.load_state_dict(torch.load(pth_file, map_location=device))
        model.eval()

    return model_equal_symbol, model_operator, model_digit


def predict_cv_image(
    model: torch.nn.Module,
    device: torch.device,
    image_cv_rgb: np.ndarray,
) -> int:
    """对单张 BGR/RGB 图像做预测，返回 argmax 类别。"""
    img = Image.fromarray(image_cv_rgb)
    img_tensor = DEFAULT_TRANSFORM(img).unsqueeze(0).to(device)
    with torch.no_grad():
        _, predicted = torch.max(model(img_tensor), 1)
    return predicted.item()


def predict_validate_code(
    img_cv_bgr: np.ndarray,
    device: torch.device,
    model_equal_symbol: torch.nn.Module,
    model_operator: torch.nn.Module,
    model_digit: torch.nn.Module,
    print_result: bool = False,
) -> tuple[int, str, int, int, int, int]:
    """
    端到端预测：灰度 + 二值化 → 等号识别 → 切三段 → 运算符 + 两个数字。

    Returns:
        (calc_result, calc_str, equal_id, operator_id, digit_left, digit_right)
    """
    img_gray = cv2.cvtColor(img_cv_bgr, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)[1]
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

    img_equal = split_image_by_ratio(
        img_gray, equal_symbol_key_start, equal_symbol_key_end
    ).copy()
    equal_id = predict_cv_image(model_equal_symbol, device, img_equal)

    key_point = key_point_chs if equal_id == 0 else key_point_symbol
    if len(key_point) != 3:
        raise ValueError("key_point must have 3 elements")

    img_digit_1 = split_image_by_ratio(img_gray, 0, key_point[0]).copy()
    img_operator = split_image_by_ratio(img_gray, key_point[0], key_point[1]).copy()
    img_digit_2 = split_image_by_ratio(img_gray, key_point[1], key_point[2]).copy()

    operator_id = predict_cv_image(model_operator, device, img_operator)
    operator_enum = get_operator_type_by_int(OperatorEnum(operator_id))
    operator_str = get_operator_type_str(operator_enum)

    digit_left = predict_cv_image(model_digit, device, img_digit_1)
    digit_right = predict_cv_image(model_digit, device, img_digit_2)

    calc_result = calculate_operator(digit_left, digit_right, operator_enum)
    calc_str = f"{digit_left} {operator_str} {digit_right} = {calc_result}"

    if print_result:
        print(calc_str)

    return calc_result, calc_str, equal_id, operator_id, digit_left, digit_right


if __name__ == "__main__":
    device = get_recommended_device()
    print("[predictor] loading models...")
    eq_model, op_model, digit_model = load_models(device)
    print("[predictor] loaded.")

    for sample in [
        "test/test1_20240102160004_server.png",
        "test/test2_20240102160811_server.png",
        "test/test3_20240102160857_server.png",
        "test/test4_20240102160902_server.png",
        "test/test5_20240102160141_server.png",
        "test/test6_20240102160146_server.png",
    ]:
        predict_validate_code(
            cv2.imread(sample), device, eq_model, op_model, digit_model, True
        )
