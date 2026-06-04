import cv2
import numpy as np
import torch

from cas_ocr_model.v1.classify.predict.ResultEnum import get_operator_type_by_int, OperatorEnum, get_operator_type_str, \
    calculate_operator
from cas_ocr_model.v1.classify.predict.load_model import load_model, predict_cv_image
from cas_ocr_model.v1.classify.utils.devices_selector import get_recommended_device
from cas_ocr_model.v1.config import config
from cas_ocr_model.v1.utils.pic.cv2plt import show_opencv_img_by_plt
from cas_ocr_model.v1.utils.pic.spilt_img import spilt_img_by_ratio


def predict_validate_code(
        img_cv_bgr: np.ndarray,
        device: torch.device,
        model_equal_symbol: torch.nn.Module,
        model_operator: torch.nn.Module,
        model_digit: torch.nn.Module,
        print_result: bool = False
) -> (int, str, int, int, int, int):
    img_gray = cv2.cvtColor(img_cv_bgr, cv2.COLOR_BGR2GRAY)
    img_gray = (
        cv2.threshold(
            img_gray,
            config.thresh,
            255,
            cv2.THRESH_BINARY
        )
    )[1]

    # convert to 3 channel
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

    # Show Gray Image
    # show_opencv_img_by_plt(img_gray)

    img_equal_symbol = spilt_img_by_ratio(
        img_gray,
        config.equal_symbol_key_start,
        config.equal_symbol_key_end
    ).copy()

    # show_opencv_img_by_plt(img_equal_symbol)

    result_equal_symbol = predict_cv_image(
        model_equal_symbol,
        device,
        img_equal_symbol
    )

    if result_equal_symbol == 0:
        # chs
        key_point = config.key_point_chs
    else:
        # symbol
        key_point = config.key_point_symbol

    if len(key_point) != 3:
        raise ValueError("key_point must have 3 elements")

    # Spilt Img
    img_digit_1 = spilt_img_by_ratio(
        img_gray,
        0,
        key_point[0]
    ).copy()

    img_operator = spilt_img_by_ratio(
        img_gray,
        key_point[0],
        key_point[1]
    ).copy()

    img_digit_2 = spilt_img_by_ratio(
        img_gray,
        key_point[1],
        key_point[2]
    ).copy()

    # Predict
    result_operator_predict = predict_cv_image(
        model_operator,
        device,
        img_operator
    )

    result_operator: OperatorEnum = OperatorEnum(
        result_operator_predict
    )
    result_operator = get_operator_type_by_int(result_operator)

    result_digit_1 = predict_cv_image(
        model_digit,
        device,
        img_digit_1
    )
    result_digit_2 = predict_cv_image(
        model_digit,
        device,
        img_digit_2
    )

    # print(result_equal_symbol, result_operator_predict, result_digit_1, result_digit_2)

    operator_str = get_operator_type_str(result_operator)

    calc_result = calculate_operator(
        left=result_digit_1,
        right=result_digit_2,
        operator_type=result_operator
    )

    calc_str = f"{result_digit_1} {operator_str} {result_digit_2} = {calc_result}"

    if print_result:
        print(calc_str)

    return (
        calc_result,
        calc_str,
        result_equal_symbol, result_operator_predict, result_digit_1, result_digit_2
    )


if __name__ == "__main__":
    device = get_recommended_device()
    print("device:", device)

    print("Load Model")
    (
        model_equal_symbol,
        model_operator,
        model_digit
    ) = load_model(device)
    print("Load Model Success")

    predict_validate_code(
        cv2.imread("test/test1_20240102160004_server.png"),
        device,
        model_equal_symbol,
        model_operator,
        model_digit,
        True
    )

    predict_validate_code(
        cv2.imread("test/test2_20240102160811_server.png"),
        device,
        model_equal_symbol,
        model_operator,
        model_digit,
        True
    )

    predict_validate_code(
        cv2.imread("test/test3_20240102160857_server.png"),
        device,
        model_equal_symbol,
        model_operator,
        model_digit,
        True
    )

    predict_validate_code(
        cv2.imread("test/test4_20240102160902_server.png"),
        device,
        model_equal_symbol,
        model_operator,
        model_digit,
        True
    )

    predict_validate_code(
        cv2.imread("test/test5_20240102160141_server.png"),
        device,
        model_equal_symbol,
        model_operator,
        model_digit,
        True
    )

    predict_validate_code(
        cv2.imread("test/test6_20240102160146_server.png"),
        device,
        model_equal_symbol,
        model_operator,
        model_digit,
        True
    )
