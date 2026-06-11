"""OpenVINO 推理。"""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

try:
    from openvino.inference_engine import IECore  # 旧版 API
except ImportError:  # 新版 openvino >= 2022
    from openvino.runtime import Core as IECore  # type: ignore[attr-defined]

from ...configs.paths import pth_save_dir_path

_INPUT_SIZE = 224


def _build_network(model_xml: str, weights_bin: str) -> Any:
    ie = IECore()
    try:
        net = ie.read_network(model=model_xml, weights=weights_bin)
    except AttributeError:
        # 新版 API：read_model
        net = ie.read_model(model=model_xml, weights=weights_bin)
    return ie, net


def _preprocess(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    image = cv2.resize(image, (_INPUT_SIZE, _INPUT_SIZE))
    image = image.transpose((2, 0, 1)).reshape(1, 3, _INPUT_SIZE, _INPUT_SIZE)
    return (image.astype(np.float32) / 255.0)


def infer_with_openvino(
    image_path: str,
    model_name: str = "resnet34_digit_latest",
    model_dir: str | None = None,
) -> int:
    """
    跑一次 OpenVINO 推理并返回 argmax。

    Args:
        image_path: 单张图像路径。
        model_name: 模型文件夹名（默认 resnet34_digit_latest）。
        model_dir: 模型目录；默认用 configs.paths.pth_save_dir_path。
    """
    model_dir = model_dir or pth_save_dir_path
    model_xml = os.path.join(model_dir, model_name, f"{model_name}.xml")
    weights_bin = os.path.join(model_dir, model_name, f"{model_name}.bin")

    ie, net = _build_network(model_xml, weights_bin)
    exec_net = ie.load_network(network=net, device_name="CPU")

    try:
        input_blob = next(iter(net.input_info))
    except AttributeError:
        input_blob = next(iter(net.inputs))
    image = _preprocess(image_path)
    result = exec_net.infer(inputs={input_blob: image})

    try:
        output_blob = next(iter(net.outputs))
    except AttributeError:
        output_blob = next(iter(result))
    return int(np.array(list(result[output_blob][0])).argmax())


if __name__ == "__main__":
    for i in range(10):
        print(f"test/{i}.png ->", infer_with_openvino(f"test/{i}.png"))
