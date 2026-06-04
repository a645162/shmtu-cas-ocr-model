"""ONNX Runtime 推理（FP32/FP16/INT8）。"""

from __future__ import annotations

import time
from typing import Literal

import cv2
import numpy as np
import onnxruntime as ort

QuantType = Literal["fp32", "fp16", "int8"]


def _preprocess(image_path: str, quant_type: QuantType) -> np.ndarray:
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image.transpose((2, 0, 1)).reshape(1, 3, 224, 224).astype(np.float32)
    image /= 255.0
    if quant_type == "fp16":
        return image.astype(np.float16)
    if quant_type == "int8":
        return image  # uint8
    return image.astype(np.float32)


def run_onnx_inference(
    onnx_model_path: str,
    image_paths: list,
    quant_type: QuantType = "fp32",
    input_name: str = "input",
) -> list:
    """对一组图像跑 ONNX 推理，返回每张图的 argmax。"""
    session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
    results = []
    for path in image_paths:
        x = _preprocess(path, quant_type)
        t0 = time.time()
        output = session.run(None, {input_name: x})
        elapsed = time.time() - t0
        results.append((path, int(np.array(output[0]).argmax()), elapsed))
    return results


if __name__ == "__main__":
    out = run_onnx_inference(
        "resnet34_digit_latest.onnx",
        [f"test/{i}.png" for i in range(10)],
        quant_type="fp32",
    )
    for item in out:
        print(item)
