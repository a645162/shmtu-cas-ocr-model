"""ONNX 模型量化：FP16 / INT8 (dynamic)。"""

from __future__ import annotations

import warnings
from pathlib import Path

import onnx


def quantize_to_fp16(src: str, dst: str) -> None:
    """把 ONNX 转为 FP16。"""
    from onnxconverter_common import float16

    model = onnx.load(src)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        model_fp16 = float16.convert_float_to_float16(model)
    onnx.save(model_fp16, dst)
    print(f"[quant] FP16 -> {dst}")


def quantize_to_int8_dynamic(src: str, dst: str) -> None:
    """对 ONNX 做动态 INT8 量化（weight=QUInt8）。"""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(Path(src), Path(dst), weight_type=QuantType.QUInt8)
    print(f"[quant] INT8 -> {dst}")


if __name__ == "__main__":
    quantize_to_fp16("resnet34_digit_latest.onnx", "resnet34_digit_latest_fp16.onnx")
    quantize_to_int8_dynamic(
        "resnet34_digit_latest_p.onnx", "resnet34_digit_latest_int8.onnx"
    )
