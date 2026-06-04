import warnings

import onnx
from onnxconverter_common import float16

model = onnx.load("resnet34_digit_latest.onnx")

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=UserWarning,
        message="the float32 number .* will be truncated to .*"
    )
    model_fp16 = float16.convert_float_to_float16(model)

onnx.save(model_fp16, "resnet34_digit_latest_fp16.onnx")
