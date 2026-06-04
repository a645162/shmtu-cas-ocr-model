import onnx
from onnxruntime.quantization import preprocess
from onnxruntime.quantization import quantize_dynamic, QuantType

from pathlib import Path

model_fp32 = "resnet34_digit_latest_p.onnx"
model_quant = "resnet34_digit_latest_int8.onnx"

model_fp32 = Path(model_fp32)
model_quant = Path(model_quant)

# python -m onnxruntime.quantization.preprocess --input resnet34_digit_latest.onnx --output resnet34_digit_latest_p.onnx

quantize_dynamic(
    model_fp32,
    model_quant,
    weight_type=QuantType.QUInt8,
)
