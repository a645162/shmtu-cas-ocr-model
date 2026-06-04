"""推理子包.

对外 API:
    CaptchaInferencer  - 统一推理引擎, 接受 backend + 标签字典, 给出结构化结果
    PyTorchBackend     - 本地 PyTorch 推理 (加载 .pt)
    OnnxBackend        - ONNX Runtime 推理 (加载 .onnx)
    build_preprocess   - 构造与训练一致的预处理 (灰度+二值化+resize+归一化)

CLI:
    python -m cas_ocr_model.inference.cli \\
        --backend pytorch --checkpoint best.pt --image 00000007.jpg
    python -m cas_ocr_model.inference.cli \\
        --backend onnx --checkpoint model.onnx --dir ./dataset --limit 100
"""
from .inference import CaptchaInferencer, InferencerConfig, InferenceResult
from .preprocess import CaptchaPreprocess, build_preprocess

# 后端按需导入 (onnx 后端需要 onnxruntime, pytorch 后端需要 torch)
__all__ = [
    "CaptchaInferencer",
    "InferencerConfig",
    "InferenceResult",
    "CaptchaPreprocess",
    "build_preprocess",
    "PyTorchBackend",
    "OnnxBackend",
]


def __getattr__(name):
    """懒加载后端, 缺依赖时给出清晰错误."""
    if name == "PyTorchBackend":
        from .backends.pytorch_backend import PyTorchBackend
        return PyTorchBackend
    if name == "OnnxBackend":
        from .backends.onnx_backend import OnnxBackend
        return OnnxBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
