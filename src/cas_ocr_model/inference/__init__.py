"""推理子包.

对外 API:
    CaptchaInferencer  - 统一推理引擎, 接受 backend + 标签字典, 给出结构化结果
    PyTorchBackend     - 本地 PyTorch 推理 (加载 .pt)
    OnnxBackend        - ONNX Runtime 推理 (加载 .onnx)
    NcnnBackend        - ncnn Python 推理 (加载 .param/.bin)
    build_preprocess   - 构造与训练一致的预处理 (灰度+二值化+resize+归一化)

CLI:
    python -m cas_ocr_model.inference.cli \\
        --checkpoint best.pt --image 00000007.jpg
"""
from .inference import CaptchaInferencer, InferencerConfig, InferenceResult
from .preprocess import CaptchaPreprocess, build_preprocess

# 后端按需导入 (pytorch 后端需要 torch)
__all__ = [
    "CaptchaInferencer",
    "InferencerConfig",
    "InferenceResult",
    "CaptchaPreprocess",
    "build_preprocess",
    "PyTorchBackend",
    "OnnxBackend",
    "NcnnBackend",
    "get_backend_availability",
    "is_backend_available",
]


def __getattr__(name):
    """懒加载后端, 缺依赖时给出清晰错误."""
    if name in {
        "PyTorchBackend",
        "OnnxBackend",
        "NcnnBackend",
        "get_backend_availability",
        "is_backend_available",
    }:
        from . import backends
        return getattr(backends, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
