"""推理后端抽象: PyTorch 与 ONNX Runtime 两种实现.

新增后端 (TensorRT / OpenVINO / CoreML) 在此目录下追加 .py 并在 __init__ 注册.
"""
from .onnx_backend import OnnxBackend
from .pytorch_backend import PyTorchBackend

__all__ = ["PyTorchBackend", "OnnxBackend"]
