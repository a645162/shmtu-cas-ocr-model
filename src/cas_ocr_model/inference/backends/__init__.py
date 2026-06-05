"""推理后端抽象: PyTorch 实现.

新增后端 (TensorRT / OpenVINO / CoreML) 在此目录下追加 .py 并在 __init__ 注册.
"""

__all__ = ["PyTorchBackend"]


def __getattr__(name):
    if name == "PyTorchBackend":
        from .pytorch_backend import PyTorchBackend
        return PyTorchBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
