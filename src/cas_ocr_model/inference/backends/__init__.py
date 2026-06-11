"""推理后端注册表.

新增后端时在这里登记类名、模块路径和依赖名.
缺少第三方库时不会拖垮整个 inference 包，只会把对应 backend 标为不可用。
"""
from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec

_BACKEND_REGISTRY: dict[str, tuple[str, str, str]] = {
    "PyTorchBackend": (".pytorch_backend", "PyTorchBackend", "torch"),
    "OnnxBackend": (".onnx_backend", "OnnxBackend", "onnxruntime"),
    "NcnnBackend": (".ncnn_backend", "NcnnBackend", "ncnn"),
}

__all__ = [
    "PyTorchBackend",
    "OnnxBackend",
    "NcnnBackend",
    "get_backend_availability",
    "is_backend_available",
]


def get_backend_availability() -> dict[str, str | None]:
    """返回 backend 类名 -> 缺失依赖名.

    值为 `None` 代表可用；否则为缺失的 Python 依赖包名。
    """
    availability: dict[str, str | None] = {}
    for backend_name, (_, _, dependency_name) in _BACKEND_REGISTRY.items():
        availability[backend_name] = None if find_spec(dependency_name) else dependency_name
    return availability


def is_backend_available(name: str) -> bool:
    missing = get_backend_availability().get(name)
    return missing is None


def __getattr__(name):
    backend_meta = _BACKEND_REGISTRY.get(name)
    if backend_meta is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, class_name, dependency_name = backend_meta
    if find_spec(dependency_name) is None:
        raise ModuleNotFoundError(
            f"{name} 不可用: 缺少依赖 `{dependency_name}`"
        )

    module = import_module(module_name, package=__name__)
    return getattr(module, class_name)
