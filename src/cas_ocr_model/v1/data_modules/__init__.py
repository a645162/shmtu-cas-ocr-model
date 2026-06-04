"""数据模块工具（懒加载 torch）。"""

__all__ = ["CustomDataset", "get_recommended_device"]


def __getattr__(name):
    if name == "CustomDataset":
        from .dataset import CustomDataset as _CustomDataset
        return _CustomDataset
    if name == "get_recommended_device":
        from .device import get_recommended_device as _device
        return _device
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
