"""神经网络模型层（懒加载第三方依赖）。"""

from .registry import ModelType, get_pth_name, model_type_to_path_str
from .operator_enum import (
    OperatorEnum,
    get_operator_type_by_int,
    get_operator_type_str,
    calculate_operator,
)

__all__ = [
    "ModelType",
    "get_pth_name",
    "model_type_to_path_str",
    "init_model",
    "OperatorEnum",
    "get_operator_type_by_int",
    "get_operator_type_str",
    "calculate_operator",
]


def __getattr__(name):
    # 懒加载：仅当用户实际访问 init_model 时才 import torch
    if name == "init_model":
        from .resnet import init_model as _init_model
        return _init_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
