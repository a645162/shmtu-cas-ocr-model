"""模型注册表：ModelType 枚举 + 文件名生成。"""

from enum import Enum


class ModelType(Enum):
    ResNet_18 = 1
    ResNet_34 = 2
    ResNet_50 = 3
    ResNet_101 = 4


_MODEL_TYPE_TO_PATH_STR = {
    ModelType.ResNet_18: "resnet18",
    ModelType.ResNet_34: "resnet34",
    ModelType.ResNet_50: "resnet50",
    ModelType.ResNet_101: "resnet101",
}


def model_type_to_path_str(model_type: ModelType) -> str:
    if model_type not in _MODEL_TYPE_TO_PATH_STR:
        raise ValueError(f"Unsupported model type: {model_type}")
    return _MODEL_TYPE_TO_PATH_STR[model_type]


def get_pth_name(
    model_type: ModelType,
    label: str = "",
    epoch_str: str = "latest",
    ext_name: str = ".pth",
) -> str:
    label = label.strip().lower()
    epoch_str = epoch_str.strip().lower()
    model_type_str = model_type_to_path_str(model_type)
    label_part = f"_{label}" if label else ""
    epoch_part = f"_{epoch_str}" if epoch_str else ""
    return f"{model_type_str}{label_part}{epoch_part}{ext_name}"
