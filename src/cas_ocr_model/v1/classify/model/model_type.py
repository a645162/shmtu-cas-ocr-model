from enum import Enum


class ModelType(Enum):
    """模型类型"""
    ResNet_18 = 1
    ResNet_34 = 2
    ResNet_50 = 3
    ResNet_101 = 4


def model_type_to_path_str(model_type: ModelType):
    """
    将模型类型转换为路径字符串
    :param model_type: 模型类型
    :return: 路径字符串
    """
    if model_type == ModelType.ResNet_18:
        return 'resnet18'
    elif model_type == ModelType.ResNet_34:
        return 'resnet34'
    elif model_type == ModelType.ResNet_50:
        return 'resnet50'
    elif model_type == ModelType.ResNet_101:
        return 'resnet101'
    else:
        raise ValueError(f'不支持的模型类型: {model_type}')


def get_pth_name(
        model_type: ModelType,
        label: str,
        epoch_str: str = "latest",
        ext_name: str = ".pth"
):
    """
    获取模型的pth文件名
    :param model_type: 模型类型
    :param label: 标签
    :param epoch_str: epoch字符串
    :return: pth文件名
    """
    label = label.strip().lower()
    epoch_str = epoch_str.strip().lower()

    model_type_str: str = model_type_to_path_str(model_type)
    if len(label) > 0:
        label = f"_{label}"
    if len(epoch_str) > 0:
        epoch_str = f"_{epoch_str}"
    return f'{model_type_str}{label}{epoch_str}{ext_name}'
