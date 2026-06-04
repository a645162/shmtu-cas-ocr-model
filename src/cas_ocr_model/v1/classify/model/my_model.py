from cas_ocr_model.v1.classify.model.model_type import ModelType

from torch import nn
from torchvision import models


def init_model(
        model_type: ModelType,
        output_features_count: int,
        pretrained: bool = False
):
    """
    初始化模型
    :param model_type: 模型类型
    :param output_features_count: 输出特征数量
    :param pretrained: 是否使用预训练模型
    :return: 初始化后的模型
    """
    if model_type == ModelType.ResNet_18:
        model = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
            if pretrained else None
        )
    elif model_type == ModelType.ResNet_34:
        model = models.resnet34(
            weights=models.ResNet34_Weights.IMAGENET1K_V1
            if pretrained else None
        )
    elif model_type == ModelType.ResNet_50:
        model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
            if pretrained else None
        )
    elif model_type == ModelType.ResNet_101:
        model = models.resnet101(
            weights=models.ResNet101_Weights.IMAGENET1K_V1
            if pretrained else None
        )
    else:
        raise ValueError(f'不支持的模型类型: {model_type}')

    model.fc = nn.Linear(model.fc.in_features, output_features_count)

    return model
