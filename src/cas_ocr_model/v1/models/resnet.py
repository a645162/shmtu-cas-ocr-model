"""ResNet 工厂：从 torchvision 加载预训练 backbone 并替换 fc。"""

from __future__ import annotations

from torch import nn
from torchvision import models

from .registry import ModelType


def init_model(
    model_type: ModelType,
    output_features_count: int,
    pretrained: bool = False,
) -> nn.Module:
    """
    初始化模型 backbone + 替换分类头。

    Args:
        model_type: ResNet 变体。
        output_features_count: 输出维度（分类数）。
        pretrained: 是否加载 ImageNet 预训练权重。

    Returns:
        替换 fc 后的 nn.Module。
    """
    if model_type == ModelType.ResNet_18:
        model = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
    elif model_type == ModelType.ResNet_34:
        model = models.resnet34(
            weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        )
    elif model_type == ModelType.ResNet_50:
        model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )
    elif model_type == ModelType.ResNet_101:
        model = models.resnet101(
            weights=models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.fc = nn.Linear(model.fc.in_features, output_features_count)
    return model
