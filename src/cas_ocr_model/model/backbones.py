"""Backbone 工厂: 把 torchvision ResNet 改造成"接收 1-通道灰度图".

当前实现: ResNet-18 / ResNet-34. 后续可加入 ConvNeXt / EfficientNet / ViT 等.

设计原则:
    * backbone 输出固定维度 (B, feat_dim) 的全局特征向量
    * 第一层 conv 改成 in_channels=1 (验证码是灰度)
    * 原 fc 层替换为 nn.Identity (下游 head 自己接)
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch.nn as nn
from torchvision import models


# ----------------------------------------------------------------------------
# 注册表
# ----------------------------------------------------------------------------


_BACKBONE_REGISTRY: Dict[str, Callable[[bool], Tuple[nn.Module, int]]] = {}


def _register(name: str):
    """注册 backbone 工厂. factory(pretrained) -> (model, feat_dim)."""

    def deco(factory: Callable[[bool], Tuple[nn.Module, int]]) -> Callable[[bool], Tuple[nn.Module, int]]:
        _BACKBONE_REGISTRY[name] = factory
        return factory

    return deco


# ----------------------------------------------------------------------------
# 具体实现
# ----------------------------------------------------------------------------


@_register("resnet18")
def _resnet18(pretrained: bool) -> Tuple[nn.Module, int]:
    net = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    )
    feat_dim = net.fc.in_features  # 512
    net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    net.fc = nn.Identity()
    return net, feat_dim


@_register("resnet34")
def _resnet34(pretrained: bool) -> Tuple[nn.Module, int]:
    net = models.resnet34(
        weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
    )
    feat_dim = net.fc.in_features  # 512
    net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    net.fc = nn.Identity()
    return net, feat_dim


# 未来可加:
#   @_register("resnet50")
#   @_register("convnext_tiny")
#   @_register("efficientnet_b0")
#   @_register("vit_small_patch16")


# ----------------------------------------------------------------------------
# 公共 API
# ----------------------------------------------------------------------------


def build_resnet_backbone(
    name: str = "resnet18",
    pretrained: bool = True,
) -> Tuple[nn.Module, int]:
    """按名称构造 backbone.

    Args:
        name: ``"resnet18"`` / ``"resnet34"`` (见 list_available_backbones)
        pretrained: 是否加载 ImageNet 预训练权重

    Returns:
        (model, feat_dim). model 已替换 conv1 / fc, 直接接 head 即可.
    """
    if name not in _BACKBONE_REGISTRY:
        raise ValueError(
            f"unknown backbone: {name}; available={list_available_backbones()}"
        )
    return _BACKBONE_REGISTRY[name](pretrained)


def list_available_backbones() -> list[str]:
    return list(_BACKBONE_REGISTRY.keys())
