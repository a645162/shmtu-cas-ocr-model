"""Backbone 工厂: 把 torchvision ResNet 改造成"接收 1-通道灰度图".

当前实现: ResNet-18 / ResNet-34. 输出保留空间特征图, 便于后续按宽度做槽位解码.

设计原则:
    * backbone 输出固定维度 (B, feat_dim, H', W') 的空间特征图
    * 第一层 conv 改成 in_channels=1 (验证码是灰度)
    * 预训练 RGB conv1 权重按通道均值迁移到灰度 conv1
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch
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


def _to_grayscale_conv(conv: nn.Conv2d) -> nn.Conv2d:
    gray = nn.Conv2d(
        1,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=False,
    )
    with torch.no_grad():
        gray.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
    return gray


def _spatialize_resnet(net: nn.Module) -> Tuple[nn.Module, int]:
    feat_dim = net.fc.in_features
    net.conv1 = _to_grayscale_conv(net.conv1)
    features = nn.Sequential(*list(net.children())[:-2])
    return features, feat_dim


@_register("resnet18")
def _resnet18(pretrained: bool) -> Tuple[nn.Module, int]:
    net = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    )
    return _spatialize_resnet(net)


@_register("resnet34")
def _resnet34(pretrained: bool) -> Tuple[nn.Module, int]:
    net = models.resnet34(
        weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
    )
    return _spatialize_resnet(net)


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
        (model, feat_dim). model 输出空间特征图, 直接接位置感知 head 即可.
    """
    if name not in _BACKBONE_REGISTRY:
        raise ValueError(
            f"unknown backbone: {name}; available={list_available_backbones()}"
        )
    return _BACKBONE_REGISTRY[name](pretrained)


def list_available_backbones() -> list[str]:
    return list(_BACKBONE_REGISTRY.keys())
