"""Backbone 工厂: 把 torchvision / timm backbone 改造成接收 1-通道灰度图.

当前默认实现保留原有 torchvision:
    * resnet18 / resnet34 / mobilenet_v3_small / mobilenet_v3_large

新增:
    * resnet50 / r50
    * resnet101 / r101
    * mobilenetv3_small_050 / _075 / _100
    * mobilenetv3_large_075 / _100 / _150d
    * mobilenetv4_conv_small / medium / large / hybrid_medium / hybrid_large 等
    * mobilenetv3_rw
    * repvgg_a0 / a1 / a2 / b0 / b1 / b1g4 / b2 / b2g4 / b3 / b3g4 / d2se
    * 任意 ``timm/<model_name>`` 形式的 timm features_only backbone

设计原则:
    * backbone 输出固定维度 (B, feat_dim, H', W') 的空间特征图
    * 第一层 conv 改成 in_channels=1, 或直接让 timm 以 in_chans=1 构造
    * 旧 backbone 名称保持不变, 避免影响已有 checkpoint
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn
import timm
from torchvision import models

TIMM_DYNAMIC_PREFIX = "timm/"
TIMM_DYNAMIC_HINT = "timm/<model_name>"


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


class _SelectLastFeature(nn.Module):
    """把 timm features_only 输出的多尺度特征压成最后一层空间特征图."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if isinstance(feats, (list, tuple)):
            if not feats:
                raise RuntimeError("timm backbone returned empty feature list")
            return feats[-1]
        return feats


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


def _spatialize_mobilenet_v3(net: nn.Module) -> Tuple[nn.Module, int]:
    first_conv = net.features[0][0]
    if not isinstance(first_conv, nn.Conv2d):
        raise TypeError("unexpected MobileNetV3 stem layout")
    net.features[0][0] = _to_grayscale_conv(first_conv)
    feat_dim = net.classifier[0].in_features
    return net.features, feat_dim


def _build_timm_spatial_backbone(model_name: str, pretrained: bool) -> Tuple[nn.Module, int]:
    backbone = timm.create_model(
        model_name,
        pretrained=pretrained,
        in_chans=1,
        features_only=True,
    )
    if not hasattr(backbone, "feature_info"):
        raise TypeError(f"timm backbone {model_name} does not expose feature_info")
    channels = backbone.feature_info.channels()
    if not channels:
        raise RuntimeError(f"timm backbone {model_name} returned empty feature_info")
    feat_dim = int(channels[-1])
    return _SelectLastFeature(backbone), feat_dim


def _register_timm_alias(alias: str, model_name: str) -> None:
    @_register(alias)
    def _factory(pretrained: bool, *, _model_name: str = model_name) -> Tuple[nn.Module, int]:
        return _build_timm_spatial_backbone(_model_name, pretrained)


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


@_register("mobilenet_v3_small")
def _mobilenet_v3_small(pretrained: bool) -> Tuple[nn.Module, int]:
    net = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    )
    return _spatialize_mobilenet_v3(net)


@_register("mobilenet_v3_large")
def _mobilenet_v3_large(pretrained: bool) -> Tuple[nn.Module, int]:
    net = models.mobilenet_v3_large(
        weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    )
    return _spatialize_mobilenet_v3(net)


for _alias, _model_name in (
    ("resnet50", "resnet50"),
    ("r50", "resnet50"),
    ("resnet101", "resnet101"),
    ("r101", "resnet101"),
    ("mobilenetv3_small_050", "mobilenetv3_small_050"),
    ("mobilenetv3_small_075", "mobilenetv3_small_075"),
    ("mobilenetv3_small_100", "mobilenetv3_small_100"),
    ("mobilenetv3_large_075", "mobilenetv3_large_075"),
    ("mobilenetv3_large_100", "mobilenetv3_large_100"),
    ("mobilenetv3_large_150d", "mobilenetv3_large_150d"),
    ("mobilenetv3_rw", "mobilenetv3_rw"),
    ("mobilenetv4_conv_small", "mobilenetv4_conv_small"),
    ("mobilenetv4_conv_small_035", "mobilenetv4_conv_small_035"),
    ("mobilenetv4_conv_small_050", "mobilenetv4_conv_small_050"),
    ("mobilenetv4_conv_medium", "mobilenetv4_conv_medium"),
    ("mobilenetv4_conv_large", "mobilenetv4_conv_large"),
    ("mobilenetv4_conv_aa_medium", "mobilenetv4_conv_aa_medium"),
    ("mobilenetv4_conv_aa_large", "mobilenetv4_conv_aa_large"),
    ("mobilenetv4_conv_blur_medium", "mobilenetv4_conv_blur_medium"),
    ("mobilenetv4_hybrid_medium", "mobilenetv4_hybrid_medium"),
    ("mobilenetv4_hybrid_medium_075", "mobilenetv4_hybrid_medium_075"),
    ("mobilenetv4_hybrid_large", "mobilenetv4_hybrid_large"),
    ("mobilenetv4_hybrid_large_075", "mobilenetv4_hybrid_large_075"),
    ("repvgg_a0", "repvgg_a0"),
    ("repvgg_a1", "repvgg_a1"),
    ("repvgg_a2", "repvgg_a2"),
    ("repvgg_b0", "repvgg_b0"),
    ("repvgg_b1", "repvgg_b1"),
    ("repvgg_b1g4", "repvgg_b1g4"),
    ("repvgg_b2", "repvgg_b2"),
    ("repvgg_b2g4", "repvgg_b2g4"),
    ("repvgg_b3", "repvgg_b3"),
    ("repvgg_b3g4", "repvgg_b3g4"),
    ("repvgg_d2se", "repvgg_d2se"),
):
    _register_timm_alias(_alias, _model_name)


# ----------------------------------------------------------------------------
# 公共 API
# ----------------------------------------------------------------------------


def is_supported_backbone(name: str) -> bool:
    if name in _BACKBONE_REGISTRY:
        return True
    if not name.startswith(TIMM_DYNAMIC_PREFIX):
        return False
    model_name = name[len(TIMM_DYNAMIC_PREFIX):].strip()
    if not model_name:
        return False
    return model_name in timm.list_models()


def resolve_backbone_name(name: str) -> str:
    raw = str(name).strip()
    if not raw:
        raise ValueError("backbone name must not be empty")
    return raw


def build_resnet_backbone(
    name: str = "resnet18",
    pretrained: bool = True,
) -> Tuple[nn.Module, int]:
    """按名称构造 backbone.

    Args:
        name:
            * registry 内置名称, 如 ``resnet18`` / ``resnet34`` / ``r50`` /
              ``resnet101`` / ``mobilenetv3_large_100``
            * 或动态 timm 名称: ``timm/<model_name>``
        pretrained: 是否加载 ImageNet 预训练权重

    Returns:
        (model, feat_dim). model 输出空间特征图, 直接接位置感知 head 即可.
    """
    name = resolve_backbone_name(name)
    if name in _BACKBONE_REGISTRY:
        return _BACKBONE_REGISTRY[name](pretrained)
    if name.startswith(TIMM_DYNAMIC_PREFIX):
        model_name = name[len(TIMM_DYNAMIC_PREFIX):].strip()
        if not model_name:
            raise ValueError(f"invalid dynamic backbone: {name}")
        if model_name not in timm.list_models():
            raise ValueError(
                f"unknown timm backbone: {model_name}; "
                f"hint=use one of timm.list_models(), e.g. {TIMM_DYNAMIC_HINT}"
            )
        return _build_timm_spatial_backbone(model_name, pretrained)
    raise ValueError(
        f"unknown backbone: {name}; available={list_available_backbones()}"
    )


def list_available_backbones() -> list[str]:
    names = sorted(_BACKBONE_REGISTRY.keys())
    return names + [TIMM_DYNAMIC_HINT]
