"""backbone 工厂回归测试."""
from __future__ import annotations

import torch

from cas_ocr_model.model.backbones import (
    build_resnet_backbone,
    is_supported_backbone,
    list_available_backbones,
)


def test_repvgg_aliases_are_listed():
    names = list_available_backbones()
    assert "repvgg_a0" in names
    assert "repvgg_b1" in names
    assert "repvgg_d2se" in names


def test_repvgg_aliases_are_supported():
    assert is_supported_backbone("repvgg_a0") is True
    assert is_supported_backbone("repvgg_b1g4") is True


def test_repvgg_alias_builds_and_runs_forward():
    backbone, feat_dim = build_resnet_backbone("repvgg_a0", pretrained=False)
    x = torch.randn(2, 1, 64, 192)
    y = backbone(x)

    assert feat_dim == 1280
    assert y.shape[0] == 2
    assert y.shape[1] == feat_dim
    assert y.ndim == 4
