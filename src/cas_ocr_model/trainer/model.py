"""Backwards-compat re-export.

模型实现已迁移到 ``cas_ocr_model.model`` 子包. 本模块仅保留旧 import 路径
的兼容性 (``from cas_ocr_model.trainer.model import CaptchaTriSlotDecoderCNN``),
所有代码都转发到新的实现.

新代码请直接:
    from cas_ocr_model.model import CaptchaTriSlotDecoderCNN, load_checkpoint, predict_triple
"""
from cas_ocr_model.model import (  # noqa: F401
    CaptchaTriSlotDecoderCNN,
    build_model_from_checkpoint,
    build_model_from_config,
    build_model_metadata,
    build_release_asset_stem,
    build_resnet_backbone,
    find_release_checkpoint,
    get_model_spec,
    inspect_checkpoint,
    list_available_backbones,
    list_model_versions,
    load_checkpoint,
    normalize_model_version,
    predict_triple,
)
from cas_ocr_model.model.heads import TriSlotDecoder  # noqa: F401

__all__ = [
    "CaptchaTriSlotDecoderCNN",
    "build_model_from_config",
    "build_model_from_checkpoint",
    "build_model_metadata",
    "build_release_asset_stem",
    "find_release_checkpoint",
    "load_checkpoint",
    "predict_triple",
    "get_model_spec",
    "inspect_checkpoint",
    "list_model_versions",
    "normalize_model_version",
    "build_resnet_backbone",
    "list_available_backbones",
    "TriSlotDecoder",
]
