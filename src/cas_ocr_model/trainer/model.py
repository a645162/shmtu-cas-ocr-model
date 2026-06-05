"""Backwards-compat re-export.

模型实现已迁移到 ``cas_ocr_model.model`` 子包. 本模块仅保留旧 import 路径
的兼容性 (``from cas_ocr_model.trainer.model import CaptchaTripleHeadCNN``),
所有代码都转发到新的实现.

新代码请直接:
    from cas_ocr_model.model import CaptchaTripleHeadCNN, load_checkpoint, predict_triple
"""
from cas_ocr_model.model import (  # noqa: F401
    CaptchaTripleHeadCNN,
    build_model_from_checkpoint,
    build_resnet_backbone,
    list_available_backbones,
    load_checkpoint,
    predict_triple,
)
from cas_ocr_model.model.heads import TripleHead  # noqa: F401

__all__ = [
    "CaptchaTripleHeadCNN",
    "build_model_from_checkpoint",
    "load_checkpoint",
    "predict_triple",
    "build_resnet_backbone",
    "list_available_backbones",
    "TripleHead",
]
