"""模型版本 2.0.

当前版本仍然是 tri-slot-decoder 主干:
    backbone -> TriSlotDecoder -> left digit / operator / right digit
"""
from __future__ import annotations

from cas_ocr_model.model.captcha_trislot_decoder_cnn import CaptchaTriSlotDecoderCNN

MODEL_VERSION = "2.0"
MODEL_FAMILY = "trislot_decoder"


def build_model(
    *,
    backbone: str = "resnet18",
    pretrained: bool = True,
    dropout: float = 0.2,
    slot_hidden_dim: int = 256,
    slot_attention_heads: int = 4,
    num_digit_classes: int = 10,
    num_operator_classes: int = 3,
) -> CaptchaTriSlotDecoderCNN:
    model = CaptchaTriSlotDecoderCNN(
        backbone=backbone,
        pretrained=pretrained,
        dropout=dropout,
        slot_hidden_dim=slot_hidden_dim,
        slot_attention_heads=slot_attention_heads,
        num_digit_classes=num_digit_classes,
        num_operator_classes=num_operator_classes,
    )
    model.model_version = MODEL_VERSION
    model.model_family = MODEL_FAMILY
    return model
