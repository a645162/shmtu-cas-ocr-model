"""PyTorch 推理后端.

加载 trainer 训练出的 CaptchaTripleHeadCNN 权重, 单图/批量推理.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from cas_ocr_model.model import CaptchaTripleHeadCNN, load_checkpoint as _load_ckpt
from cas_ocr_model.trainer.config import (
    DIGIT_LABELS,
    NUM_DIGIT_CLASSES,
    NUM_OPERATOR_CLASSES,
    OPERATOR_LABELS,
)


class PyTorchBackend:
    """本地 PyTorch 推理后端.

    用法:
        backend = PyTorchBackend("runs/exp1/best.pt", device="cuda")
        logits = backend.infer(image_tensor)  # (B, 1, H, W) float32 in [0,1]
    """

    def __init__(
        self,
        checkpoint: str | Path,
        backbone: str = "resnet18",
        device: str = "cpu",
        num_digit_classes: int = NUM_DIGIT_CLASSES,
        num_operator_classes: int = NUM_OPERATOR_CLASSES,
    ) -> None:
        self.model = CaptchaTripleHeadCNN(
            backbone=backbone,
            pretrained=False,
            num_digit_classes=num_digit_classes,
            num_operator_classes=num_operator_classes,
        )
        self.device = torch.device(device)
        _load_ckpt(self.model, str(checkpoint), device=self.device)
        self.model.eval()

    @torch.no_grad()
    def infer(self, image: torch.Tensor) -> dict[str, np.ndarray]:
        """image: (B, 1, H, W) float32 in [0, 1], 已 preprocess.

        返回 dict, 3 个 logits 数组:
            digit_left_logits:  (B, 10) float32
            operator_logits:    (B, 4)  float32
            digit_right_logits: (B, 10) float32
        """
        image = image.to(self.device, non_blocking=True)
        out = self.model(image)
        return {
            "digit_left_logits": out["digit_left_logits"].cpu().numpy(),
            "operator_logits": out["operator_logits"].cpu().numpy(),
            "digit_right_logits": out["digit_right_logits"].cpu().numpy(),
        }

    @property
    def label_dict(self) -> dict[str, list[str]]:
        return {"digit_labels": DIGIT_LABELS, "operator_labels": OPERATOR_LABELS}
