"""PyTorch 推理后端.

加载 trainer 训练出的 CaptchaTripleHeadCNN 权重, 单图/批量推理.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from cas_ocr_model.model import build_model_from_checkpoint
from cas_ocr_model.trainer.config import (
    DIGIT_LABELS,
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
    ) -> None:
        del backbone
        self.device = torch.device(device)
        self.model = build_model_from_checkpoint(str(checkpoint), device=self.device)
        self.model.eval()

    @torch.no_grad()
    def infer(self, image: torch.Tensor) -> dict[str, np.ndarray]:
        """image: (B, 1, H, W) float32 in [0, 1], 已 preprocess.

        返回 dict, 3 个 logits 数组:
            digit_left_logits:  (B, 10) float32
            operator_logits:    (B, 3)  float32
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
