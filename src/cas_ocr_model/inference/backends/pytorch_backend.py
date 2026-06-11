"""PyTorch 推理后端.

加载 trainer 训练出的 CaptchaTriSlotDecoderCNN 权重, 单图/批量推理.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from cas_ocr_model.common.console import tag_print
from cas_ocr_model.model import build_model_from_checkpoint
from cas_ocr_model.model.stats import collect_model_stats, format_model_stats
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
        raw = torch.load(checkpoint, map_location="cpu")
        cfg = raw.get("config", {}) if isinstance(raw, dict) else {}
        data_cfg = cfg.get("data", {})
        image_size_h = int(data_cfg.get("image_size_h", 64))
        image_size_w = int(data_cfg.get("image_size_w", 192))
        self.device = torch.device(device)
        self.model = build_model_from_checkpoint(str(checkpoint), device=self.device)
        self.model.eval()
        tag_print(
            "model-stats",
            f"{format_model_stats(collect_model_stats(self.model, image_size_h, image_size_w))}",
        )

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
