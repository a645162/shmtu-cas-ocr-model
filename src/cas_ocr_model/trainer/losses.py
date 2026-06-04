"""3-head 联合损失 + 评估指标.

L = w_dl * CE(logits_dl, y_dl) + w_op * CE(logits_op, y_op) + w_dr * CE(logits_dr, y_dr)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class LossWeights:
    digit_left: float = 1.0
    operator: float = 1.0
    digit_right: float = 1.0


class TripleHeadLoss(nn.Module):
    """3 个独立 CE, 标量加权求和."""

    def __init__(self, weights: LossWeights | None = None, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        self.label_smoothing = label_smoothing
        # reduction='mean' 内部按 batch 取均值, 然后加权求和.
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        outputs: 来自 CaptchaTripleHeadCNN.forward(), 含 3 个 logits
        targets: 来自 collate_triple, 含 3 个 long tensor
        """
        loss_dl = self.ce(outputs["digit_left_logits"], targets["digit_left"])
        loss_op = self.ce(outputs["operator_logits"], targets["operator"])
        loss_dr = self.ce(outputs["digit_right_logits"], targets["digit_right"])

        total = (
            self.weights.digit_left * loss_dl
            + self.weights.operator * loss_op
            + self.weights.digit_right * loss_dr
        )

        # 标量拆解, 方便日志
        return {
            "loss": total,
            "loss_digit_left": loss_dl.detach(),
            "loss_operator": loss_op.detach(),
            "loss_digit_right": loss_dr.detach(),
        }


# ----------------------------------------------------------------------------
# 评估指标
# ----------------------------------------------------------------------------


@torch.no_grad()
def compute_accuracy(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> dict[str, float]:
    """3 个 head 的 top-1 准确率. 额外给出 3-head 全对的 'expression_acc'."""
    pred_dl = outputs["digit_left_logits"].argmax(dim=-1)
    pred_op = outputs["operator_logits"].argmax(dim=-1)
    pred_dr = outputs["digit_right_logits"].argmax(dim=-1)

    acc_dl = (pred_dl == targets["digit_left"]).float().mean().item()
    acc_op = (pred_op == targets["operator"]).float().mean().item()
    acc_dr = (pred_dr == targets["digit_right"]).float().mean().item()

    full = (pred_dl == targets["digit_left"]) & (pred_op == targets["operator"]) & (
        pred_dr == targets["digit_right"]
    )
    acc_full = full.float().mean().item()

    return {
        "acc_digit_left": acc_dl,
        "acc_operator": acc_op,
        "acc_digit_right": acc_dr,
        "acc_expression": acc_full,
    }
