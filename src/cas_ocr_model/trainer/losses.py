"""3-head 联合损失与指标."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossWeights:
    digit_left: float = 1.0
    operator: float = 1.0
    digit_right: float = 1.0
    slot_order: float = 0.1
    slot_overlap: float = 0.05


class TripleHeadLoss(nn.Module):
    """分类损失 + 槽位结构约束."""

    def __init__(
        self,
        weights: LossWeights | None = None,
        label_smoothing: float = 0.0,
        focal_gamma: float = 0.0,
        slot_margin: float = 0.10,
    ) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma
        self.slot_margin = slot_margin

    def _classification_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        num_classes = logits.size(-1)

        if self.label_smoothing > 0:
            smooth = self.label_smoothing / num_classes
            target_dist = torch.full_like(log_probs, smooth)
            target_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing + smooth)
            loss = -(target_dist * log_probs).sum(dim=-1)
        else:
            loss = F.nll_loss(log_probs, targets, reduction="none")

        if self.focal_gamma > 0:
            pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1).exp()
            loss = ((1.0 - pt).clamp_min(0.0) ** self.focal_gamma) * loss
        return loss.mean()

    def _slot_order_loss(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        centers = outputs.get("slot_centers")
        if centers is None:
            return outputs["digit_left_logits"].new_zeros(())
        left_gap = F.relu(self.slot_margin - (centers[:, 1] - centers[:, 0]))
        right_gap = F.relu(self.slot_margin - (centers[:, 2] - centers[:, 1]))
        return (left_gap + right_gap).mean()

    def _slot_overlap_loss(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        attn = outputs.get("slot_attention")
        if attn is None:
            return outputs["digit_left_logits"].new_zeros(())
        overlap = torch.bmm(attn, attn.transpose(1, 2))
        eye = torch.eye(overlap.size(-1), device=overlap.device, dtype=overlap.dtype).unsqueeze(0)
        return ((overlap - eye) * (1.0 - eye)).pow(2).mean()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        loss_dl = self._classification_loss(outputs["digit_left_logits"], targets["digit_left"])
        loss_op = self._classification_loss(outputs["operator_logits"], targets["operator"])
        loss_dr = self._classification_loss(outputs["digit_right_logits"], targets["digit_right"])
        loss_order = self._slot_order_loss(outputs)
        loss_overlap = self._slot_overlap_loss(outputs)

        total = (
            self.weights.digit_left * loss_dl
            + self.weights.operator * loss_op
            + self.weights.digit_right * loss_dr
            + self.weights.slot_order * loss_order
            + self.weights.slot_overlap * loss_overlap
        )
        return {
            "loss": total,
            "loss_digit_left": loss_dl.detach(),
            "loss_operator": loss_op.detach(),
            "loss_digit_right": loss_dr.detach(),
            "loss_slot_order": loss_order.detach(),
            "loss_slot_overlap": loss_overlap.detach(),
        }


@torch.no_grad()
def compute_accuracy(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> dict[str, float]:
    pred_dl = outputs["digit_left_logits"].argmax(dim=-1)
    pred_op = outputs["operator_logits"].argmax(dim=-1)
    pred_dr = outputs["digit_right_logits"].argmax(dim=-1)

    acc_dl = (pred_dl == targets["digit_left"]).float().mean().item()
    acc_op = (pred_op == targets["operator"]).float().mean().item()
    acc_dr = (pred_dr == targets["digit_right"]).float().mean().item()
    full = (pred_dl == targets["digit_left"]) & (pred_op == targets["operator"]) & (
        pred_dr == targets["digit_right"]
    )
    return {
        "acc_digit_left": acc_dl,
        "acc_operator": acc_op,
        "acc_digit_right": acc_dr,
        "acc_expression": full.float().mean().item(),
    }
