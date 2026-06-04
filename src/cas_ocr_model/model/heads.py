"""3 个独立分类头容器.

当前: 共享 backbone 输出, 3 个独立 Linear 头.
未来可加入: 注意力门控 / head 之间的关联 (如 digit/operator 语义互约束) /
            head-specific feature refinement.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TripleHead(nn.Module):
    """(digit_left, operator, digit_right) 三个独立分类头.

    每个头接收同一个 feat 向量, 各自映射到 num_classes.
    """

    def __init__(
        self,
        feat_dim: int,
        num_digit_classes: int = 10,
        num_operator_classes: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.num_digit_classes = num_digit_classes
        self.num_operator_classes = num_operator_classes

        self.dropout = nn.Dropout(p=dropout)
        self.head_digit_left = nn.Linear(feat_dim, num_digit_classes)
        self.head_operator = nn.Linear(feat_dim, num_operator_classes)
        self.head_digit_right = nn.Linear(feat_dim, num_digit_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for head in (self.head_digit_left, self.head_operator, self.head_digit_right):
            nn.init.normal_(head.weight, std=0.01)
            nn.init.zeros_(head.bias)

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        """feat: (B, feat_dim) -> 3 个 logits 字典."""
        feat = self.dropout(feat)
        return {
            "digit_left_logits": self.head_digit_left(feat),
            "operator_logits": self.head_operator(feat),
            "digit_right_logits": self.head_digit_right(feat),
        }
