"""位置感知 TriSlot Decoder."""
from __future__ import annotations

import torch
import torch.nn as nn


class TriSlotDecoder(nn.Module):
    """把空间特征图按宽度解码为 digit / operator / digit 三个槽位."""

    def __init__(
        self,
        feat_dim: int,
        num_digit_classes: int = 10,
        num_operator_classes: int = 3,
        hidden_dim: int = 256,
        attention_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_dim % attention_heads != 0:
            raise ValueError("hidden_dim 必须能被 attention_heads 整除")

        self.project = nn.Sequential(
            nn.Conv2d(feat_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        self.position_conv = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim
        )
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.slot_queries = nn.Parameter(torch.randn(3, hidden_dim))
        self.slot_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.slot_norm = nn.LayerNorm(hidden_dim)
        self.slot_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(p=dropout)
        self.head_digit_left = nn.Linear(hidden_dim, num_digit_classes)
        self.head_operator = nn.Linear(hidden_dim, num_operator_classes)
        self.head_digit_right = nn.Linear(hidden_dim, num_digit_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.slot_queries, std=0.02)
        for head in (self.head_digit_left, self.head_operator, self.head_digit_right):
            nn.init.normal_(head.weight, std=0.01)
            nn.init.zeros_(head.bias)

    def _build_tokens(self, feat_map: torch.Tensor) -> torch.Tensor:
        feat_map = self.project(feat_map)
        tokens = feat_map.mean(dim=2).transpose(1, 2)
        pos = self.position_conv(tokens.transpose(1, 2)).transpose(1, 2)
        return self.token_norm(tokens + pos)

    def forward(
        self,
        feat_map: torch.Tensor,
        return_aux: bool = False,
    ) -> dict[str, torch.Tensor]:
        tokens = self._build_tokens(feat_map)
        batch = tokens.size(0)
        queries = self.slot_queries.unsqueeze(0).expand(batch, -1, -1)
        slot_feat, attn = self.slot_attn(
            query=queries,
            key=tokens,
            value=tokens,
            need_weights=return_aux,
            average_attn_weights=False,
        )
        slot_feat = self.slot_norm(slot_feat + self.slot_ffn(slot_feat))
        slot_feat = self.dropout(slot_feat)

        outputs = {
            "digit_left_logits": self.head_digit_left(slot_feat[:, 0]),
            "operator_logits": self.head_operator(slot_feat[:, 1]),
            "digit_right_logits": self.head_digit_right(slot_feat[:, 2]),
        }
        if not return_aux:
            return outputs

        attn = attn.mean(dim=1)
        width = attn.size(-1)
        positions = torch.linspace(0.0, 1.0, width, device=attn.device, dtype=attn.dtype)
        slot_centers = (attn * positions.view(1, 1, -1)).sum(dim=-1)
        outputs["slot_attention"] = attn
        outputs["slot_centers"] = slot_centers
        return outputs

