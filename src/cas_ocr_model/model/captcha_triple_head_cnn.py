"""当前简单实现: ResNet-18/34 backbone + 3-head.

设计原则:
    * 单 CNN 一次前向同时输出 digit_left / operator / digit_right
    * 不再像 v1 那样切 3 段送 3 个独立模型
    * 运算符统一为 4 类 (+, -, *, /), 无等号头

未来可扩展方向 (不破坏当前接口):
    * 替换 backbone 为 ConvNeXt / ViT
    * 引入 attention 融合左右数字特征
    * 算式语义约束 (head 间互注意力)
    * 字符序列建模 (CTC / 自回归)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .backbones import build_resnet_backbone
from .heads import TripleHead


class CaptchaTripleHeadCNN(nn.Module):
    """backbone (灰度 1 通道) + 3-head 分类器."""

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.2,
        num_digit_classes: int = 10,
        num_operator_classes: int = 4,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.num_digit_classes = num_digit_classes
        self.num_operator_classes = num_operator_classes

        self.backbone, feat_dim = build_resnet_backbone(backbone, pretrained=pretrained)
        self.heads = TripleHead(
            feat_dim=feat_dim,
            num_digit_classes=num_digit_classes,
            num_operator_classes=num_operator_classes,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """x: (B, 1, H, W) 灰度图, 像素已归一到 [0, 1].

        返回 dict: {digit_left_logits, operator_logits, digit_right_logits}
        """
        feat = self.backbone(x)
        return self.heads(feat)


# ----------------------------------------------------------------------------
# 推理辅助
# ----------------------------------------------------------------------------


@torch.no_grad()
def predict_triple(
    model: CaptchaTripleHeadCNN,
    image: torch.Tensor,
    operator_labels: list[str],
    digit_labels: list[str],
) -> list[dict[str, str]]:
    """单 batch 推理, 返回 [{digit_left, operator, digit_right, expression}, ...].

    image: (B, 1, H, W), 已 preprocess, 在 model 所在 device 上.
    """
    model.eval()
    out = model(image)
    dl = out["digit_left_logits"].argmax(dim=-1).cpu().tolist()
    op = out["operator_logits"].argmax(dim=-1).cpu().tolist()
    dr = out["digit_right_logits"].argmax(dim=-1).cpu().tolist()

    results: list[dict[str, str]] = []
    for a, b, c in zip(dl, op, dr):
        results.append(
            {
                "digit_left": digit_labels[a],
                "operator": operator_labels[b],
                "digit_right": digit_labels[c],
                "expression": f"{digit_labels[a]}{operator_labels[b]}{digit_labels[c]}",
            }
        )
    return results


def load_checkpoint(
    model: CaptchaTripleHeadCNN,
    ckpt_path: str,
    device: Optional[torch.device] = None,
) -> CaptchaTripleHeadCNN:
    """加载权重. 兼容 DDP 保存的 state_dict (带 'module.' 前缀)."""
    sd = torch.load(ckpt_path, map_location=device or "cpu")
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    if device is not None:
        model.to(device)
    return model
