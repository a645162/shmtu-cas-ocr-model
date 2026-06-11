"""导出 TorchScript, 供 pnnx 转 ncnn 使用.

输入: best.pt (或任意 CaptchaTriSlotDecoderCNN 权重)
输出: traced .pt, 输入 (1, 1, H, W) float32 in [0, 1], 输出 3 个 logits tuple.

用法:
    python -m cas_ocr_model.trainer.export_torchscript \\
        --checkpoint ./runs/exp1/best.pt \\
        --output ./runs/exp1/model.ts.pt \\
        --image-size-h 64 --image-size-w 192
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from cas_ocr_model.common.console import tag_print
from cas_ocr_model.model.stats import collect_model_stats, format_model_stats

from .model import build_model_from_checkpoint


class ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.model(x)
        return out["digit_left_logits"], out["operator_logits"], out["digit_right_logits"]


def main() -> None:
    p = argparse.ArgumentParser(description="导出 captcha 模型 TorchScript")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, default="./model.ts.pt")
    p.add_argument("--image-size-h", type=int, default=64)
    p.add_argument("--image-size-w", type=int, default=192)
    args = p.parse_args()

    model = build_model_from_checkpoint(args.checkpoint, device="cpu")
    model.eval()
    tag_print(
        "model-stats",
        f"{format_model_stats(collect_model_stats(model, args.image_size_h, args.image_size_w))}",
    )

    wrapper = ExportWrapper(model)
    wrapper.eval()
    dummy = torch.randn(1, 1, args.image_size_h, args.image_size_w, dtype=torch.float32)
    traced = torch.jit.trace(wrapper, dummy, strict=True)
    traced.save(args.output)
    tag_print("export", f"saved -> {args.output}")


if __name__ == "__main__":
    main()
