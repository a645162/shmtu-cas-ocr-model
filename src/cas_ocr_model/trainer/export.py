"""导出 ONNX, 供推理端 (C++ ncnn / Rust onnxruntime) 替换 v1 的"3 个独立模型".

输入: best.pt (或任意 CaptchaTripleHeadCNN 权重)
输出: model.onnx, 输入 (1, 1, H, W) float32 in [0, 1], 输出 3 个 logits.

用法:
    python -m cas_ocr_model.trainer.export \\
        --checkpoint ./runs/exp1/best.pt \\
        --output ./runs/exp1/model.onnx \\
        --image-size-h 64 --image-size-w 192
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from .model import build_model_from_checkpoint
from cas_ocr_model.model.stats import collect_model_stats, format_model_stats


class ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.model(x)
        return out["digit_left_logits"], out["operator_logits"], out["digit_right_logits"]


def main() -> None:
    p = argparse.ArgumentParser(description="导出 captcha 模型 ONNX")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, default="./model.onnx")
    p.add_argument("--backbone", type=str, default="resnet18")
    p.add_argument("--image-size-h", type=int, default=64)
    p.add_argument("--image-size-w", type=int, default=192)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--dynamic-batch", action="store_true", help="动态 batch 维度")
    p.add_argument(
        "--legacy-exporter",
        action="store_true",
        help="使用旧版 TorchScript ONNX 导出器 (dynamo=False)，兼容性通常更稳",
    )
    args = p.parse_args()

    model = build_model_from_checkpoint(args.checkpoint, device="cpu")
    model.eval()
    print(
        f"[model-stats] "
        f"{format_model_stats(collect_model_stats(model, args.image_size_h, args.image_size_w))}"
    )
    wrapper = ExportWrapper(model)

    dummy = torch.randn(1, 1, args.image_size_h, args.image_size_w, dtype=torch.float32)
    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch"},
            "digit_left_logits": {0: "batch"},
            "operator_logits": {0: "batch"},
            "digit_right_logits": {0: "batch"},
        }

    torch.onnx.export(
        wrapper,
        dummy,
        args.output,
        input_names=["input"],
        output_names=["digit_left_logits", "operator_logits", "digit_right_logits"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=not args.legacy_exporter,
    )
    print(f"[export] saved -> {args.output}")


if __name__ == "__main__":
    main()
