"""导出 ONNX, 供推理端 (C++ ncnn / Rust onnxruntime) 替换 v1 的"3 个独立模型".

输入: best.pt (或任意 CaptchaTripleHeadCNN 权重)
输出: model.onnx, 输入 (1, 1, H, W) in [0, 1], 输出 3 个 logits.
支持导出 fp32 / fp16.

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


def resolve_export_device(device_name: str, precision: str) -> torch.device:
    if device_name == "auto":
        if precision == "fp16":
            if not torch.cuda.is_available():
                raise SystemExit("导出 fp16 ONNX 需要 CUDA 环境, 或改为 --precision fp32")
            return torch.device("cuda")
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("指定了 --device cuda, 但当前 CUDA 不可用")
        return torch.device("cuda")
    return torch.device("cpu")


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
    p.add_argument("--precision", choices=("fp16", "fp32"), default="fp32")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = p.parse_args()

    device = resolve_export_device(args.device, args.precision)
    export_dtype = torch.float16 if args.precision == "fp16" else torch.float32

    model = build_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()
    print(
        f"[model-stats] "
        f"{format_model_stats(collect_model_stats(model, args.image_size_h, args.image_size_w))}"
    )
    wrapper = ExportWrapper(model).to(device=device, dtype=export_dtype)
    if export_dtype == torch.float16:
        wrapper = wrapper.half()
    else:
        wrapper = wrapper.float()

    dummy = torch.randn(1, 1, args.image_size_h, args.image_size_w, device=device, dtype=export_dtype)
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
