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

from .config import NUM_DIGIT_CLASSES, NUM_OPERATOR_CLASSES
from .model import CaptchaTripleHeadCNN, load_checkpoint


def main() -> None:
    p = argparse.ArgumentParser(description="导出 captcha 模型 ONNX")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, default="./model.onnx")
    p.add_argument("--backbone", type=str, default="resnet18")
    p.add_argument("--image-size-h", type=int, default=64)
    p.add_argument("--image-size-w", type=int, default=192)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--dynamic-batch", action="store_true", help="动态 batch 维度")
    args = p.parse_args()

    model = CaptchaTripleHeadCNN(
        backbone=args.backbone,
        pretrained=False,
        num_digit_classes=NUM_DIGIT_CLASSES,
        num_operator_classes=NUM_OPERATOR_CLASSES,
    )
    load_checkpoint(model, args.checkpoint, device="cpu")
    model.eval()

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
        model,
        dummy,
        args.output,
        input_names=["input"],
        output_names=["digit_left_logits", "operator_logits", "digit_right_logits"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"[export] saved -> {args.output}")


if __name__ == "__main__":
    main()
