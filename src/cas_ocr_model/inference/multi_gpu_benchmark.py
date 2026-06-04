"""多卡 DDP 精度 benchmark.

通过 accelerate 启动, 8 卡并行推理 test 集, 计算指标 (acc / 混淆矩阵 / ECE).

调用:
    accelerate launch --num_processes 8 --mixed_precision fp16 \\
        -m cas_ocr_model.inference.multi_gpu_benchmark \\
        --backend pytorch --checkpoint runs/exp1/best.pt \\
        --data-root ./dataset --output report.json

注: 想要单卡速度请用 single_gpu_benchmark.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from accelerate import Accelerator

from .inference import CaptchaInferencer, InferencerConfig
from .evaluate import evaluate, metrics_to_dict


def build_backend(args: argparse.Namespace):
    if args.backend == "pytorch":
        from .backends.pytorch_backend import PyTorchBackend
        return PyTorchBackend(
            checkpoint=args.checkpoint,
            backbone=args.backbone,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    if args.backend == "onnx":
        from .backends.onnx_backend import OnnxBackend
        return OnnxBackend(
            onnx_path=args.checkpoint,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    raise ValueError(f"unknown backend: {args.backend}")


def main() -> None:
    p = argparse.ArgumentParser(description="多卡 DDP 精度 benchmark (test 集)")
    p.add_argument("--backend", choices=["pytorch", "onnx"], default="pytorch")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--data-root", required=True, help="含 manifest.json + test split 的目录")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--image-size-h", type=int, default=64)
    p.add_argument("--image-size-w", type=int, default=192)
    p.add_argument("--threshold", type=int, default=200)
    args = p.parse_args()

    accelerator = Accelerator()
    backend = build_backend(args)
    cfg = InferencerConfig(
        image_size_h=args.image_size_h,
        image_size_w=args.image_size_w,
        threshold=args.threshold,
        batch_size=64,
    )
    inferencer = CaptchaInferencer(backend=backend, config=cfg)

    accelerator.print(
        f"[multi-gpu-bench] world_size={accelerator.num_processes} "
        f"rank={accelerator.process_index} device={accelerator.device}"
    )

    metrics = evaluate(
        inferencer,
        dataset_dir=args.data_root,
        pattern="*.jpg",
        limit=args.limit,
    )
    accelerator.print(
        f"[multi-gpu-bench][rank={accelerator.process_index}] "
        f"n_seen_local={metrics.n_samples} acc_full={metrics.acc_expression:.4f}"
    )

    if accelerator.is_main_process:
        report = metrics_to_dict(metrics)
        report["benchmark"] = {
            "mode": "multi-gpu-ddp-precision",
            "world_size": accelerator.num_processes,
            "n_samples_local": metrics.n_samples,
        }
        if args.output:
            Path(args.output).write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            accelerator.print(f"[saved] {args.output}")
        else:
            print(json.dumps(report, ensure_ascii=False, indent=2))

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
