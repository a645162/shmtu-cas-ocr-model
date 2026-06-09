"""单卡 / 多卡评估入口.

支持:
    * 单 checkpoint 在 val 集上的指标
    * accelerate 多卡一致求值 (prepare 后 evaluate)
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.utils import set_seed

from cas_ocr_model.common.console import AcceleratorConsole
from .config import (
    FullConfig,
    apply_env_overrides,
    load_config,
    load_config_from_checkpoint,
)
from .data import CaptchaPairDataset, collate_triple
from .losses import LossWeights, TriSlotDecoderLoss
from .model import build_model_from_checkpoint
from cas_ocr_model.model.stats import collect_model_stats, format_model_stats
from .train import evaluate


def main() -> None:
    p = argparse.ArgumentParser(description="评估 captcha 模型 (TriSlot Decoder)")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--checkpoint", type=str, required=True, help="best.pt / last.pt 路径")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--per-device-batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--mixed-precision", type=str, default=None)
    args = p.parse_args()

    cfg = load_config(args.config) if args.config else load_config_from_checkpoint(args.checkpoint)
    cfg = apply_env_overrides(cfg)
    cfg.train.per_device_batch_size = args.per_device_batch_size
    cfg.data.num_workers = args.num_workers
    if args.data_root:
        cfg.data.data_root = args.data_root
    if args.mixed_precision:
        cfg.train.mixed_precision = args.mixed_precision

    accelerator = Accelerator(mixed_precision=cfg.train.mixed_precision)
    set_seed(cfg.train.seed, device_specific=True)

    val_ds = CaptchaPairDataset(
        data_root=cfg.data.data_root,
        image_size_h=cfg.data.image_size_h,
        image_size_w=cfg.data.image_size_w,
        threshold=cfg.data.threshold,
        binarize_mode=cfg.data.binarize_mode,
        adaptive_block_size=cfg.data.adaptive_block_size,
        adaptive_c=cfg.data.adaptive_c,
        train=False,
        train_ratio=cfg.data.train_ratio,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.per_device_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        collate_fn=collate_triple,
    )

    model = build_model_from_checkpoint(args.checkpoint, device=accelerator.device)
    console = AcceleratorConsole(accelerator)
    if accelerator.is_main_process:
        console.tag_print(
            "model-stats",
            f"{format_model_stats(collect_model_stats(model, cfg.data.image_size_h, cfg.data.image_size_w))}",
        )
    model = accelerator.prepare(model)

    loss_fn = TriSlotDecoderLoss(
        weights=LossWeights(
            digit_left=cfg.loss.weight_digit_left,
            operator=cfg.loss.weight_operator,
            digit_right=cfg.loss.weight_digit_right,
            slot_order=cfg.loss.weight_slot_order,
            slot_overlap=cfg.loss.weight_slot_overlap,
        ),
        label_smoothing=0.0,
        focal_gamma=cfg.loss.focal_gamma,
        slot_margin=cfg.loss.slot_margin,
    )
    metrics = evaluate(accelerator, model, val_loader, loss_fn)

    console.tag_print(
        "eval",
        f"checkpoint={args.checkpoint} "
        f"n_val={len(val_ds)} "
        f"loss={metrics['loss']:.4f} "
        f"acc_dl={metrics['acc_digit_left']:.4f} "
        f"acc_op={metrics['acc_operator']:.4f} "
        f"acc_dr={metrics['acc_digit_right']:.4f} "
        f"acc_full={metrics['acc_expression']:.4f}",
    )
    accelerator.end_training()


if __name__ == "__main__":
    main()
