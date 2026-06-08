"""accelerate 训练入口: 8 卡 DDP + fp16 + 线性 warmup + AdamW.

调用方式 (任选其一):

1) accelerate launch:
   accelerate launch --num_processes 8 --mixed_precision fp16 \\
       -m cas_ocr_model.trainer.train \\
       --data-root ../../../../dataset --output-dir ./runs/exp1

2) torchrun:
   torchrun --nproc_per_node=8 -m cas_ocr_model.trainer.train \\
       --data-root ../../../../dataset --output-dir ./runs/exp1

3) YAML/TOML 配置 + CLI 覆盖:
   accelerate launch --num_processes 8 --mixed_precision fp16 \\
       -m cas_ocr_model.trainer.train --config configs/8gpu_ddp.yaml

主进程 (rank 0) 负责:
    * 写日志 (accelerate.print)
    * 保存 checkpoint (unwrap DDP 后存 model_state_dict)
    * 记录 best metric
非 rank 0 不写盘.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.utils import set_seed

from cas_ocr_model.datasets.format import DatasetManifest
from .config import (
    FullConfig,
    NUM_DIGIT_CLASSES,
    NUM_OPERATOR_CLASSES,
    cfg_to_dict,
    ensure_output_dir,
    load_config,
    merge_args_to_config,
    parse_args,
)
from .data import CaptchaPairDataset, collate_triple
from .losses import LossWeights, TripleHeadLoss, compute_accuracy
from .model import CaptchaTripleHeadCNN


# ----------------------------------------------------------------------------
# 学习率调度
# ----------------------------------------------------------------------------


def make_linear_warmup_cosine(total_steps: int, warmup_ratio: float) -> torch.optim.lr_scheduler.LambdaLR:
    """线性 warmup -> cosine 衰减."""
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


# ----------------------------------------------------------------------------
# 训练主循环
# ----------------------------------------------------------------------------


def train_one_epoch(
    accelerator: Accelerator,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss_fn: TripleHeadLoss,
    grad_clip: float,
    log_every_n: int,
    epoch: int,
) -> dict[str, float]:
    model.train()
    metric_sum = {"loss": 0.0}
    n = 0
    t0 = time.time()
    for step, (images, labels) in enumerate(loader):
        outputs = model(images, return_aux=True)
        losses = loss_fn(outputs, labels)
        loss = losses["loss"]
        accs = compute_accuracy(outputs, labels)

        accelerator.backward(loss)
        if grad_clip and grad_clip > 0:
            accelerator.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        bs = images.size(0)
        for k, v in losses.items():
            if k != "loss":
                metric_sum.setdefault(k, 0.0)
                metric_sum[k] += v.item() * bs
        metric_sum["loss"] += loss.item() * bs
        for k, v in accs.items():
            metric_sum.setdefault(k, 0.0)
            metric_sum[k] += v * bs
        n += bs

        if (step + 1) % log_every_n == 0:
            elapsed = time.time() - t0
            accelerator.print(
                f"[epoch={epoch} step={step + 1}/{len(loader)} "
                f"loss={loss.item():.4f} "
                f"acc_dl={accs['acc_digit_left']:.3f} "
                f"acc_op={accs['acc_operator']:.3f} "
                f"acc_dr={accs['acc_digit_right']:.3f} "
                f"acc_full={accs['acc_expression']:.3f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} "
                f"elapsed={elapsed:.1f}s]"
            )

    return {k: v / max(1, n) for k, v in metric_sum.items()}


# ----------------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # 1) 配置组装
    cfg = load_config(args.config) if args.config else FullConfig()
    cfg = merge_args_to_config(cfg, args)

    # 2) accelerate 初始化 (DDP/混合精度/进程管理统一交给它)
    accelerator = Accelerator(
        mixed_precision=cfg.train.mixed_precision,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        project_dir=cfg.train.output_dir,
    )
    set_seed(cfg.train.seed, device_specific=True)

    # rank 0 准备输出目录
    if accelerator.is_main_process:
        ensure_output_dir(cfg.train.output_dir)
        accelerator.print(f"[config] {cfg_to_dict(cfg)}")

    # 3) 数据 (按 manifest 读取 train/val/test)
    train_ds = CaptchaPairDataset(
        data_root=cfg.data.data_root,
        image_size_h=cfg.data.image_size_h,
        image_size_w=cfg.data.image_size_w,
        threshold=cfg.data.threshold,
        binarize_mode=cfg.data.binarize_mode,
        adaptive_block_size=cfg.data.adaptive_block_size,
        adaptive_c=cfg.data.adaptive_c,
        split="train",
    )
    val_ds = CaptchaPairDataset(
        data_root=cfg.data.data_root,
        image_size_h=cfg.data.image_size_h,
        image_size_w=cfg.data.image_size_w,
        threshold=cfg.data.threshold,
        binarize_mode=cfg.data.binarize_mode,
        adaptive_block_size=cfg.data.adaptive_block_size,
        adaptive_c=cfg.data.adaptive_c,
        split="val",
    )
    has_test = (Path(cfg.data.data_root) / "manifest.json").is_file() and bool(
        DatasetManifest.load(cfg.data.data_root).splits.get("test")
    )
    test_ds = None
    if has_test:
        try:
            test_ds = CaptchaPairDataset(
                data_root=cfg.data.data_root,
                image_size_h=cfg.data.image_size_h,
                image_size_w=cfg.data.image_size_w,
                threshold=cfg.data.threshold,
                binarize_mode=cfg.data.binarize_mode,
                adaptive_block_size=cfg.data.adaptive_block_size,
                adaptive_c=cfg.data.adaptive_c,
                split="test",
            )
        except RuntimeError as e:
            accelerator.print(f"[data] test split unavailable: {e}; skipping")
            test_ds = None
    accelerator.print(
        f"[data] train={len(train_ds)} val={len(val_ds)} "
        f"test={len(test_ds) if test_ds else 0} "
        f"image_size=({cfg.data.image_size_h},{cfg.data.image_size_w}) "
        f"binarize={cfg.data.binarize_mode}"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.per_device_batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collate_triple,
        drop_last=True,
        persistent_workers=cfg.data.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.per_device_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collate_triple,
        drop_last=False,
        persistent_workers=cfg.data.num_workers > 0,
    )
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.train.per_device_batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            collate_fn=collate_triple,
            drop_last=False,
            persistent_workers=cfg.data.num_workers > 0,
        )

    # 4) 模型 / 损失 / 优化器
    model = CaptchaTripleHeadCNN(
        backbone=cfg.model.backbone,
        pretrained=cfg.model.pretrained,
        dropout=cfg.model.dropout,
        slot_hidden_dim=cfg.model.slot_hidden_dim,
        slot_attention_heads=cfg.model.slot_attention_heads,
        num_digit_classes=NUM_DIGIT_CLASSES,
        num_operator_classes=NUM_OPERATOR_CLASSES,
    )
    loss_fn = TripleHeadLoss(
        weights=LossWeights(
            digit_left=cfg.loss.weight_digit_left,
            operator=cfg.loss.weight_operator,
            digit_right=cfg.loss.weight_digit_right,
            slot_order=cfg.loss.weight_slot_order,
            slot_overlap=cfg.loss.weight_slot_overlap,
        ),
        label_smoothing=cfg.loss.label_smoothing,
        focal_gamma=cfg.loss.focal_gamma,
        slot_margin=cfg.loss.slot_margin,
    )

    # 8 卡 DDP 建议: lr * world_size 线性缩放; 让用户用 --learning-rate 自行缩放,
    # 这里不做隐式改动, 保持 CLI 直观.
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )

    # 学习率调度基于加速后总步数
    steps_per_epoch = math.ceil(len(train_loader) / cfg.train.gradient_accumulation_steps)
    total_steps = steps_per_epoch * cfg.train.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_linear_warmup_cosine(total_steps, cfg.train.warmup_ratio)
    )

    # 5) accelerate prepare (DDP + 自动混合精度 cast)
    to_prepare = [model, optimizer, train_loader, val_loader, scheduler]
    if test_loader is not None:
        to_prepare.append(test_loader)
    prepared = accelerator.prepare(*to_prepare)
    model, optimizer, train_loader, val_loader, scheduler = prepared[:5]
    test_loader = prepared[5] if len(prepared) > 5 else None

    # 6) 可选: 断点续训
    start_epoch = 0
    best_acc = -1.0
    if cfg.train.resume_from:
        ckpt = torch.load(cfg.train.resume_from, map_location="cpu")
        accelerator.load_state(cfg.train.resume_from)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("best_acc", -1.0)
        accelerator.print(f"[resume] from {cfg.train.resume_from} epoch={start_epoch} best={best_acc:.4f}")

    # 7) 训练循环
    for epoch in range(start_epoch, cfg.train.epochs):
        accelerator.print(f"\n[epoch {epoch + 1}/{cfg.train.epochs}]")
        t0 = time.time()

        train_metrics = train_one_epoch(
            accelerator=accelerator,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            grad_clip=cfg.train.grad_clip,
            log_every_n=cfg.train.log_every_n_steps,
            epoch=epoch + 1,
        )

        # 验证 (每个 epoch 结束, 用于 early-stop 决策 / 日志)
        val_metrics = evaluate(accelerator, model, val_loader, loss_fn)

        train_time = time.time() - t0
        accelerator.print(
            f"[epoch {epoch + 1}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc_full={train_metrics['acc_expression']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc_full={val_metrics['acc_expression']:.4f} "
            f"time={train_time:.1f}s"
        )

        # 最后一个 epoch 跑 test 集, 给出最终泛化指标
        is_last = (epoch + 1) == cfg.train.epochs
        if is_last and test_loader is not None:
            test_metrics = evaluate(accelerator, model, test_loader, loss_fn)
            accelerator.print(
                f"[test][final] "
                f"loss={test_metrics['loss']:.4f} "
                f"acc_dl={test_metrics['acc_digit_left']:.4f} "
                f"acc_op={test_metrics['acc_operator']:.4f} "
                f"acc_dr={test_metrics['acc_digit_right']:.4f} "
                f"acc_full={test_metrics['acc_expression']:.4f}"
            )

        # 8) checkpoint (rank 0 only)
        if accelerator.is_main_process and (epoch + 1) % cfg.train.save_every_n_epochs == 0:
            save_checkpoint(
                accelerator=accelerator,
                model=model,
                epoch=epoch,
                cfg=cfg,
                metrics=val_metrics,
                is_best=val_metrics["acc_expression"] > best_acc,
                best_acc=max(best_acc, val_metrics["acc_expression"]),
            )

    accelerator.print("[done] training complete")
    accelerator.wait_for_everyone()
    accelerator.end_training()


# ----------------------------------------------------------------------------
# 验证 (在 train.py 内部复用, eval.py 单独包装)
# ----------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    accelerator: Accelerator,
    model: nn.Module,
    loader: DataLoader,
    loss_fn: TripleHeadLoss,
) -> dict[str, float]:
    model.eval()
    metric_sum = {"loss": 0.0}
    n = 0
    for images, labels in loader:
        outputs = model(images, return_aux=True)
        losses = loss_fn(outputs, labels)
        accs = compute_accuracy(outputs, labels)
        bs = images.size(0)
        for k, v in losses.items():
            if k != "loss":
                metric_sum.setdefault(k, 0.0)
                metric_sum[k] += v.item() * bs
        metric_sum["loss"] += losses["loss"].item() * bs
        for k, v in accs.items():
            metric_sum.setdefault(k, 0.0)
            metric_sum[k] += v * bs
        n += bs
    return {k: v / max(1, n) for k, v in metric_sum.items()}


# ----------------------------------------------------------------------------
# Checkpoint
# ----------------------------------------------------------------------------


def save_checkpoint(
    accelerator: Accelerator,
    model: nn.Module,
    epoch: int,
    cfg: FullConfig,
    metrics: dict[str, float],
    is_best: bool,
    best_acc: float,
) -> None:
    """rank 0 写盘. DDP unwrap 后存 model_state_dict.

    文件:
        output_dir/last.pt
        output_dir/best.pt (仅当 is_best)
    """
    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    unwrapped = accelerator.unwrap_model(model)
    state = {
        "epoch": epoch,
        "model_state_dict": unwrapped.state_dict(),
        "metrics": metrics,
        "best_acc": best_acc,
        "config": cfg_to_dict(cfg),
    }
    last_path = output_dir / "last.pt"
    accelerator.save(state, str(last_path))
    accelerator.print(f"[ckpt] saved {last_path} (epoch={epoch + 1}, val_acc={metrics['acc_expression']:.4f})")

    if is_best:
        best_path = output_dir / "best.pt"
        accelerator.save(state, str(best_path))
        accelerator.print(f"[ckpt] NEW BEST -> {best_path} (val_acc={best_acc:.4f})")


if __name__ == "__main__":
    main()
