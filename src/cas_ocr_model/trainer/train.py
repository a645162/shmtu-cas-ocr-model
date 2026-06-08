"""accelerate 训练入口: 8 卡 DDP + fp16 + 线性 warmup + AdamW.

调用方式 (任选其一):

1) accelerate launch:
   accelerate launch --num_processes 8 --num_machines 1 --dynamo_backend no --mixed_precision fp16 \\
       -m cas_ocr_model.trainer.train \\
       --data-root ../../../../dataset --output-dir ./runs/exp1

2) torchrun:
   torchrun --nproc_per_node=8 -m cas_ocr_model.trainer.train \\
       --data-root ../../../../dataset --output-dir ./runs/exp1

3) YAML/TOML 配置 + CLI 覆盖:
   accelerate launch --num_processes 8 --num_machines 1 --dynamo_backend no --mixed_precision fp16 \\
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
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

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
from cas_ocr_model.model.stats import collect_model_stats, format_model_stats

try:
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
except ImportError:  # pragma: no cover - rich 是可选运行时依赖
    Progress = None


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


def parse_report_to(report_to: str | None) -> str | list[str] | None:
    """把配置中的 tracker 字段转换成 accelerate 可接受的格式."""
    if report_to is None:
        return None
    raw = report_to.strip()
    if not raw or raw.lower() in {"none", "null", "false", "off"}:
        return None
    if raw.lower() == "all":
        return "all"
    parts = [item.strip() for item in raw.split(",") if item.strip()]
    if not parts:
        return None
    return parts if len(parts) > 1 else parts[0]


def ensure_tracker_dependencies(report_to: str | list[str] | None) -> None:
    """在显式启用 tracker 时给出清晰依赖报错."""
    if report_to is None or report_to == "all":
        return
    trackers = [report_to] if isinstance(report_to, str) else list(report_to)
    if "wandb" in trackers:
        try:
            import wandb  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "启用 wandb 需要先安装依赖: pip install -e .[wandb] 或 pip install wandb"
            ) from e


def build_tracker_init_kwargs(cfg: FullConfig) -> dict[str, dict[str, Any]]:
    """构造 accelerate.init_trackers 的后端特定参数."""
    wandb_kwargs: dict[str, Any] = {}
    if cfg.train.wandb_run_name:
        wandb_kwargs["name"] = cfg.train.wandb_run_name
    if cfg.train.wandb_entity:
        wandb_kwargs["entity"] = cfg.train.wandb_entity
    if cfg.train.wandb_tags:
        wandb_kwargs["tags"] = cfg.train.wandb_tags
    return {"wandb": wandb_kwargs} if wandb_kwargs else {}


def reduce_metric_sums(
    accelerator: Accelerator,
    metric_sum: dict[str, float],
    sample_count: int,
) -> tuple[dict[str, float], int]:
    """把各 rank 的累计值做 sum reduce, 返回全局平均值."""
    if not metric_sum:
        return {}, 0
    keys = list(metric_sum.keys())
    payload = [metric_sum[k] for k in keys]
    payload.append(float(sample_count))
    tensor = torch.tensor(payload, device=accelerator.device, dtype=torch.float64)
    reduced = accelerator.reduce(tensor, reduction="sum")
    total_count = max(1, int(reduced[-1].item()))
    metrics = {
        key: reduced[idx].item() / total_count
        for idx, key in enumerate(keys)
    }
    return metrics, total_count


def format_seconds(seconds: float) -> str:
    """把秒数格式化成 h:mm:ss / m:ss."""
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def should_use_rich_progress(accelerator: Accelerator, cfg: FullConfig) -> bool:
    """仅主进程且为交互式终端时启用 rich 进度条."""
    return bool(
        cfg.train.use_rich_progress
        and Progress is not None
        and accelerator.is_main_process
        and sys.stderr.isatty()
    )


def create_epoch_progress(epoch: int, total_epochs: int, total_steps: int) -> tuple[Progress, int]:
    """构造单 epoch 的 rich 进度条."""
    progress = Progress(
        TextColumn(f"[bold cyan]epoch {epoch}/{total_epochs}[/bold cyan]"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("loss={task.fields[loss]:.4f}"),
        TextColumn("acc={task.fields[acc]:.4f}"),
        TextColumn("lr={task.fields[lr]}"),
        TextColumn("{task.fields[samples_per_s]} img/s"),
        transient=False,
    )
    task_id = progress.add_task(
        "train",
        total=total_steps,
        loss=0.0,
        acc=0.0,
        lr="0.00e+00",
        samples_per_s="0",
    )
    return progress, task_id


def create_eval_progress(stage: str, total_steps: int) -> tuple[Progress, int]:
    """构造验证 / 测试阶段的轻量进度条."""
    progress = Progress(
        TextColumn(f"[bold green]{stage}[/bold green]"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=False,
    )
    task_id = progress.add_task(stage, total=total_steps)
    return progress, task_id


def describe_backbone_weights(backbone: str, pretrained: bool) -> str:
    """返回启动日志里展示的 backbone 权重来源."""
    if not pretrained:
        return "None"
    mapping = {
        "resnet18": "ResNet18_Weights.IMAGENET1K_V1",
        "resnet34": "ResNet34_Weights.IMAGENET1K_V1",
        "mobilenet_v3_small": "MobileNet_V3_Small_Weights.IMAGENET1K_V1",
    }
    return mapping.get(backbone, "ImageNet pretrained")


def maybe_log_metrics(
    accelerator: Accelerator,
    metrics: dict[str, float],
    step: int,
) -> None:
    """统一封装 tracker 日志调用."""
    if not metrics or not accelerator.is_main_process or not getattr(accelerator, "trackers", None):
        return
    accelerator.log(metrics, step=step)


def sync_wandb_config(accelerator: Accelerator, cfg: FullConfig) -> None:
    """显式同步 config 到 wandb, 让 run 页面稳定可见."""
    if not accelerator.is_main_process or not getattr(accelerator, "trackers", None):
        return
    tracker_names = {tracker.name for tracker in accelerator.trackers}
    if "wandb" not in tracker_names:
        return
    run = accelerator.get_tracker("wandb", unwrap=True)
    run.config.update(cfg_to_dict(cfg), allow_val_change=True)


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
    total_epochs: int,
    steps_per_epoch: int,
    global_step: int,
    enable_rich_progress: bool,
) -> tuple[dict[str, float], int]:
    model.train()
    metric_sum = {"loss": 0.0}
    window_sum = {"loss": 0.0}
    n = 0
    window_n = 0
    epoch_update_step = 0
    epoch_start = time.time()
    last_log_time = epoch_start
    optimizer.zero_grad(set_to_none=True)

    progress, task_id = create_epoch_progress(epoch, total_epochs, steps_per_epoch) if enable_rich_progress else (None, None)
    progress_ctx = progress if progress is not None else nullcontext()

    with progress_ctx:
        for step, (images, labels) in enumerate(loader, start=1):
            with accelerator.accumulate(model):
                outputs = model(images, return_aux=True)
                losses = loss_fn(outputs, labels)
                loss = losses["loss"]
                accs = compute_accuracy(outputs, labels)

                accelerator.backward(loss)
                if accelerator.sync_gradients and grad_clip and grad_clip > 0:
                    accelerator.clip_grad_norm_(model.parameters(), grad_clip)
                if accelerator.sync_gradients:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            bs = images.size(0)
            batch_metrics = {"loss": loss.item()}
            batch_metrics.update({k: v.item() for k, v in losses.items() if k != "loss"})
            batch_metrics.update(accs)
            for k, v in batch_metrics.items():
                metric_sum.setdefault(k, 0.0)
                metric_sum[k] += v * bs
                window_sum.setdefault(k, 0.0)
                window_sum[k] += v * bs
            n += bs
            window_n += bs

            if not accelerator.sync_gradients:
                continue

            epoch_update_step += 1
            global_step += 1
            if progress is not None:
                progress.advance(task_id, 1)

            is_last_update = epoch_update_step == steps_per_epoch
            if epoch_update_step % log_every_n != 0 and not is_last_update:
                continue

            window_metrics, global_window_n = reduce_metric_sums(accelerator, window_sum, window_n)
            now = time.time()
            interval_elapsed = max(now - last_log_time, 1e-6)
            total_elapsed = max(now - epoch_start, 1e-6)
            samples_per_s = global_window_n / interval_elapsed
            eta_seconds = (steps_per_epoch - epoch_update_step) * (total_elapsed / max(1, epoch_update_step))
            lr = scheduler.get_last_lr()[0]

            if progress is not None:
                progress.update(
                    task_id,
                    loss=window_metrics["loss"],
                    acc=window_metrics["acc_expression"],
                    lr=f"{lr:.2e}",
                    samples_per_s=f"{samples_per_s:.0f}",
                    refresh=True,
                )
            else:
                accelerator.print(
                    f"[train] epoch={epoch}/{total_epochs} "
                    f"step={epoch_update_step}/{steps_per_epoch} "
                    f"global_step={global_step} "
                    f"loss={window_metrics['loss']:.4f} "
                    f"acc_full={window_metrics['acc_expression']:.4f} "
                    f"acc_dl={window_metrics['acc_digit_left']:.4f} "
                    f"acc_op={window_metrics['acc_operator']:.4f} "
                    f"acc_dr={window_metrics['acc_digit_right']:.4f} "
                    f"lr={lr:.2e} "
                    f"throughput={samples_per_s:.1f}img/s "
                    f"eta={format_seconds(eta_seconds)}"
                )

            maybe_log_metrics(
                accelerator,
                {
                    "train/epoch": float(epoch),
                    "train/epoch_step": float(epoch_update_step),
                    "train/lr": lr,
                    "train/samples_per_s": samples_per_s,
                    **{f"train/{k}": v for k, v in window_metrics.items()},
                },
                step=global_step,
            )
            window_sum = {"loss": 0.0}
            window_n = 0
            last_log_time = now

    metrics, _ = reduce_metric_sums(accelerator, metric_sum, n)
    return metrics, global_step


# ----------------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # 1) 配置组装
    cfg = load_config(args.config) if args.config else FullConfig()
    cfg = merge_args_to_config(cfg, args)
    report_to = parse_report_to(cfg.train.report_to)
    ensure_tracker_dependencies(report_to)

    # 2) accelerate 初始化 (DDP/混合精度/进程管理统一交给它)
    accelerator = Accelerator(
        mixed_precision=cfg.train.mixed_precision,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        project_dir=cfg.train.output_dir,
        log_with=report_to,
    )
    set_seed(cfg.train.seed, device_specific=True)

    # rank 0 准备输出目录
    if accelerator.is_main_process:
        ensure_output_dir(cfg.train.output_dir)
    accelerator.wait_for_everyone()

    if report_to is not None:
        init_kwargs = build_tracker_init_kwargs(cfg)
        accelerator.init_trackers(
            project_name=cfg.train.tracker_project_name,
            config=cfg_to_dict(cfg),
            init_kwargs=init_kwargs or None,
        )
        sync_wandb_config(accelerator, cfg)

    effective_batch_size = (
        cfg.train.per_device_batch_size
        * accelerator.num_processes
        * cfg.train.gradient_accumulation_steps
    )
    accelerator.print(
        f"[init] rank={accelerator.process_index} world_size={accelerator.num_processes} "
        f"device={accelerator.device} mixed_precision={cfg.train.mixed_precision} "
        f"grad_accum={cfg.train.gradient_accumulation_steps}"
    )
    if accelerator.is_main_process:
        accelerator.print(
            f"[config] output={cfg.train.output_dir} data={cfg.data.data_root} "
            f"backbone={cfg.model.backbone} batch/device={cfg.train.per_device_batch_size} "
            f"effective_batch={effective_batch_size} tracker={cfg.train.report_to}"
        )
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
    accelerator.print(
        f"[model] backbone={cfg.model.backbone} "
        f"pretrained={cfg.model.pretrained} "
        f"weights={describe_backbone_weights(cfg.model.backbone, cfg.model.pretrained)} "
        f"gray_stem=rgb_mean"
    )
    if accelerator.is_main_process:
        accelerator.print(
            f"[model-stats] "
            f"{format_model_stats(collect_model_stats(model, cfg.data.image_size_h, cfg.data.image_size_w))}"
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

    # 5) accelerate prepare (DDP + 自动混合精度 cast)
    to_prepare = [model, optimizer, train_loader, val_loader]
    if test_loader is not None:
        to_prepare.append(test_loader)
    prepared = accelerator.prepare(*to_prepare)
    model, optimizer, train_loader, val_loader = prepared[:4]
    test_loader = prepared[4] if len(prepared) > 4 else None

    # 学习率调度必须基于 prepare 之后的本进程步数.
    # DDP 下每个 rank 只迭代自己那一份数据; 若用 prepare 前的全局长度,
    # rich 进度条会卡在 1/world_size, warmup/cosine 也会慢 world_size 倍.
    steps_per_epoch = math.ceil(len(train_loader) / cfg.train.gradient_accumulation_steps)
    total_steps = steps_per_epoch * cfg.train.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_linear_warmup_cosine(total_steps, cfg.train.warmup_ratio)
    )

    # 6) 可选: 断点续训
    start_epoch = 0
    best_acc = -1.0
    global_step = 0
    if cfg.train.resume_from:
        ckpt = torch.load(cfg.train.resume_from, map_location="cpu")
        accelerator.load_state(cfg.train.resume_from)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("best_acc", -1.0)
        global_step = start_epoch * steps_per_epoch
        accelerator.print(f"[resume] from {cfg.train.resume_from} epoch={start_epoch} best={best_acc:.4f}")

    # 7) 训练循环
    enable_rich_progress = should_use_rich_progress(accelerator, cfg)
    for epoch in range(start_epoch, cfg.train.epochs):
        accelerator.print(f"\n[epoch {epoch + 1}/{cfg.train.epochs}]")
        t0 = time.time()

        train_metrics, global_step = train_one_epoch(
            accelerator=accelerator,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            grad_clip=cfg.train.grad_clip,
            log_every_n=cfg.train.log_every_n_steps,
            epoch=epoch + 1,
            total_epochs=cfg.train.epochs,
            steps_per_epoch=steps_per_epoch,
            global_step=global_step,
            enable_rich_progress=enable_rich_progress,
        )

        # 验证 (每个 epoch 结束, 用于 early-stop 决策 / 日志)
        val_metrics = evaluate(
            accelerator,
            model,
            val_loader,
            loss_fn,
            stage=f"val {epoch + 1}/{cfg.train.epochs}",
            enable_rich_progress=enable_rich_progress,
        )
        is_best = val_metrics["acc_expression"] > best_acc
        best_acc = max(best_acc, val_metrics["acc_expression"])

        train_time = time.time() - t0
        accelerator.print(
            f"[epoch-summary] epoch={epoch + 1}/{cfg.train.epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc_full={train_metrics['acc_expression']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc_full={val_metrics['acc_expression']:.4f} "
            f"time={train_time:.1f}s "
            f"best_val_acc={best_acc:.4f}"
        )
        maybe_log_metrics(
            accelerator,
            {
                "epoch": float(epoch + 1),
                "train/time_s": train_time,
                **{f"train_epoch/{k}": v for k, v in train_metrics.items()},
                **{f"val/{k}": v for k, v in val_metrics.items()},
            },
            step=global_step,
        )

        # 最后一个 epoch 跑 test 集, 给出最终泛化指标
        is_last = (epoch + 1) == cfg.train.epochs
        if is_last and test_loader is not None:
            test_metrics = evaluate(
                accelerator,
                model,
                test_loader,
                loss_fn,
                stage="test",
                enable_rich_progress=enable_rich_progress,
            )
            accelerator.print(
                f"[test][final] "
                f"loss={test_metrics['loss']:.4f} "
                f"acc_dl={test_metrics['acc_digit_left']:.4f} "
                f"acc_op={test_metrics['acc_operator']:.4f} "
                f"acc_dr={test_metrics['acc_digit_right']:.4f} "
                f"acc_full={test_metrics['acc_expression']:.4f}"
            )
            maybe_log_metrics(
                accelerator,
                {f"test/{k}": v for k, v in test_metrics.items()},
                step=global_step,
            )

        # 8) checkpoint (rank 0 only)
        if accelerator.is_main_process and (epoch + 1) % cfg.train.save_every_n_epochs == 0:
            save_checkpoint(
                accelerator=accelerator,
                model=model,
                epoch=epoch,
                cfg=cfg,
                metrics=val_metrics,
                is_best=is_best,
                best_acc=best_acc,
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
    *,
    stage: str = "val",
    enable_rich_progress: bool = False,
) -> dict[str, float]:
    model.eval()
    metric_sum = {"loss": 0.0}
    n = 0
    progress, task_id = create_eval_progress(stage, len(loader)) if enable_rich_progress else (None, None)
    progress_ctx = progress if progress is not None else nullcontext()
    with progress_ctx:
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
            if progress is not None:
                progress.advance(task_id, 1)
    metrics, _ = reduce_metric_sums(accelerator, metric_sum, n)
    return metrics


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
