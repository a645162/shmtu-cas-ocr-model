"""accelerate 训练入口: 8 卡 DDP + mixed precision + 线性 warmup + AdamW.

调用方式 (任选其一):

1) accelerate launch:
   accelerate launch --num_processes 8 --num_machines 1 --dynamo_backend inductor --mixed_precision bf16 \\
       -m cas_ocr_model.trainer.train \\
       --data-root ../../../../dataset --output-dir ./runs/exp1

2) torchrun:
   torchrun --nproc_per_node=8 -m cas_ocr_model.trainer.train \\
       --data-root ../../../../dataset --output-dir ./runs/exp1

3) YAML/TOML 配置 + CLI 覆盖:
   accelerate launch --num_processes 8 --num_machines 1 --dynamo_backend inductor --mixed_precision bf16 \\
       -m cas_ocr_model.trainer.train --config configs/8gpu_ddp.yaml

主进程 (rank 0) 负责:
    * 写日志 (accelerate.print)
    * 保存 checkpoint (unwrap DDP 后存 model_state_dict)
    * 记录 best metric
非 rank 0 不写盘.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
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
    apply_env_overrides,
    cfg_to_dict,
    ensure_output_dir,
    load_config,
    merge_args_to_config,
    parse_args,
)
from .data import CaptchaPairDataset, collate_triple
from .losses import LossWeights, TriSlotDecoderLoss, compute_accuracy
from .model import build_model_from_config, build_model_metadata
from cas_ocr_model.model import ModelStats
from cas_ocr_model.model.stats import collect_model_stats, format_model_stats

from cas_ocr_model.common.console import AcceleratorConsole
from cas_ocr_model.common.checkpoint_pip import (
    capture_pip_list_snapshot,
    extract_checkpoint_pip_list,
    write_pip_list_json,
)

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


BASE_METRICS = {
    "loss": 0.0,
    "acc_digit_left": 0.0,
    "acc_operator": 0.0,
    "acc_digit_right": 0.0,
    "acc_expression": 0.0,
}
NONFINITE_BACKPROP_STEP_STOP = "nonfinite_backprop_steps"
NONFINITE_BACKPROP_EPOCH_STOP = "nonfinite_backprop_epochs"


@dataclass
class TrainEpochResult:
    metrics: dict[str, float]
    global_step: int
    had_nonfinite_backprop: bool
    had_nonfinite_gradient: bool
    nonfinite_backprop_events: int
    consecutive_nonfinite_backprop_steps: int
    stop_reason: str | None = None


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
    if not raw or raw.lower() in {"none", "null", "false", "off", "disabled"}:
        return None
    if raw.lower() == "all":
        return "all"
    parts = [item.strip() for item in raw.split(",") if item.strip()]
    if not parts:
        return None
    return parts if len(parts) > 1 else parts[0]


def is_truthy_env(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def wandb_is_disabled_by_env() -> tuple[bool, str | None]:
    for name in ("SHMTU_DISABLE_WANDB", "SHMTU_WANDB_DISABLED", "WANDB_DISABLED"):
        value = os.environ.get(name)
        if is_truthy_env(value):
            return True, f"{name}={value}"
    mode = os.environ.get("WANDB_MODE")
    if mode and mode.strip().lower() == "disabled":
        return True, f"WANDB_MODE={mode}"
    return False, None


def is_wandb_installed() -> bool:
    try:
        import wandb  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_report_to(report_to: str | None) -> tuple[str | list[str] | None, str]:
    raw = (report_to or "").strip()
    lowered = raw.lower()
    env_disabled, env_reason = wandb_is_disabled_by_env()

    if lowered in {"", "auto"}:
        if env_disabled:
            return None, f"disabled-by-env:{env_reason}"
        if is_wandb_installed():
            return "wandb", "auto-wandb"
        return None, "auto-no-wandb"

    parsed = parse_report_to(report_to)
    if parsed is None:
        return None, f"config:{raw or 'none'}"
    if env_disabled and ("wandb" in ([parsed] if isinstance(parsed, str) else list(parsed))):
        return None, f"disabled-by-env:{env_reason}"
    return parsed, f"config:{raw}"


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


def resolve_default_wandb_run_name(output_dir: str) -> str:
    output_path = Path(output_dir).resolve()
    runs_root_raw = os.environ.get("SHMTU_RUNS_ROOT")
    if runs_root_raw:
        runs_root = Path(runs_root_raw).resolve()
        try:
            rel_path = output_path.relative_to(runs_root)
        except ValueError:
            rel_path = None
        if rel_path is not None and rel_path.parts:
            return rel_path.as_posix()

    if len(output_path.parts) >= 2:
        return "/".join(output_path.parts[-2:])
    return output_path.name


def ensure_wandb_run_name(cfg: FullConfig, report_to: str | list[str] | None) -> None:
    if cfg.train.wandb_run_name:
        return
    if report_to is None:
        return
    trackers = [report_to] if isinstance(report_to, str) else list(report_to)
    if "wandb" not in trackers and report_to != "all":
        return
    cfg.train.wandb_run_name = resolve_default_wandb_run_name(cfg.train.output_dir)


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


def reduce_bool_any(accelerator: Accelerator, flag: bool) -> bool:
    tensor = torch.tensor(
        [1 if flag else 0],
        device=accelerator.device,
        dtype=torch.int32,
    )
    reduced = accelerator.reduce(tensor, reduction="sum")
    return int(reduced.item()) > 0


def format_seconds(seconds: float) -> str:
    """把秒数格式化成 h:mm:ss / m:ss."""
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def styled_metric(value: float, *, style: str, fmt: str = ".4f") -> str:
    return f"[{style}]{value:{fmt}}[/]"


def styled_text(value: str, *, style: str) -> str:
    return f"[{style}]{value}[/]"


def render_epoch_summary(
    *,
    epoch: int,
    total_epochs: int,
    train_metrics: dict[str, float],
    train_time: float,
    nonfinite_events: int,
    nonfinite_steps: int,
    nonfinite_epochs: int,
    val_metrics: dict[str, float] | None = None,
    best_acc: float | None = None,
    best_epoch: int | None = None,
    best_val_loss: float | None = None,
    best_val_loss_epoch: int | None = None,
    is_best: bool | None = None,
    stale_epochs: int | None = None,
    stop_reason: str | None = None,
) -> str:
    parts = [
        styled_text("[epoch-summary]", style="tag.epoch-summary"),
        f"epoch={styled_text(f'{epoch}/{total_epochs}', style='bold cyan')}",
        f"train_loss={styled_metric(train_metrics['loss'], style='bold yellow')}",
        f"train_acc_full={styled_metric(train_metrics['acc_expression'], style='metric.good')}",
    ]
    if val_metrics is not None:
        parts.extend(
            [
                f"val_loss={styled_metric(val_metrics['loss'], style='bold bright_blue')}",
                f"val_acc_full={styled_metric(val_metrics['acc_expression'], style='metric.good')}",
            ]
        )
    parts.append(f"time={styled_text(f'{train_time:.1f}s', style='bold magenta')}")
    if best_acc is not None and best_epoch is not None:
        parts.append(
            "best_val_acc="
            f"{styled_metric(best_acc, style='metric.best')}@"
            f"{styled_text(str(best_epoch), style='metric.best')}"
        )
    if best_val_loss is not None and best_val_loss_epoch is not None:
        parts.append(
            "best_val_loss="
            f"{styled_metric(best_val_loss, style='metric.best')}@"
            f"{styled_text(str(best_val_loss_epoch), style='metric.best')}"
        )
    if is_best is not None:
        parts.append(
            "is_best="
            f"{styled_text(str(int(is_best)), style='metric.best' if is_best else 'bold white')}"
        )
    if stale_epochs is not None:
        stale_style = "bold red" if stale_epochs > 0 else "bold green"
        parts.append(f"stale_epochs={styled_text(str(stale_epochs), style=stale_style)}")
    parts.append(
        f"nonfinite_events={styled_text(str(nonfinite_events), style='bold red' if nonfinite_events > 0 else 'bold green')}"
    )
    parts.append(
        f"nonfinite_steps={styled_text(str(nonfinite_steps), style='bold red' if nonfinite_steps > 0 else 'bold green')}"
    )
    parts.append(
        f"nonfinite_epochs={styled_text(str(nonfinite_epochs), style='bold red' if nonfinite_epochs > 0 else 'bold green')}"
    )
    if stop_reason:
        parts.append(f"stop_reason={styled_text(stop_reason, style='bold bright_red')}")
    return " ".join(parts)


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
        "mobilenet_v3_large": "MobileNet_V3_Large_Weights.IMAGENET1K_V2",
        "resnet50": "timm pretrained",
        "r50": "timm pretrained",
        "resnet101": "timm pretrained",
        "r101": "timm pretrained",
        "mobilenetv3_small_050": "timm pretrained",
        "mobilenetv3_small_075": "timm pretrained",
        "mobilenetv3_small_100": "timm pretrained",
        "mobilenetv3_large_075": "timm pretrained",
        "mobilenetv3_large_100": "timm pretrained",
        "mobilenetv3_large_150d": "timm pretrained",
        "mobilenetv3_rw": "timm pretrained",
        "mobilenetv4_conv_small": "timm pretrained",
        "mobilenetv4_conv_small_035": "timm pretrained",
        "mobilenetv4_conv_small_050": "timm pretrained",
        "mobilenetv4_conv_medium": "timm pretrained",
        "mobilenetv4_conv_large": "timm pretrained",
        "mobilenetv4_conv_aa_medium": "timm pretrained",
        "mobilenetv4_conv_aa_large": "timm pretrained",
        "mobilenetv4_conv_blur_medium": "timm pretrained",
        "mobilenetv4_hybrid_medium": "timm pretrained",
        "mobilenetv4_hybrid_medium_075": "timm pretrained",
        "mobilenetv4_hybrid_large": "timm pretrained",
        "mobilenetv4_hybrid_large_075": "timm pretrained",
        "repvgg_a0": "timm pretrained",
        "repvgg_a1": "timm pretrained",
        "repvgg_a2": "timm pretrained",
        "repvgg_b0": "timm pretrained",
        "repvgg_b1": "timm pretrained",
        "repvgg_b1g4": "timm pretrained",
        "repvgg_b2": "timm pretrained",
        "repvgg_b2g4": "timm pretrained",
        "repvgg_b3": "timm pretrained",
        "repvgg_b3g4": "timm pretrained",
        "repvgg_d2se": "timm pretrained",
    }
    if backbone.startswith("timm/"):
        return "timm pretrained"
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


def prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def sync_wandb_config(accelerator: Accelerator, cfg: FullConfig) -> None:
    """显式同步 config 到 wandb, 让 run 页面稳定可见."""
    if not accelerator.is_main_process or not getattr(accelerator, "trackers", None):
        return
    tracker_names = {tracker.name for tracker in accelerator.trackers}
    if "wandb" not in tracker_names:
        return
    run = accelerator.get_tracker("wandb", unwrap=True)
    run.config.update(cfg_to_dict(cfg), allow_val_change=True)


def sync_wandb_model_stats(
    accelerator: Accelerator,
    stats: ModelStats,
) -> None:
    """显式同步模型统计到 wandb summary."""
    if not accelerator.is_main_process or not getattr(accelerator, "trackers", None):
        return
    tracker_names = {tracker.name for tracker in accelerator.trackers}
    if "wandb" not in tracker_names:
        return
    run = accelerator.get_tracker("wandb", unwrap=True)
    run.summary["model/params"] = stats.total_params
    run.summary["model/trainable_params"] = stats.trainable_params
    run.summary["model/params_m"] = stats.total_params / 1_000_000
    run.summary["model/trainable_params_m"] = stats.trainable_params / 1_000_000
    if stats.flops is not None:
        run.summary["model/flops"] = stats.flops
        run.summary["model/flops_g"] = stats.flops / 1_000_000_000
        run.summary["model/flops_m"] = stats.flops / 1_000_000
    run.summary["model/input_shape"] = str(stats.input_shape)
    run.summary["model/stats"] = format_model_stats(stats)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def load_metrics_history(output_dir: str | Path) -> list[dict[str, Any]]:
    history_path = Path(output_dir) / "metrics_history.json"
    if not history_path.is_file():
        return []
    try:
        payload = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return payload if isinstance(payload, list) else []


def upsert_epoch_record(history: list[dict[str, Any]], record: dict[str, Any]) -> list[dict[str, Any]]:
    epoch = int(record["epoch"])
    updated = False
    new_history: list[dict[str, Any]] = []
    for item in history:
        if int(item.get("epoch", -1)) == epoch:
            new_history.append(record)
            updated = True
        else:
            new_history.append(item)
    if not updated:
        new_history.append(record)
    new_history.sort(key=lambda item: int(item.get("epoch", 0)))
    return new_history


def save_epoch_metrics(
    output_dir: str | Path,
    record: dict[str, Any],
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output_dir = Path(output_dir)
    epoch_file = output_dir / "epochs" / f"epoch_{int(record['epoch']):04d}.json"
    history_file = output_dir / "metrics_history.json"
    history = upsert_epoch_record(history, record)
    atomic_write_json(epoch_file, record)
    atomic_write_json(history_file, history)
    save_results_csv(output_dir, record, history)
    return history


def flatten_epoch_record_for_csv(
    record: dict[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    row: dict[str, Any] = {}
    metric_groups = {"train", "val", "test"}
    for key, value in record.items():
        if key == "config":
            continue
        full_key = f"{prefix}{key}" if not prefix else f"{prefix}/{key}"
        if isinstance(value, dict):
            row.update(flatten_epoch_record_for_csv(value, prefix=full_key))
            continue
        if value is None:
            if key in metric_groups:
                continue
            row[full_key] = ""
            continue
        if isinstance(value, bool):
            row[full_key] = int(value)
            continue
        row[full_key] = value
    return row


def build_results_csv_rows(history: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    fieldnames: list[str] = []
    rows: list[dict[str, Any]] = []
    for record in sorted(history, key=lambda item: int(item.get("epoch", 0))):
        row = flatten_epoch_record_for_csv(record)
        rows.append(row)
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames, rows


def rewrite_results_csv(path: Path, history: list[dict[str, Any]]) -> None:
    fieldnames, rows = build_results_csv_rows(history)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    tmp_path.replace(path)


def save_results_csv(
    output_dir: str | Path,
    record: dict[str, Any],
    history: list[dict[str, Any]],
) -> None:
    output_dir = Path(output_dir)
    csv_path = output_dir / "results.csv"
    row = flatten_epoch_record_for_csv(record)

    if not csv_path.is_file():
        rewrite_results_csv(csv_path, history)
        return

    try:
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            fieldnames = list(reader.fieldnames or [])
            last_epoch = 0
            for existing_row in reader:
                raw_epoch = existing_row.get("epoch")
                if raw_epoch in (None, ""):
                    continue
                try:
                    last_epoch = int(float(raw_epoch))
                except ValueError:
                    continue
    except OSError:
        rewrite_results_csv(csv_path, history)
        return

    row_keys = list(row.keys())
    if (
        fieldnames
        and set(row_keys).issubset(fieldnames)
        and int(record["epoch"]) > last_epoch
    ):
        with csv_path.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writerow({key: row.get(key, "") for key in fieldnames})
        return

    rewrite_results_csv(csv_path, history)


def resolve_early_stop_patience(cfg: FullConfig) -> int:
    raw = int(cfg.train.early_stop_patience)
    if raw == 0:
        return 0
    if raw == -1:
        return max(1, math.ceil(cfg.train.epochs * 0.2))
    if raw < -1:
        raise ValueError("train.early_stop_patience 只允许 -1 / 0 / 正整数")
    return raw


def resolve_nonfinite_backprop_patience(field_name: str, raw: int) -> int:
    raw = int(raw)
    if raw < 0:
        raise ValueError(f"{field_name} 只允许 0 / 正整数")
    return raw


def infer_early_stop_state(history: list[dict[str, Any]]) -> tuple[float, int, int]:
    best_acc = -1.0
    best_epoch = 0
    epochs_without_improve = 0
    for item in sorted(history, key=lambda row: int(row.get("epoch", 0))):
        epoch = int(item.get("epoch", 0))
        val_metrics = item.get("val") or {}
        val_acc = val_metrics.get("acc_expression")
        if val_acc is None:
            continue
        val_acc = float(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
    return best_acc, epochs_without_improve, best_epoch


def infer_best_val_loss_state(history: list[dict[str, Any]]) -> tuple[float, int]:
    best_val_loss = float("inf")
    best_val_loss_epoch = 0
    for item in sorted(history, key=lambda row: int(row.get("epoch", 0))):
        epoch = int(item.get("epoch", 0))
        val_metrics = item.get("val") or {}
        val_loss = val_metrics.get("loss")
        if val_loss is None:
            continue
        val_loss = float(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_loss_epoch = epoch
    return best_val_loss, best_val_loss_epoch


def make_metric_sum() -> dict[str, float]:
    return dict(BASE_METRICS)


def find_first_nonfinite_gradient(model: nn.Module) -> str | None:
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        if not bool(torch.isfinite(grad).all().item()):
            return name
    return None


# ----------------------------------------------------------------------------
# 训练主循环
# ----------------------------------------------------------------------------


def train_one_epoch(
    accelerator: Accelerator,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss_fn: TriSlotDecoderLoss,
    grad_clip: float,
    log_every_n: int,
    epoch: int,
    total_epochs: int,
    steps_per_epoch: int,
    global_step: int,
    enable_rich_progress: bool,
    nonfinite_backprop_step_patience: int,
    consecutive_nonfinite_backprop_steps: int,
    console: AcceleratorConsole | None = None,
) -> TrainEpochResult:
    if console is None:
        console = AcceleratorConsole(accelerator)
    model.train()
    metric_sum = make_metric_sum()
    window_sum = make_metric_sum()
    n = 0
    window_n = 0
    epoch_update_step = 0
    epoch_start = time.time()
    last_log_time = epoch_start
    had_nonfinite_backprop = False
    had_nonfinite_gradient = False
    nonfinite_backprop_events = 0
    stop_reason = None
    optimizer.zero_grad(set_to_none=True)

    progress, task_id = create_epoch_progress(epoch, total_epochs, steps_per_epoch) if enable_rich_progress else (None, None)
    progress_ctx = progress if progress is not None else nullcontext()

    with progress_ctx:
        for loader_step, (images, labels) in enumerate(loader, start=1):
            with accelerator.accumulate(model):
                outputs = model(images, return_aux=True)
                losses = loss_fn(outputs, labels)
                loss = losses["loss"]
                local_nonfinite_loss = not bool(torch.isfinite(loss.detach()).all().item())
                has_nonfinite_loss = reduce_bool_any(accelerator, local_nonfinite_loss)
                local_nonfinite_grad_name = None
                has_nonfinite_grad = False

                if has_nonfinite_loss:
                    optimizer.zero_grad(set_to_none=True)
                else:
                    accelerator.backward(loss)
                    local_nonfinite_grad_name = find_first_nonfinite_gradient(model)
                    has_nonfinite_grad = reduce_bool_any(
                        accelerator,
                        local_nonfinite_grad_name is not None,
                    )
                    if has_nonfinite_grad:
                        optimizer.zero_grad(set_to_none=True)
                    else:
                        consecutive_nonfinite_backprop_steps = 0
                        if accelerator.sync_gradients and grad_clip and grad_clip > 0:
                            accelerator.clip_grad_norm_(model.parameters(), grad_clip)
                        if accelerator.sync_gradients:
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad(set_to_none=True)

                if has_nonfinite_loss or has_nonfinite_grad:
                    had_nonfinite_backprop = True
                    if has_nonfinite_grad:
                        had_nonfinite_gradient = True
                    nonfinite_backprop_events += 1
                    consecutive_nonfinite_backprop_steps += 1
                    reason = "loss" if has_nonfinite_loss else "grad"
                    detail = ""
                    if local_nonfinite_loss:
                        detail = " local=loss"
                    elif local_nonfinite_grad_name:
                        detail = f" local_grad={local_nonfinite_grad_name}"
                    console.tag_print(
                        "nonfinite-backprop",
                        f"epoch={epoch}/{total_epochs} "
                        f"loader_step={loader_step}/{len(loader)} "
                        f"update_step={epoch_update_step}/{steps_per_epoch} "
                        f"global_step={global_step} "
                        f"reason={reason} "
                        f"consecutive_steps={consecutive_nonfinite_backprop_steps}"
                        f"{detail}",
                    )
                    maybe_log_metrics(
                        accelerator,
                        {
                            "nonfinite_backprop/event": 1.0,
                            "nonfinite_backprop/is_loss": float(has_nonfinite_loss),
                            "nonfinite_backprop/is_grad": float(has_nonfinite_grad),
                            "nonfinite_backprop/consecutive_steps": float(consecutive_nonfinite_backprop_steps),
                            "nonfinite_backprop/epoch": float(epoch),
                            "nonfinite_backprop/loader_step": float(loader_step),
                            "nonfinite_backprop/global_step": float(global_step),
                        },
                        step=global_step,
                    )
                    if (
                        nonfinite_backprop_step_patience > 0
                        and consecutive_nonfinite_backprop_steps >= nonfinite_backprop_step_patience
                    ):
                        stop_reason = NONFINITE_BACKPROP_STEP_STOP
                        console.tag_print(
                            "nonfinite-stop",
                            f"triggered by consecutive steps at "
                            f"epoch={epoch} loader_step={loader_step} "
                            f"consecutive_steps={consecutive_nonfinite_backprop_steps} "
                            f"patience={nonfinite_backprop_step_patience}",
                        )
                        break
                    continue

                accs = compute_accuracy(outputs, labels)

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
                console.tag_print(
                    "train",
                    f"epoch={epoch}/{total_epochs} "
                    f"step={epoch_update_step}/{steps_per_epoch} "
                    f"global_step={global_step} "
                    f"loss={window_metrics['loss']:.4f} "
                    f"acc_full={window_metrics['acc_expression']:.4f} "
                    f"acc_dl={window_metrics['acc_digit_left']:.4f} "
                    f"acc_op={window_metrics['acc_operator']:.4f} "
                    f"acc_dr={window_metrics['acc_digit_right']:.4f} "
                    f"lr={lr:.2e} "
                    f"throughput={samples_per_s:.1f}img/s "
                    f"eta={format_seconds(eta_seconds)}",
                )

            maybe_log_metrics(
                accelerator,
                {
                    "step/epoch": float(epoch),
                    "step/epoch_step": float(epoch_update_step),
                    "step/global_step": float(global_step),
                    "step/train/lr": lr,
                    "step/train/samples_per_s": samples_per_s,
                    **prefix_metrics("step/train/", window_metrics),
                },
                step=global_step,
            )
            window_sum = make_metric_sum()
            window_n = 0
            last_log_time = now

        if stop_reason is not None:
            if progress is not None:
                progress.refresh()
            metrics, _ = reduce_metric_sums(accelerator, metric_sum, n)
            return TrainEpochResult(
                metrics=metrics,
                global_step=global_step,
                had_nonfinite_backprop=had_nonfinite_backprop,
                had_nonfinite_gradient=had_nonfinite_gradient,
                nonfinite_backprop_events=nonfinite_backprop_events,
                consecutive_nonfinite_backprop_steps=consecutive_nonfinite_backprop_steps,
                stop_reason=stop_reason,
            )

    metrics, _ = reduce_metric_sums(accelerator, metric_sum, n)
    return TrainEpochResult(
        metrics=metrics,
        global_step=global_step,
        had_nonfinite_backprop=had_nonfinite_backprop,
        had_nonfinite_gradient=had_nonfinite_gradient,
        nonfinite_backprop_events=nonfinite_backprop_events,
        consecutive_nonfinite_backprop_steps=consecutive_nonfinite_backprop_steps,
    )


# ----------------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # 1) 配置组装
    cfg = load_config(args.config) if args.config else FullConfig()
    cfg = apply_env_overrides(cfg)
    cfg = merge_args_to_config(cfg, args)
    report_to, tracker_resolution = resolve_report_to(cfg.train.report_to)
    ensure_wandb_run_name(cfg, report_to)
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
            init_kwargs=init_kwargs,
        )
        sync_wandb_config(accelerator, cfg)

    effective_batch_size = (
        cfg.train.per_device_batch_size
        * accelerator.num_processes
        * cfg.train.gradient_accumulation_steps
    )
    console = AcceleratorConsole(accelerator)

    console.tag_print(
        "init",
        f"rank={accelerator.process_index} world_size={accelerator.num_processes} "
        f"device={accelerator.device} mixed_precision={cfg.train.mixed_precision} "
        f"grad_accum={cfg.train.gradient_accumulation_steps}",
    )
    if accelerator.is_main_process:
        maybe_log_metrics(
            accelerator,
            {
                "run/world_size": float(accelerator.num_processes),
                "run/effective_batch_size": float(effective_batch_size),
            },
            step=0,
        )
        console.tag_print(
            "config",
            f"output={cfg.train.output_dir} data={cfg.data.data_root} "
            f"backbone={cfg.model.backbone} batch/device={cfg.train.per_device_batch_size} "
            f"effective_batch={effective_batch_size} tracker={report_to or 'none'} "
            f"(raw={cfg.train.report_to}, reason={tracker_resolution})",
        )
        console.tag_print("config", f"{cfg_to_dict(cfg)}")

    # 3) 数据 (按 manifest 读取 train/val/test)
    train_ds = CaptchaPairDataset(
        data_root=cfg.data.data_root,
        image_size_h=cfg.data.image_size_h,
        image_size_w=cfg.data.image_size_w,
        threshold=cfg.data.threshold,
        binarize_mode=cfg.data.binarize_mode,
        adaptive_block_size=cfg.data.adaptive_block_size,
        adaptive_c=cfg.data.adaptive_c,
        augmentation=cfg.data.augmentation,
        enable_augmentation=True,
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
            console.tag_print("data", f"test split unavailable: {e}; skipping")
            test_ds = None
    console.tag_print(
        "data",
        f"train={len(train_ds)} val={len(val_ds)} "
        f"test={len(test_ds) if test_ds else 0} "
        f"image_size=({cfg.data.image_size_h},{cfg.data.image_size_w}) "
        f"binarize={cfg.data.binarize_mode}",
    )
    if accelerator.is_main_process:
        maybe_log_metrics(
            accelerator,
            {
                "data/train_samples": float(len(train_ds)),
                "data/val_samples": float(len(val_ds)),
                "data/test_samples": float(len(test_ds) if test_ds else 0),
            },
            step=0,
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
    model_metadata = build_model_metadata(cfg_to_dict(cfg))
    model = build_model_from_config(
        cfg_to_dict(cfg),
        num_digit_classes=NUM_DIGIT_CLASSES,
        num_operator_classes=NUM_OPERATOR_CLASSES,
    )
    console.tag_print(
        "model",
        f"version={model_metadata['version']} "
        f"family={model_metadata['family']} "
        f"backbone={cfg.model.backbone} "
        f"pretrained={cfg.model.pretrained} "
        f"weights={describe_backbone_weights(cfg.model.backbone, cfg.model.pretrained)} "
        f"asset_stem={model_metadata['asset_stem']} "
        f"gray_stem=rgb_mean",
    )
    if accelerator.is_main_process:
        model_stats = collect_model_stats(model, cfg.data.image_size_h, cfg.data.image_size_w)
        console.tag_print(
            "model-stats",
            f"{format_model_stats(model_stats)}",
        )
        maybe_log_metrics(
            accelerator,
            {
                "model/params": float(model_stats.total_params),
                "model/trainable_params": float(model_stats.trainable_params),
                "model/params_m": model_stats.total_params / 1_000_000,
                "model/trainable_params_m": model_stats.trainable_params / 1_000_000,
                **(
                    {
                        "model/flops": float(model_stats.flops),
                        "model/flops_g": model_stats.flops / 1_000_000_000,
                        "model/flops_m": model_stats.flops / 1_000_000,
                    }
                    if model_stats.flops is not None
                    else {}
                ),
            },
            step=0,
        )
        sync_wandb_model_stats(accelerator, model_stats)
    loss_fn = TriSlotDecoderLoss(
        weights=LossWeights(
            digit_left=cfg.loss.weight_digit_left,
            operator=cfg.loss.weight_operator,
            digit_right=cfg.loss.weight_digit_right,
            slot_order=cfg.loss.weight_slot_order,
            slot_overlap=cfg.loss.weight_slot_overlap,
            slot_right_boundary=(
                cfg.loss.weight_slot_right_boundary if cfg.loss.enable_slot_right_boundary else 0.0
            ),
            slot_attention_variance=(
                cfg.loss.weight_slot_attention_variance if cfg.loss.enable_slot_attention_variance else 0.0
            ),
        ),
        label_smoothing=cfg.loss.label_smoothing,
        focal_gamma=cfg.loss.focal_gamma,
        slot_margin=cfg.loss.slot_margin,
        slot_right_boundary_max=cfg.loss.slot_right_boundary_max,
        slot_attention_max_variance=cfg.loss.slot_attention_max_variance,
        operator_class_weights=(
            cfg.loss.operator_class_weights if cfg.loss.enable_operator_class_balance else None
        ),
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
    best_epoch = 0
    best_val_loss = float("inf")
    best_val_loss_epoch = 0
    global_step = 0
    epochs_without_improve = 0
    consecutive_nonfinite_backprop_steps = 0
    consecutive_nonfinite_backprop_epochs = 0
    metrics_history: list[dict[str, Any]] = load_metrics_history(cfg.train.output_dir)
    early_stop_patience = resolve_early_stop_patience(cfg)
    nonfinite_backprop_step_patience = resolve_nonfinite_backprop_patience(
        "train.nonfinite_backprop_step_patience",
        cfg.train.nonfinite_backprop_step_patience,
    )
    nonfinite_backprop_epoch_patience = resolve_nonfinite_backprop_patience(
        "train.nonfinite_backprop_epoch_patience",
        cfg.train.nonfinite_backprop_epoch_patience,
    )
    if early_stop_patience == 0:
        console.tag_print("early-stop", "disabled (train.early_stop_patience=0)")
    elif cfg.train.early_stop_patience == -1:
        console.tag_print(
            "early-stop",
            f"enabled raw=-1 resolved_patience={early_stop_patience} "
            f"(20% of epochs={cfg.train.epochs})",
        )
    else:
        console.tag_print("early-stop", f"enabled patience={early_stop_patience}")
    if nonfinite_backprop_step_patience == 0 and nonfinite_backprop_epoch_patience == 0:
        console.tag_print("nonfinite-stop", "disabled (both patience=0)")
    else:
        console.tag_print(
            "nonfinite-stop",
            f"step_patience={nonfinite_backprop_step_patience} "
            f"epoch_patience={nonfinite_backprop_epoch_patience}",
        )
    if cfg.train.resume_from:
        ckpt = torch.load(cfg.train.resume_from, map_location="cpu")
        model_state = ckpt.get("model_state_dict")
        if model_state is not None:
            accelerator.unwrap_model(model).load_state_dict(model_state)
        optimizer_state = ckpt.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        scheduler_state = ckpt.get("scheduler_state_dict")
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("best_acc", -1.0)
        best_epoch = int(ckpt.get("best_epoch", 0))
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        best_val_loss_epoch = int(ckpt.get("best_val_loss_epoch", 0))
        global_step = ckpt.get("global_step", start_epoch * steps_per_epoch)
        consecutive_nonfinite_backprop_steps = int(ckpt.get("consecutive_nonfinite_backprop_steps", 0))
        consecutive_nonfinite_backprop_epochs = int(ckpt.get("consecutive_nonfinite_backprop_epochs", 0))
        if not metrics_history:
            metrics_history = ckpt.get("metrics_history", [])
        if "epochs_without_improve" in ckpt:
            epochs_without_improve = int(ckpt.get("epochs_without_improve", 0))
        elif metrics_history:
            inferred_best_acc, inferred_stale_epochs, inferred_best_epoch = infer_early_stop_state(metrics_history)
            best_acc = max(best_acc, inferred_best_acc)
            if best_epoch <= 0:
                best_epoch = inferred_best_epoch
            epochs_without_improve = inferred_stale_epochs
        if metrics_history and (not math.isfinite(best_val_loss) or best_val_loss == float("inf")):
            best_val_loss, best_val_loss_epoch = infer_best_val_loss_state(metrics_history)
        console.tag_print(
            "resume",
            f"from {cfg.train.resume_from} epoch={start_epoch} "
            f"best={best_acc:.4f} best_epoch={best_epoch} "
            f"best_val_loss={best_val_loss:.4f} best_val_loss_epoch={best_val_loss_epoch} "
            f"stale_epochs={epochs_without_improve} "
            f"nonfinite_steps={consecutive_nonfinite_backprop_steps} "
            f"nonfinite_epochs={consecutive_nonfinite_backprop_epochs}",
        )
    elif metrics_history:
        inferred_best_acc, inferred_stale_epochs, inferred_best_epoch = infer_early_stop_state(metrics_history)
        best_acc = max(best_acc, inferred_best_acc)
        best_epoch = inferred_best_epoch
        best_val_loss, best_val_loss_epoch = infer_best_val_loss_state(metrics_history)
        epochs_without_improve = inferred_stale_epochs

    if early_stop_patience > 0 and epochs_without_improve >= early_stop_patience:
        console.tag_print(
            "early-stop",
            f"checkpoint already reached patience: "
            f"stale_epochs={epochs_without_improve} patience={early_stop_patience}; skip training",
        )
        accelerator.wait_for_everyone()
        accelerator.end_training()
        return
    if (
        nonfinite_backprop_step_patience > 0
        and consecutive_nonfinite_backprop_steps >= nonfinite_backprop_step_patience
    ):
        console.tag_print(
            "nonfinite-stop",
            f"checkpoint already reached step patience: "
            f"consecutive_steps={consecutive_nonfinite_backprop_steps} "
            f"patience={nonfinite_backprop_step_patience}; skip training",
        )
        accelerator.wait_for_everyone()
        accelerator.end_training()
        return
    if (
        nonfinite_backprop_epoch_patience > 0
        and consecutive_nonfinite_backprop_epochs >= nonfinite_backprop_epoch_patience
    ):
        console.tag_print(
            "nonfinite-stop",
            f"checkpoint already reached epoch patience: "
            f"consecutive_epochs={consecutive_nonfinite_backprop_epochs} "
            f"patience={nonfinite_backprop_epoch_patience}; skip training",
        )
        accelerator.wait_for_everyone()
        accelerator.end_training()
        return

    # 7) 训练循环
    enable_rich_progress = should_use_rich_progress(accelerator, cfg)
    for epoch in range(start_epoch, cfg.train.epochs):
        console.rule(f"Epoch {epoch + 1}/{cfg.train.epochs}", style="bold blue")
        t0 = time.time()

        train_result = train_one_epoch(
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
            nonfinite_backprop_step_patience=nonfinite_backprop_step_patience,
            consecutive_nonfinite_backprop_steps=consecutive_nonfinite_backprop_steps,
            console=console,
        )
        train_metrics = train_result.metrics
        global_step = train_result.global_step
        consecutive_nonfinite_backprop_steps = train_result.consecutive_nonfinite_backprop_steps
        epoch_had_nonfinite_backprop = train_result.had_nonfinite_backprop
        epoch_had_nonfinite_gradient = train_result.had_nonfinite_gradient
        if epoch_had_nonfinite_backprop:
            consecutive_nonfinite_backprop_epochs += 1
        else:
            consecutive_nonfinite_backprop_epochs = 0
        nonfinite_epoch_stop_triggered = (
            nonfinite_backprop_epoch_patience > 0
            and consecutive_nonfinite_backprop_epochs >= nonfinite_backprop_epoch_patience
        )
        train_time = time.time() - t0

        if train_result.stop_reason == NONFINITE_BACKPROP_STEP_STOP:
            stop_reason = train_result.stop_reason
            console.print(
                render_epoch_summary(
                    epoch=epoch + 1,
                    total_epochs=cfg.train.epochs,
                    train_metrics=train_metrics,
                    train_time=train_time,
                    nonfinite_events=train_result.nonfinite_backprop_events,
                    nonfinite_steps=consecutive_nonfinite_backprop_steps,
                    nonfinite_epochs=consecutive_nonfinite_backprop_epochs,
                    stop_reason=stop_reason,
                )
            )
            maybe_log_metrics(
                accelerator,
                {
                    "epoch/index": float(epoch + 1),
                    "epoch/global_step": float(global_step),
                    "epoch/time_s": train_time,
                    "epoch/train/lr": scheduler.get_last_lr()[0],
                    "epoch/train/samples": float(len(train_ds)),
                    "nonfinite_backprop/events": float(train_result.nonfinite_backprop_events),
                    "nonfinite_backprop/consecutive_steps": float(consecutive_nonfinite_backprop_steps),
                    "nonfinite_backprop/consecutive_epochs": float(consecutive_nonfinite_backprop_epochs),
                    "nonfinite_backprop/triggered": 1.0,
                    **prefix_metrics("epoch/train/", train_metrics),
                },
                step=global_step,
            )
            if accelerator.is_main_process:
                epoch_record = {
                    "epoch": epoch + 1,
                    "total_epochs": cfg.train.epochs,
                    "global_step": global_step,
                    "best_val_acc": best_acc,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "best_val_loss_epoch": best_val_loss_epoch,
                    "is_best": False,
                    "epochs_without_improve": epochs_without_improve,
                    "early_stop_patience": early_stop_patience,
                    "early_stop_triggered": False,
                    "nonfinite_backprop_events": train_result.nonfinite_backprop_events,
                    "epoch_had_nonfinite_backprop": epoch_had_nonfinite_backprop,
                    "epoch_had_nonfinite_gradient": epoch_had_nonfinite_gradient,
                    "consecutive_nonfinite_backprop_steps": consecutive_nonfinite_backprop_steps,
                    "consecutive_nonfinite_backprop_epochs": consecutive_nonfinite_backprop_epochs,
                    "nonfinite_backprop_step_patience": nonfinite_backprop_step_patience,
                    "nonfinite_backprop_epoch_patience": nonfinite_backprop_epoch_patience,
                    "stop_reason": stop_reason,
                    "time_s": train_time,
                    "train": train_metrics,
                    "val": None,
                    "test": None,
                    "config": cfg_to_dict(cfg),
                }
                metrics_history = save_epoch_metrics(
                    cfg.train.output_dir,
                    epoch_record,
                    metrics_history,
                )
                save_checkpoint(
                    accelerator=accelerator,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    cfg=cfg,
                    metrics=train_metrics,
                    metrics_stage="train",
                    is_best=False,
                    best_acc=best_acc,
                    best_epoch=best_epoch,
                    best_val_loss=best_val_loss,
                    best_val_loss_epoch=best_val_loss_epoch,
                    epochs_without_improve=epochs_without_improve,
                    consecutive_nonfinite_backprop_steps=consecutive_nonfinite_backprop_steps,
                    consecutive_nonfinite_backprop_epochs=consecutive_nonfinite_backprop_epochs,
                    early_stop_patience=early_stop_patience,
                    early_stop_triggered=False,
                    stop_reason=stop_reason,
                    metrics_history=metrics_history,
                    save_latest=not epoch_had_nonfinite_gradient,
                )
            break

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
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_val_loss_epoch = epoch + 1
        if is_best:
            epochs_without_improve = 0
            best_epoch = epoch + 1
        else:
            epochs_without_improve += 1
        best_acc = max(best_acc, val_metrics["acc_expression"])
        early_stop_triggered = (
            early_stop_patience > 0 and epochs_without_improve >= early_stop_patience
        )
        nonfinite_stop_triggered = nonfinite_epoch_stop_triggered
        if nonfinite_stop_triggered:
            console.tag_print(
                "nonfinite-stop",
                f"triggered by consecutive epochs at "
                f"epoch={epoch + 1} consecutive_epochs={consecutive_nonfinite_backprop_epochs} "
                f"patience={nonfinite_backprop_epoch_patience}",
            )
        console.print(
            render_epoch_summary(
                epoch=epoch + 1,
                total_epochs=cfg.train.epochs,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                train_time=train_time,
                best_acc=best_acc,
                best_epoch=best_epoch,
                best_val_loss=best_val_loss,
                best_val_loss_epoch=best_val_loss_epoch,
                is_best=is_best,
                stale_epochs=epochs_without_improve,
                nonfinite_events=train_result.nonfinite_backprop_events,
                nonfinite_steps=consecutive_nonfinite_backprop_steps,
                nonfinite_epochs=consecutive_nonfinite_backprop_epochs,
            )
        )
        maybe_log_metrics(
            accelerator,
            {
                "epoch/index": float(epoch + 1),
                "epoch/global_step": float(global_step),
                "epoch/time_s": train_time,
                "epoch/train/lr": scheduler.get_last_lr()[0],
                "epoch/train/samples": float(len(train_ds)),
                "epoch/val/samples": float(len(val_ds)),
                "epoch/is_best": float(is_best),
                "best/epoch": float(best_epoch),
                "best/val/acc_expression": best_acc,
                "best/val/loss": best_val_loss,
                "best/val/loss_epoch": float(best_val_loss_epoch),
                "epoch/gap/acc_expression": train_metrics["acc_expression"] - val_metrics["acc_expression"],
                "epoch/gap/loss": train_metrics["loss"] - val_metrics["loss"],
                "early_stop/patience": float(early_stop_patience),
                "early_stop/stale_epochs": float(epochs_without_improve),
                "early_stop/triggered": float(early_stop_triggered),
                "nonfinite_backprop/events": float(train_result.nonfinite_backprop_events),
                "nonfinite_backprop/epoch_flag": float(epoch_had_nonfinite_backprop),
                "nonfinite_backprop/consecutive_steps": float(consecutive_nonfinite_backprop_steps),
                "nonfinite_backprop/consecutive_epochs": float(consecutive_nonfinite_backprop_epochs),
                "nonfinite_backprop/step_patience": float(nonfinite_backprop_step_patience),
                "nonfinite_backprop/epoch_patience": float(nonfinite_backprop_epoch_patience),
                "nonfinite_backprop/triggered": float(nonfinite_stop_triggered),
                **prefix_metrics("epoch/train/", train_metrics),
                **prefix_metrics("epoch/val/", val_metrics),
            },
            step=global_step,
        )

        # 最后一个 epoch 或触发 early stop 时跑 test 集, 给出最终泛化指标
        is_last = (epoch + 1) == cfg.train.epochs
        is_final_epoch = is_last or early_stop_triggered
        test_metrics: dict[str, float] | None = None
        if is_final_epoch and not nonfinite_stop_triggered and test_loader is not None:
            test_metrics = evaluate(
                accelerator,
                model,
                test_loader,
                loss_fn,
                stage="test",
                enable_rich_progress=enable_rich_progress,
            )
            console.tag_print(
                "test",
                f"final "
                f"loss={test_metrics['loss']:.4f} "
                f"acc_dl={test_metrics['acc_digit_left']:.4f} "
                f"acc_op={test_metrics['acc_operator']:.4f} "
                f"acc_dr={test_metrics['acc_digit_right']:.4f} "
                f"acc_full={test_metrics['acc_expression']:.4f}",
            )
            maybe_log_metrics(
                accelerator,
                {
                    "epoch/test/samples": float(len(test_ds)),
                    **prefix_metrics("epoch/test/", test_metrics),
                },
                step=global_step,
            )

        if accelerator.is_main_process:
            stop_reason = None
            if nonfinite_stop_triggered:
                stop_reason = NONFINITE_BACKPROP_EPOCH_STOP
            elif early_stop_triggered:
                stop_reason = "early_stop"
            elif is_last:
                stop_reason = "max_epochs"
            epoch_record: dict[str, Any] = {
                "epoch": epoch + 1,
                "total_epochs": cfg.train.epochs,
                "global_step": global_step,
                "best_val_acc": best_acc,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_val_loss_epoch": best_val_loss_epoch,
                "is_best": is_best,
                "epochs_without_improve": epochs_without_improve,
                "early_stop_patience": early_stop_patience,
                "early_stop_triggered": early_stop_triggered,
                "nonfinite_backprop_events": train_result.nonfinite_backprop_events,
                "epoch_had_nonfinite_backprop": epoch_had_nonfinite_backprop,
                "epoch_had_nonfinite_gradient": epoch_had_nonfinite_gradient,
                "consecutive_nonfinite_backprop_steps": consecutive_nonfinite_backprop_steps,
                "consecutive_nonfinite_backprop_epochs": consecutive_nonfinite_backprop_epochs,
                "nonfinite_backprop_step_patience": nonfinite_backprop_step_patience,
                "nonfinite_backprop_epoch_patience": nonfinite_backprop_epoch_patience,
                "stop_reason": stop_reason,
                "time_s": train_time,
                "train": train_metrics,
                "val": val_metrics,
                "test": test_metrics,
                "config": cfg_to_dict(cfg),
            }
            metrics_history = save_epoch_metrics(
                cfg.train.output_dir,
                epoch_record,
                metrics_history,
            )

        # 8) checkpoint (rank 0 only)
        should_save_checkpoint = (
            (epoch + 1) % cfg.train.save_every_n_epochs == 0
            or is_final_epoch
            or nonfinite_stop_triggered
        )
        if accelerator.is_main_process and should_save_checkpoint:
            save_checkpoint(
                accelerator=accelerator,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                cfg=cfg,
                metrics=val_metrics,
                metrics_stage="val",
                is_best=is_best,
                best_acc=best_acc,
                best_epoch=best_epoch,
                best_val_loss=best_val_loss,
                best_val_loss_epoch=best_val_loss_epoch,
                epochs_without_improve=epochs_without_improve,
                consecutive_nonfinite_backprop_steps=consecutive_nonfinite_backprop_steps,
                consecutive_nonfinite_backprop_epochs=consecutive_nonfinite_backprop_epochs,
                early_stop_patience=early_stop_patience,
                early_stop_triggered=early_stop_triggered,
                stop_reason=stop_reason,
                metrics_history=metrics_history,
                save_latest=not epoch_had_nonfinite_gradient,
            )

        if nonfinite_stop_triggered:
            break
        if early_stop_triggered:
            console.tag_print(
                "early-stop",
                f"triggered at epoch={epoch + 1} "
                f"stale_epochs={epochs_without_improve} patience={early_stop_patience}",
            )
            break

    console.success("training complete")
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
    loss_fn: TriSlotDecoderLoss,
    *,
    stage: str = "val",
    enable_rich_progress: bool = False,
) -> dict[str, float]:
    model.eval()
    metric_sum = make_metric_sum()
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_pytorch_release_manifest(
    *,
    output_dir: Path,
    checkpoint_path: Path,
    model_metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    digest_path = checkpoint_path.parent / "SHA256SUMS.txt"
    release_files: list[Path] = [checkpoint_path]
    files = [
        {
            "path": checkpoint_path.relative_to(output_dir).as_posix(),
            "release_asset_name": checkpoint_path.name,
            "sha256": sha256_file(checkpoint_path),
        }
    ]
    pip_list = extract_checkpoint_pip_list(torch.load(checkpoint_path, map_location="cpu"))
    if pip_list is not None:
        pip_list_path = checkpoint_path.with_name(f"{model_metadata['asset_stem']}.pip-list.json")
        write_pip_list_json(pip_list_path, pip_list)
        release_files.append(pip_list_path)
        files.append(
            {
                "path": pip_list_path.relative_to(output_dir).as_posix(),
                "release_asset_name": pip_list_path.name,
                "sha256": sha256_file(pip_list_path),
            }
        )

    digest_lines = [f"{sha256_file(path)}  {path.name}" for path in release_files]
    digest_path.write_text("\n".join(digest_lines) + "\n", encoding="utf-8")
    manifest_path = output_dir / "model-assets.json"
    manifest = {
        "schema_version": 1,
        "models": [model_metadata],
        "artifacts": [
            {
                **model_metadata,
                "engine": "pytorch",
                "precision": "fp32",
                "format": "checkpoint",
                "files": files,
            }
        ],
        "digests": [
            {
                "engine": "pytorch",
                "path": digest_path.relative_to(output_dir).as_posix(),
                "release_asset_name": digest_path.name,
                "sha256": sha256_file(digest_path),
            }
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def save_checkpoint(
    accelerator: Accelerator,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
    cfg: FullConfig,
    metrics: dict[str, float],
    metrics_stage: str,
    is_best: bool,
    best_acc: float,
    best_epoch: int,
    best_val_loss: float,
    best_val_loss_epoch: int,
    epochs_without_improve: int,
    consecutive_nonfinite_backprop_steps: int,
    consecutive_nonfinite_backprop_epochs: int,
    early_stop_patience: int,
    early_stop_triggered: bool,
    stop_reason: str | None,
    metrics_history: list[dict[str, Any]],
    save_latest: bool = True,
) -> None:
    """rank 0 写盘. DDP unwrap 后存 model_state_dict.

    文件:
        output_dir/last.pt
        output_dir/best.pt (仅当 is_best)
    """
    _console = AcceleratorConsole(accelerator)
    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    unwrapped = accelerator.unwrap_model(model)
    cfg_dict = cfg_to_dict(cfg)
    model_metadata = build_model_metadata(cfg_dict)
    try:
        pip_list, pip_list_metadata = capture_pip_list_snapshot()
    except Exception as exc:
        pip_list = None
        pip_list_metadata = None
        _console.tag_print("WARN", f"capture pip list failed: {exc}")
    state = {
        "epoch": epoch,
        "model_state_dict": unwrapped.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "global_step": global_step,
        "metrics": metrics,
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_loss_epoch": best_val_loss_epoch,
        "epochs_without_improve": epochs_without_improve,
        "consecutive_nonfinite_backprop_steps": consecutive_nonfinite_backprop_steps,
        "consecutive_nonfinite_backprop_epochs": consecutive_nonfinite_backprop_epochs,
        "early_stop_patience": early_stop_patience,
        "early_stop_triggered": early_stop_triggered,
        "stop_reason": stop_reason,
        "metrics_history": metrics_history,
        "config": cfg_dict,
        "model_metadata": model_metadata,
    }
    if pip_list is not None and pip_list_metadata is not None:
        state["pip_list"] = pip_list
        state["pip_list_metadata"] = pip_list_metadata
    last_path = output_dir / "last.pt"
    if save_latest:
        accelerator.save(state, str(last_path))
        _console.tag_print(
            "ckpt",
            f"saved {last_path} "
            f"(epoch={epoch + 1}, {metrics_stage}_acc={metrics['acc_expression']:.4f})",
        )
    else:
        _console.tag_print(
            "ckpt",
            f"skip {last_path} because non-finite gradient was detected in epoch={epoch + 1}",
        )

    if is_best:
        best_path = output_dir / "best.pt"
        release_root = output_dir / "release"
        release_pytorch_dir = release_root / "pytorch"
        release_pytorch_dir.mkdir(parents=True, exist_ok=True)
        release_path = release_pytorch_dir / f"{model_metadata['asset_stem']}.pt"
        accelerator.save(state, str(best_path))
        accelerator.save(state, str(release_path))
        write_pytorch_release_manifest(
            output_dir=release_root,
            checkpoint_path=release_path,
            model_metadata=model_metadata,
        )
        _console.tag_print("ckpt", f"NEW BEST -> {best_path} (val_acc={best_acc:.4f})")
        _console.tag_print("ckpt", f"release  -> {release_path}")


if __name__ == "__main__":
    main()
