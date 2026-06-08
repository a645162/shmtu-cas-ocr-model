"""模型统计工具: 参数量 / FLOPs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode


@dataclass
class ModelStats:
    total_params: int
    trainable_params: int
    flops: Optional[int]
    input_shape: tuple[int, ...]


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """返回 (总参数量, 可训练参数量)."""
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def _infer_model_device(model: nn.Module) -> torch.device:
    for tensor in model.parameters():
        return tensor.device
    for tensor in model.buffers():
        return tensor.device
    return torch.device("cpu")


def estimate_flops(
    model: nn.Module,
    input_shape: tuple[int, ...],
) -> Optional[int]:
    """估算单次 forward FLOPs; 失败时返回 None."""
    device = _infer_model_device(model)
    was_training = model.training
    try:
        model.eval()
        dummy = torch.randn(*input_shape, device=device, dtype=torch.float32)
        with torch.no_grad():
            with FlopCounterMode(display=False) as counter:
                model(dummy)
        return int(counter.get_total_flops())
    except Exception:
        return None
    finally:
        model.train(was_training)


def collect_model_stats(
    model: nn.Module,
    image_size_h: int,
    image_size_w: int,
    batch_size: int = 1,
) -> ModelStats:
    """统计参数量与 FLOPs."""
    input_shape = (batch_size, 1, image_size_h, image_size_w)
    total_params, trainable_params = count_parameters(model)
    flops = estimate_flops(model, input_shape=input_shape)
    return ModelStats(
        total_params=total_params,
        trainable_params=trainable_params,
        flops=flops,
        input_shape=input_shape,
    )


def format_params_m(num_params: int) -> str:
    """参数量固定按 M 显示."""
    return f"{num_params / 1_000_000:.2f}M"


def format_flops(num_flops: Optional[int]) -> str:
    """按 MFLOPs / GFLOPs 显示单次 forward 计算量."""
    if num_flops is None:
        return "N/A"
    if num_flops >= 1_000_000_000:
        return f"{num_flops / 1_000_000_000:.4f} GFLOPs"
    if num_flops >= 1_000_000:
        return f"{num_flops / 1_000_000:.2f} MFLOPs"
    if num_flops >= 1_000:
        return f"{num_flops / 1_000:.2f} KFLOPs"
    return f"{num_flops} FLOPs"


def format_model_stats(stats: ModelStats) -> str:
    """格式化成统一日志字符串."""
    return (
        f"params={format_params_m(stats.total_params)} "
        f"trainable={format_params_m(stats.trainable_params)} "
        f"flops={format_flops(stats.flops)} "
        f"input={stats.input_shape}"
    )
