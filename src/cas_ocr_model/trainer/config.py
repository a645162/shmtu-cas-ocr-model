"""训练配置 — dataclass 形式, 兼容 YAML 反序列化.

支持两种用法:
    1) 命令行参数 (train.py 入口) 直接覆盖
    2) 加载 YAML 配置文件 (configs/*.yaml) 后用 CLI 覆盖
"""
from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from cas_ocr_model.expression import CANONICAL_OPERATOR_LABELS
from cas_ocr_model.preprocess_ops import BINARIZE_MODES

# ----------------------------------------------------------------------------
# 标签字典
# ----------------------------------------------------------------------------

# 数字 0-9
DIGIT_LABELS: list[str] = [str(i) for i in range(10)]  # 10 类
DIGIT2IDX = {s: i for i, s in enumerate(DIGIT_LABELS)}

# 运算符 3 类. 数据集中的 "加/减/乘" 与 "+/-/*" 会在 expression 解析阶段自动归一.
OPERATOR_LABELS: list[str] = list(CANONICAL_OPERATOR_LABELS)
OP2IDX = {s: i for i, s in enumerate(OPERATOR_LABELS)}

NUM_DIGIT_CLASSES = len(DIGIT_LABELS)   # 10
NUM_OPERATOR_CLASSES = len(OPERATOR_LABELS)  # 3


# ----------------------------------------------------------------------------
# 配置项
# ----------------------------------------------------------------------------


@dataclass
class DataConfig:
    """数据相关配置."""

    data_root: str = "./dataset"
    """数据集根目录, 内部包含 00000000.jpg / 00000000.json 配对."""

    image_size_h: int = 64
    """resize 高度. 验证码图片普遍较矮 (30~60 px), 64 是合理上限."""

    image_size_w: int = 192
    """resize 宽度. 三段 (数字+运算符+数字) 等宽, 每段约 64 px."""

    threshold: int = 200
    """二值化阈值. 与 v1/configs/defaults.thresh 对齐."""

    binarize_mode: str = "min_channel_otsu"
    """二值化模式. 随机颜色验证码推荐 min_channel_otsu."""

    adaptive_block_size: int = 25
    """adaptive 模式邻域窗口大小, 必须为奇数."""

    adaptive_c: int = 15
    """adaptive 模式阈值偏移."""

    train_ratio: float = 0.9
    """训练/验证划分 (按文件数, 顺序切片; 收集器生成时已随机)."""

    num_workers: int = 4
    """DataLoader 工作进程数, 每张 GPU 各自独立."""

    pin_memory: bool = True
    """半精度传输加速."""


@dataclass
class ModelConfig:
    """模型结构配置."""

    backbone: str = "resnet18"
    """backbone 名称 (当前支持 resnet18 / resnet34)."""

    pretrained: bool = True
    """是否加载 ImageNet 预训练权重. 验证码数据少, 强烈建议 True."""

    dropout: float = 0.2
    """head 之前的 dropout 概率."""

    slot_hidden_dim: int = 256
    """宽度序列特征维度. 验证码位置固定, 这个中间维度决定 3 个槽位的表达能力."""

    slot_attention_heads: int = 4
    """3 个槽位查询宽度序列时的注意力头数."""


@dataclass
class TrainConfig:
    """训练超参."""

    output_dir: str = "./runs/exp1"
    """输出目录, 包含 checkpoints / logs / best.pt."""

    seed: int = 42
    """随机种子 (Python / NumPy / PyTorch / CUDA)."""

    epochs: int = 30
    """总 epoch 数."""

    per_device_batch_size: int = 256
    """单卡 batch size. 8 卡 x 256 = 2048 effective."""

    learning_rate: float = 1e-3
    """初始学习率 (AdamW). 8 卡 DDP 一般按 base lr * world_size 线性缩放."""

    weight_decay: float = 1e-4
    warmup_ratio: float = 0.05
    """线性 warmup 占总步数比例."""

    grad_clip: float = 1.0
    """梯度裁剪阈值 (L2 norm)."""

    log_every_n_steps: int = 20
    """accelerate.print 日志间隔."""

    save_every_n_epochs: int = 1
    """每隔 N epoch 保存一次 (rank 0)."""

    mixed_precision: str = "fp16"
    """混合精度: 'no' / 'fp16' / 'bf16'. 默认 fp16 (与用户偏好对齐)."""

    gradient_accumulation_steps: int = 1
    """梯度累积, 进一步放大 effective batch."""

    resume_from: Optional[str] = None
    """checkpoint 路径, 断点续训."""


@dataclass
class LossConfig:
    """3-head 损失权重."""

    weight_digit_left: float = 1.0
    weight_operator: float = 1.0
    weight_digit_right: float = 1.0
    label_smoothing: float = 0.0
    """标签平滑, 0~0.1 之间. 验证码单字识别建议 0.05."""

    focal_gamma: float = 0.0
    """Focal gamma. 难样本较多时可设为 1~2, 默认关闭以保持稳定."""

    weight_slot_order: float = 0.1
    """槽位中心顺序约束权重: 左数字 < 运算符 < 右数字."""

    weight_slot_overlap: float = 0.05
    """槽位注意力去重权重, 减少三个头盯住同一段区域."""

    slot_margin: float = 0.10
    """槽位中心最小间距, 取值范围基于归一化宽度坐标 [0, 1]."""


@dataclass
class FullConfig:
    """总配置, 顶层入口."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)


# ----------------------------------------------------------------------------
# CLI 解析
# ----------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """解析 train.py 的命令行参数. 解析结果与 FullConfig 字段一一对应."""
    p = argparse.ArgumentParser(description="CAS CAPTCHA 3-head DDP 训练 (accelerate)")
    p.add_argument("--config", type=str, default=None,
                   help="可选 YAML 配置文件路径, 提供各 Config 的初始值")

    # data
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--image-size-h", type=int, default=None)
    p.add_argument("--image-size-w", type=int, default=None)
    p.add_argument("--threshold", type=int, default=None)
    p.add_argument("--binarize-mode", type=str, choices=list(BINARIZE_MODES), default=None)
    p.add_argument("--adaptive-block-size", type=int, default=None)
    p.add_argument("--adaptive-c", type=int, default=None)
    p.add_argument("--train-ratio", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=None)

    # model
    p.add_argument("--backbone", type=str, default=None)
    p.add_argument("--pretrained", type=lambda v: v.lower() in ("1", "true", "yes"), default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--slot-hidden-dim", type=int, default=None)
    p.add_argument("--slot-attention-heads", type=int, default=None)

    # train
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--per-device-batch-size", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--warmup-ratio", type=float, default=None)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--mixed-precision", type=str, default=None)
    p.add_argument("--gradient-accumulation-steps", type=int, default=None)
    p.add_argument("--resume-from", type=str, default=None)

    # loss
    p.add_argument("--label-smoothing", type=float, default=None)
    p.add_argument("--focal-gamma", type=float, default=None)
    p.add_argument("--weight-slot-order", type=float, default=None)
    p.add_argument("--weight-slot-overlap", type=float, default=None)
    p.add_argument("--slot-margin", type=float, default=None)

    return p.parse_args(argv)


def merge_args_to_config(cfg: FullConfig, args: argparse.Namespace) -> FullConfig:
    """把 CLI 中显式传入的字段 (非 None) 覆盖到 cfg 上, 返回新 cfg."""
    cli = {k: v for k, v in vars(args).items() if v is not None and k != "config"}

    # 顶层字段 (data_root / image_size_h / ...)
    for k, v in cli.items():
        for sub in (cfg.data, cfg.model, cfg.train, cfg.loss):
            if hasattr(sub, k):
                setattr(sub, k, v)
                break

    return cfg


def load_from_yaml(path: str) -> FullConfig:
    """从 YAML 加载 (PyYAML). 字段名与 datacass 字段严格一致.

    缺依赖时给清晰报错, 而不是裸 ImportError.
    """
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError(
            "加载 YAML 配置需要 PyYAML, 请先 pip install pyyaml"
        ) from e

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    cfg = FullConfig()
    if "data" in raw:
        for k, v in raw["data"].items():
            setattr(cfg.data, k, v)
    if "model" in raw:
        for k, v in raw["model"].items():
            setattr(cfg.model, k, v)
    if "train" in raw:
        for k, v in raw["train"].items():
            setattr(cfg.train, k, v)
    if "loss" in raw:
        for k, v in raw["loss"].items():
            setattr(cfg.loss, k, v)
    return cfg


def cfg_to_dict(cfg: FullConfig) -> dict:
    """把 FullConfig 序列化为普通 dict (供 accelerate.log_with)."""
    return {
        "data": asdict(cfg.data),
        "model": asdict(cfg.model),
        "train": asdict(cfg.train),
        "loss": asdict(cfg.loss),
    }


def ensure_output_dir(output_dir: str) -> str:
    """rank 0 端调用, 创建目录. 多卡并发时由 accelerate 协调."""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir
