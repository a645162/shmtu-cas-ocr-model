"""训练配置 — dataclass 形式, 兼容 YAML / TOML 反序列化.

支持两种用法:
    1) 命令行参数 (train.py 入口) 直接覆盖
    2) 加载 YAML/TOML 配置文件 (configs/*.{yaml,toml}) 后用 CLI 覆盖
"""
from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Optional

from cas_ocr_model.common.expression import CANONICAL_OPERATOR_LABELS
from cas_ocr_model.common.preprocess import BINARIZE_MODES

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
class AugmentationConfig:
    """训练集增强配置. 仅作用于训练 split."""

    enabled: bool = False
    """总开关."""

    translate_enabled: bool = True
    translate_prob: float = 0.7
    translate_x_px: int = 6
    translate_y_px: int = 3
    """随机平移概率与最大像素位移."""

    affine_enabled: bool = True
    affine_prob: float = 0.4
    rotate_deg: float = 2.5
    shear_deg: float = 4.0
    scale_min: float = 0.97
    scale_max: float = 1.03
    """轻微仿射增强参数."""

    morphology_enabled: bool = True
    morphology_prob: float = 0.15
    morphology_kernel_size: int = 3
    """二值图膨胀/腐蚀扰动."""

    noise_enabled: bool = True
    noise_prob: float = 0.10
    noise_density: float = 0.001
    """稀疏椒盐噪点密度."""

    binarize_jitter_enabled: bool = False
    binarize_jitter_prob: float = 0.0
    threshold_jitter: int = 0
    adaptive_c_jitter: int = 0
    alt_binarize_modes: list[str] = field(default_factory=list)
    """训练期可选二值化参数扰动."""

    rethreshold_after_aug: bool = True
    """增强后重新压回 0/255 二值图."""


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

    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    """训练集专用增强配置."""


@dataclass
class ModelConfig:
    """模型结构配置."""

    version: str = "2.0"
    """模型版本. 当前默认 2.0."""

    backbone: str = "resnet18"
    """backbone 名称, 支持内置别名或 ``timm/<model_name>``."""

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

    epochs: int = 500
    """总 epoch 数."""

    early_stop_patience: int = -1
    """验证集连续多少个 epoch 不提升后提前停止. 0=关闭, -1=总 epoch 的 20%."""

    per_device_batch_size: int = 256
    """单卡 batch size. 8 卡 x 256 = 2048 effective."""

    learning_rate: float = 1e-3
    """初始学习率 (AdamW). 8 卡 DDP 一般按 base lr * world_size 线性缩放."""

    weight_decay: float = 1e-4
    warmup_ratio: float = 0.05
    """线性 warmup 占总步数比例."""

    grad_clip: float = 1.0
    """梯度裁剪阈值 (L2 norm)."""

    nonfinite_backprop_step_patience: int = 10
    """反向传播出现非有限值时, 连续多少个 step 后停止训练. 0=关闭."""

    nonfinite_backprop_epoch_patience: int = 10
    """反向传播出现非有限值时, 连续多少个 epoch 后停止训练. 0=关闭."""

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

    report_to: str = "auto"
    """实验跟踪后端: auto / none / wandb / 逗号分隔列表 / all."""

    tracker_project_name: str = "cas-ocr-train"
    """accelerate tracker project 名称."""

    wandb_run_name: Optional[str] = None
    """可选 wandb run name. 不填时默认按 output_dir 自动生成."""

    wandb_entity: Optional[str] = None
    """可选 wandb entity / team."""

    wandb_tags: list[str] = field(default_factory=list)
    """可选 wandb tags."""

    use_rich_progress: bool = True
    """主进程使用 rich 进度条; 非 TTY 环境自动回退文本日志."""


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

    enable_slot_right_boundary: bool = False
    """是否启用第三个槽位的右边界约束, 防止盯到 '=' 和结果区域."""

    weight_slot_right_boundary: float = 0.02
    """第三个槽位右边界约束权重."""

    slot_right_boundary_max: float = 0.68
    """第三个槽位中心允许的最右边界 (归一化坐标)."""

    enable_slot_attention_variance: bool = False
    """是否启用槽位注意力方差上界约束, 防止 attention 过散."""

    weight_slot_attention_variance: float = 0.01
    """槽位注意力方差约束权重."""

    slot_attention_max_variance: float = 0.035
    """单个槽位 attention 宽度方差的上界阈值."""

    enable_operator_class_balance: bool = False
    """是否对 operator head 启用类别权重."""

    operator_class_weights: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    """operator head 的类别权重, 顺序与 OPERATOR_LABELS 一致."""


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
                   help="可选 YAML/TOML 配置文件路径, 提供各 Config 的初始值")

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
    p.add_argument("--aug-enabled", type=lambda v: v.lower() in ("1", "true", "yes"), default=None)
    p.add_argument("--aug-translate-enabled", type=lambda v: v.lower() in ("1", "true", "yes"), default=None)
    p.add_argument("--aug-translate-prob", type=float, default=None)
    p.add_argument("--aug-translate-x-px", type=int, default=None)
    p.add_argument("--aug-translate-y-px", type=int, default=None)
    p.add_argument("--aug-affine-enabled", type=lambda v: v.lower() in ("1", "true", "yes"), default=None)
    p.add_argument("--aug-affine-prob", type=float, default=None)
    p.add_argument("--aug-rotate-deg", type=float, default=None)
    p.add_argument("--aug-shear-deg", type=float, default=None)
    p.add_argument("--aug-scale-min", type=float, default=None)
    p.add_argument("--aug-scale-max", type=float, default=None)
    p.add_argument("--aug-morphology-enabled", type=lambda v: v.lower() in ("1", "true", "yes"), default=None)
    p.add_argument("--aug-morphology-prob", type=float, default=None)
    p.add_argument("--aug-morphology-kernel-size", type=int, default=None)
    p.add_argument("--aug-noise-enabled", type=lambda v: v.lower() in ("1", "true", "yes"), default=None)
    p.add_argument("--aug-noise-prob", type=float, default=None)
    p.add_argument("--aug-noise-density", type=float, default=None)
    p.add_argument("--aug-binarize-jitter-enabled", type=lambda v: v.lower() in ("1", "true", "yes"), default=None)
    p.add_argument("--aug-binarize-jitter-prob", type=float, default=None)
    p.add_argument("--aug-threshold-jitter", type=int, default=None)
    p.add_argument("--aug-adaptive-c-jitter", type=int, default=None)
    p.add_argument(
        "--aug-alt-binarize-modes",
        type=lambda v: [s.strip() for s in v.split(",") if s.strip()],
        default=None,
    )
    p.add_argument("--aug-rethreshold-after-aug", type=lambda v: v.lower() in ("1", "true", "yes"), default=None)

    # model
    p.add_argument("--model-version", type=str, default=None)
    p.add_argument("--backbone", type=str, default=None)
    p.add_argument("--pretrained", type=lambda v: v.lower() in ("1", "true", "yes"), default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--slot-hidden-dim", type=int, default=None)
    p.add_argument("--slot-attention-heads", type=int, default=None)

    # train
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--early-stop-patience", type=int, default=None)
    p.add_argument("--per-device-batch-size", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--warmup-ratio", type=float, default=None)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--nonfinite-backprop-step-patience", type=int, default=None)
    p.add_argument("--nonfinite-backprop-epoch-patience", type=int, default=None)
    p.add_argument("--mixed-precision", type=str, default=None)
    p.add_argument("--gradient-accumulation-steps", type=int, default=None)
    p.add_argument("--resume-from", type=str, default=None)
    p.add_argument("--report-to", type=str, default=None)
    p.add_argument("--tracker-project-name", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument(
        "--wandb-tags",
        type=lambda v: [s.strip() for s in v.split(",") if s.strip()],
        default=None,
    )
    p.add_argument("--use-rich-progress", type=lambda v: v.lower() in ("1", "true", "yes"), default=None)

    # loss
    p.add_argument("--label-smoothing", type=float, default=None)
    p.add_argument("--focal-gamma", type=float, default=None)
    p.add_argument("--weight-slot-order", type=float, default=None)
    p.add_argument("--weight-slot-overlap", type=float, default=None)
    p.add_argument("--slot-margin", type=float, default=None)
    p.add_argument("--enable-slot-right-boundary", type=lambda v: v.lower() in ("1", "true", "yes"), default=None)
    p.add_argument("--weight-slot-right-boundary", type=float, default=None)
    p.add_argument("--slot-right-boundary-max", type=float, default=None)
    p.add_argument("--enable-slot-attention-variance", type=lambda v: v.lower() in ("1", "true", "yes"), default=None)
    p.add_argument("--weight-slot-attention-variance", type=float, default=None)
    p.add_argument("--slot-attention-max-variance", type=float, default=None)
    p.add_argument("--enable-operator-class-balance", type=lambda v: v.lower() in ("1", "true", "yes"), default=None)
    p.add_argument(
        "--operator-class-weights",
        type=lambda v: [float(s.strip()) for s in v.split(",") if s.strip()],
        default=None,
    )

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

    nested_cli = {
        "model_version": ("model", "version"),
        "aug_enabled": ("data", "augmentation", "enabled"),
        "aug_translate_enabled": ("data", "augmentation", "translate_enabled"),
        "aug_translate_prob": ("data", "augmentation", "translate_prob"),
        "aug_translate_x_px": ("data", "augmentation", "translate_x_px"),
        "aug_translate_y_px": ("data", "augmentation", "translate_y_px"),
        "aug_affine_enabled": ("data", "augmentation", "affine_enabled"),
        "aug_affine_prob": ("data", "augmentation", "affine_prob"),
        "aug_rotate_deg": ("data", "augmentation", "rotate_deg"),
        "aug_shear_deg": ("data", "augmentation", "shear_deg"),
        "aug_scale_min": ("data", "augmentation", "scale_min"),
        "aug_scale_max": ("data", "augmentation", "scale_max"),
        "aug_morphology_enabled": ("data", "augmentation", "morphology_enabled"),
        "aug_morphology_prob": ("data", "augmentation", "morphology_prob"),
        "aug_morphology_kernel_size": ("data", "augmentation", "morphology_kernel_size"),
        "aug_noise_enabled": ("data", "augmentation", "noise_enabled"),
        "aug_noise_prob": ("data", "augmentation", "noise_prob"),
        "aug_noise_density": ("data", "augmentation", "noise_density"),
        "aug_binarize_jitter_enabled": ("data", "augmentation", "binarize_jitter_enabled"),
        "aug_binarize_jitter_prob": ("data", "augmentation", "binarize_jitter_prob"),
        "aug_threshold_jitter": ("data", "augmentation", "threshold_jitter"),
        "aug_adaptive_c_jitter": ("data", "augmentation", "adaptive_c_jitter"),
        "aug_alt_binarize_modes": ("data", "augmentation", "alt_binarize_modes"),
        "aug_rethreshold_after_aug": ("data", "augmentation", "rethreshold_after_aug"),
    }
    for arg_name, path in nested_cli.items():
        if arg_name not in cli:
            continue
        obj = cfg
        for attr in path[:-1]:
            obj = getattr(obj, attr)
        setattr(obj, path[-1], cli[arg_name])

    return cfg


def _merge_dataclass(target: object, raw: dict) -> None:
    if not isinstance(raw, dict):
        return
    field_names = {f.name for f in fields(target)}
    for key, value in raw.items():
        if key not in field_names:
            continue
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge_dataclass(current, value)
            continue
        setattr(target, key, value)


def _merge_raw_config(cfg: FullConfig, raw: dict) -> FullConfig:
    if "data" in raw:
        _merge_dataclass(cfg.data, raw["data"])
    if "model" in raw:
        _merge_dataclass(cfg.model, raw["model"])
    if "train" in raw:
        _merge_dataclass(cfg.train, raw["train"])
    if "loss" in raw:
        _merge_dataclass(cfg.loss, raw["loss"])
    return cfg


def load_from_yaml(path: str) -> FullConfig:
    """从 YAML 加载. 字段名与 dataclass 字段严格一致."""
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
    return _merge_raw_config(cfg, raw)


def load_from_toml(path: str) -> FullConfig:
    """从 TOML 加载. 字段名与 dataclass 字段严格一致."""
    try:
        import tomllib  # py311+
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # py39/py310 fallback
        except ImportError as e:
            raise RuntimeError(
                "加载 TOML 配置需要 Python 3.11+ 或安装 tomli: pip install tomli"
            ) from e

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"config file not found: {p}")
    with p.open("rb") as fh:
        raw = tomllib.load(fh) or {}

    cfg = FullConfig()
    return _merge_raw_config(cfg, raw)


def load_config(path: str) -> FullConfig:
    """按文件后缀自动加载 YAML / TOML 配置."""
    suffix = Path(path).suffix.lower()
    if suffix in (".yaml", ".yml"):
        return load_from_yaml(path)
    if suffix == ".toml":
        return load_from_toml(path)
    raise ValueError(f"unsupported config format: {path} (expected .yaml/.yml/.toml)")


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
