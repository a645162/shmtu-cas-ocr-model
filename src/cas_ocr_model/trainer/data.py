"""数据集: 优先读 manifest.json (train/val/test), 兼容旧的顺序切片.

manifest 来自 ``cas_ocr_model.datasets.split``, 字段:
    splits: {train: [...], val: [...], test: [...]}
    stats:  {n_total, n_train, n_val, n_test, seed, ...}

DDP 友好:
    * Dataset 本身不切 rank, 上层用 ``DistributedSampler`` 切分
    * 单进程 (无 DDP) 也能直接用
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import torch
from torch.utils.data import Dataset

from cas_ocr_model.datasets.format import DatasetManifest, MANIFEST_FILENAME, scan_dataset
from cas_ocr_model.common.expression import parse_captcha_expression
from cas_ocr_model.common.preprocess import binarize_captcha, decode_color_image
from .augment import augment_binary_image, sample_binarize_params
from .config import AugmentationConfig, DIGIT2IDX, OP2IDX


Split = Literal["train", "val", "test"]


# ----------------------------------------------------------------------------
# 单样本结构
# ----------------------------------------------------------------------------


@dataclass
class CaptchaSample:
    image_path: Path
    json_path: Path
    digit_left: int   # 0-9
    operator: int     # 0-2  (索引到 OPERATOR_LABELS)
    digit_right: int  # 0-9
    expression: str   # 原始 expression, 调试用


def _scan_samples_from_manifest(data_root: Path, split_files: list[str]) -> list[CaptchaSample]:
    """按 manifest 给出的文件名列表构造样本, 保证三 split 互不重叠且可复现."""
    out: list[CaptchaSample] = []
    for jpg_name in split_files:
        jpg_path = data_root / jpg_name
        json_path = jpg_path.with_suffix(".json")
        if not jpg_path.is_file() or not json_path.is_file():
            continue
        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        expr = str(meta.get("expression", ""))
        parsed = parse_captcha_expression(expr)
        if parsed is None:
            continue
        out.append(
            CaptchaSample(
                image_path=jpg_path,
                json_path=json_path,
                digit_left=DIGIT2IDX[parsed.digit_left],
                operator=OP2IDX[parsed.operator],
                digit_right=DIGIT2IDX[parsed.digit_right],
                expression=expr,
            )
        )
    return out


def _scan_samples_legacy(data_root: Path, train: bool, train_ratio: float) -> list[CaptchaSample]:
    """旧行为兼容: 无 manifest 时按 train_ratio 顺序切片."""
    scan = scan_dataset(data_root)
    if scan.n_paired == 0:
        return []
    files = list(scan.paired_names)
    n_train = max(1, int(len(files) * train_ratio))
    files = files[:n_train] if train else files[n_train:]
    return _scan_samples_from_manifest(data_root, files)


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------


class CaptchaPairDataset(Dataset):
    """验证码 (jpg + json) 数据集. 返回 (image_tensor, label_dict).

    用法 (推荐, 有 manifest):
        train_ds = CaptchaPairDataset(data_root, split="train")
        val_ds   = CaptchaPairDataset(data_root, split="val")
        test_ds  = CaptchaPairDataset(data_root, split="test")

    向后兼容 (无 manifest):
        train_ds = CaptchaPairDataset(data_root, train=True, train_ratio=0.9)
        val_ds   = CaptchaPairDataset(data_root, train=False, train_ratio=0.9)
    """

    def __init__(
        self,
        data_root: str | Path,
        image_size_h: int = 64,
        image_size_w: int = 192,
        threshold: int = 200,
        binarize_mode: str = "min_channel_otsu",
        adaptive_block_size: int = 25,
        adaptive_c: int = 15,
        augmentation: AugmentationConfig | None = None,
        enable_augmentation: bool = False,
        split: Split | None = None,
        train: bool = True,            # legacy
        train_ratio: float = 0.9,      # legacy
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"data_root not found: {self.data_root}")

        # 优先 manifest
        manifest_path = self.data_root / MANIFEST_FILENAME
        if manifest_path.is_file():
            if split is None:
                # 旧 API 兼容: train=True -> "train", train=False -> "val"
                split = "train" if train else "val"
            manifest = DatasetManifest.load(self.data_root)
            split_files = manifest.splits.get(split, [])
            if not split_files:
                raise RuntimeError(
                    f"manifest 中没有 {split} split, 可用: {list(manifest.splits.keys())}"
                )
            self.samples = _scan_samples_from_manifest(self.data_root, split_files)
            self.split_name: str = split
            self.manifest = manifest
        else:
            # 无 manifest, 走旧逻辑
            if split is None:
                split = "train" if train else "val"
            self.samples = _scan_samples_legacy(self.data_root, train, train_ratio)
            self.split_name = split
            self.manifest = None

        if not self.samples:
            raise RuntimeError(
                f"no valid samples in {self.data_root} (split={self.split_name}); "
                "请确认 split 已生成或采集更多样本"
            )

        self.image_size = (image_size_h, image_size_w)
        self.threshold = threshold
        self.binarize_mode = binarize_mode
        self.adaptive_block_size = adaptive_block_size
        self.adaptive_c = adaptive_c
        self.augmentation = augmentation or AugmentationConfig()
        self.enable_augmentation = enable_augmentation and self.augmentation.enabled

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, int]]:
        s = self.samples[idx]
        rng = random.Random(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        img = _load_and_preprocess(
            s.image_path,
            self.image_size,
            self.threshold,
            self.binarize_mode,
            self.adaptive_block_size,
            self.adaptive_c,
            self.augmentation,
            self.enable_augmentation,
            rng,
        )
        labels = {
            "digit_left": s.digit_left,
            "operator": s.operator,
            "digit_right": s.digit_right,
        }
        return img, labels


# ----------------------------------------------------------------------------
# 预处理
# ----------------------------------------------------------------------------


def _load_and_preprocess(
    path: Path,
    image_size: tuple[int, int],
    threshold: int,
    binarize_mode: str,
    adaptive_block_size: int,
    adaptive_c: int,
    augmentation: AugmentationConfig,
    enable_augmentation: bool,
    rng: random.Random,
) -> torch.Tensor:
    img = decode_color_image(path)
    if img is None:
        return torch.zeros(1, image_size[0], image_size[1], dtype=torch.float32)

    mode = binarize_mode
    threshold_value = threshold
    adaptive_c_value = adaptive_c
    if enable_augmentation:
        mode, threshold_value, adaptive_c_value = sample_binarize_params(
            augmentation,
            base_mode=binarize_mode,
            base_threshold=threshold,
            base_adaptive_c=adaptive_c,
            rng=rng,
        )

    binary = binarize_captcha(
        img,
        threshold=threshold_value,
        binarize_mode=mode,
        adaptive_block_size=adaptive_block_size,
        adaptive_c=adaptive_c_value,
    )
    if enable_augmentation:
        binary = augment_binary_image(binary, augmentation, rng)
    resized = cv2.resize(binary, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(resized).float().unsqueeze(0) / 255.0


# ----------------------------------------------------------------------------
# collate_fn
# ----------------------------------------------------------------------------


def collate_triple(
    batch: list[tuple[torch.Tensor, dict[str, int]]],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """DataLoader 用. batch = [(image, {d1, op, d2}), ...]"""
    images = torch.stack([b[0] for b in batch], dim=0)
    labels = {
        "digit_left": torch.tensor([b[1]["digit_left"] for b in batch], dtype=torch.long),
        "operator": torch.tensor([b[1]["operator"] for b in batch], dtype=torch.long),
        "digit_right": torch.tensor([b[1]["digit_right"] for b in batch], dtype=torch.long),
    }
    return images, labels
