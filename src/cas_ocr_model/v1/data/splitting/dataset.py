"""按目录类别把数据随机划分为 train/val/test。"""

from __future__ import annotations

import os
import random
import shutil
from collections.abc import Sequence


def split_dataset(
    dataset_dir: str,
    save_dir_base: str,
    splits: Sequence[str] | None = None,
    split_ratio: Sequence[float] | None = None,
) -> None:
    """
    按类别子目录把数据复制到 {save_dir_base}/{split_name}/{class_name}。

    Args:
        dataset_dir: 源数据根目录，每个子目录是一类。
        save_dir_base: 划分后输出根目录。
        splits: 划分名列表，默认 ['train', 'val', 'test']。
        split_ratio: 与 splits 等长的比例列表。
    """
    if split_ratio is None and splits is None:
        splits = ["train", "val", "test"]
        split_ratio = [0.8, 0.1, 0.1]
    if splits is None or split_ratio is None:
        splits = ["train", "val", "test"]
        split_ratio = [0.8, 0.1, 0.1]

    for split_name in splits:
        os.makedirs(os.path.join(save_dir_base, split_name), exist_ok=True)

    classes = os.listdir(dataset_dir)
    for class_name in classes:
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        files = os.listdir(class_dir)
        random.shuffle(files)

        num_files = len(files)
        split_counts = [int(num_files * ratio) for ratio in split_ratio]
        split_counts[-1] += num_files - sum(split_counts)

        start_index = 0
        for split_name, count in zip(splits, split_counts, strict=False):
            split_dir = os.path.join(save_dir_base, split_name, class_name)
            os.makedirs(split_dir, exist_ok=True)
            end_index = start_index + count
            for file_name in files[start_index:end_index]:
                src = os.path.join(class_dir, file_name)
                dst = os.path.join(split_dir, file_name)
                shutil.copy(src, dst)
            start_index = end_index


if __name__ == "__main__":
    split_dataset(
        dataset_dir="../../workdir/Classify/EqualSymbol",
        save_dir_base="../../workdir/Datasets/EqualSymbol",
        splits=["train", "val"],
        split_ratio=[0.9, 0.1],
    )
    split_dataset(
        dataset_dir="../../workdir/Classify/Operator",
        save_dir_base="../../workdir/Datasets/Operator",
        splits=["train", "val"],
        split_ratio=[0.9, 0.1],
    )
    split_dataset(
        dataset_dir="../../workdir/Classify/Digit",
        save_dir_base="../../workdir/Datasets/Digit",
        splits=["train", "val"],
        split_ratio=[0.9, 0.1],
    )
