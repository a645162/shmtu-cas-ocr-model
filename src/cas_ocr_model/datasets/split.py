"""数据集物理分割: train / val / test 三段.

调用:
    python -m cas_ocr_model.datasets.split \\
        --dataset-root ./dataset \\
        --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 \\
        --seed 42

行为:
    1) 扫描 dataset_root, 配对 jpg + json
    2) 按 seed 随机洗牌, 按 (train_ratio, val_ratio, test_ratio) 顺序切片
    3) 写 manifest.json (含三段文件名列表 + 来源 + 统计)
    4) 不复制/移动图片, 仅在 manifest 里列文件名; DataLoader 按需加载

注: 选 manifest 索引而非物理目录的原因:
    * DDP 下多卡不需要重复图片数据 (避免 NFS / 共享存储冗余)
    * 灵活调整比例, 重新跑一次 split 即可
    * manifest 内置来源 (backend, ocr_url) 便于追溯
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

from .format import DatasetManifest, scan_dataset


# ----------------------------------------------------------------------------
# 核心分割
# ----------------------------------------------------------------------------


def split_dataset(
    dataset_root: str | Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    source: dict | None = None,
) -> DatasetManifest:
    """扫描 dataset_root, 写 manifest.json.

    Returns:
        写好的 manifest 对象.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"train/val/test 比例之和必须为 1, got {train_ratio}+{val_ratio}+{test_ratio}"
        )

    dataset_root = Path(dataset_root)
    scan = scan_dataset(dataset_root)
    if scan.n_paired == 0:
        raise RuntimeError(
            f"dataset_root={dataset_root} 下没有任何 (jpg+json) 配对, "
            "请先运行 datasets/dataset_collector.py 采集数据"
        )

    # 随机洗牌
    rng = random.Random(seed)
    files = list(scan.paired_names)
    rng.shuffle(files)

    # 顺序切片
    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    splits = {
        "train": files[:n_train],
        "val": files[n_train : n_train + n_val],
        "test": files[n_train + n_val :],
    }

    manifest = DatasetManifest(
        version=1,
        created_at=int(time.time()),
        splits=splits,
        stats={
            "n_total": n,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "seed": seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
        },
        source=source or {},
    )

    out_path = manifest.save(dataset_root)
    print(f"[split] wrote manifest -> {out_path}")
    print(
        f"[split] n_total={n} n_train={n_train} n_val={n_val} n_test={n_test} "
        f"missing_json={len(scan.missing_json)} missing_jpg={len(scan.missing_jpg)}"
    )
    if scan.missing_json:
        print(f"[split][WARN] {len(scan.missing_json)} 个 jpg 缺 json, 已跳过")
    if scan.missing_jpg:
        print(f"[split][WARN] {len(scan.missing_jpg)} 个 json 缺 jpg, 已跳过")
    return manifest


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="数据集 train/val/test 分割 (写 manifest.json)")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--source-backend", type=str, default=None,
                   help="可选: 写入 manifest.source.backend (如 restful/tcp/pytorch)")
    p.add_argument("--source-url", type=str, default=None)
    args = p.parse_args()

    source: dict = {}
    if args.source_backend:
        source["backend"] = args.source_backend
    if args.source_url:
        source["ocr_url"] = args.source_url

    split_dataset(
        dataset_root=args.dataset_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        source=source,
    )


if __name__ == "__main__":
    main()
