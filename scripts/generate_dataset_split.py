#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cas_ocr_model.datasets.split import split_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="生成 dataset manifest.json 划分")
    p.add_argument("--dataset-root", required=True, help="原始数据目录, 例如 ./dataset")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--source-backend", type=str, default=None)
    p.add_argument("--source-url", type=str, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    source: dict[str, str] = {}
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
