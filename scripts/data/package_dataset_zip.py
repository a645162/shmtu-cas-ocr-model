#!/usr/bin/env python3
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="打包 dataset 为 zip, 压缩包根目录固定为 datasets/")
    p.add_argument("--dataset-root", required=True, help="原始数据目录, 例如 ./dataset")
    p.add_argument("--output-zip", required=True, help="输出 zip 路径")
    p.add_argument("--zip-root-name", default="datasets", help="压缩包内根目录名")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output_zip = Path(args.output_zip).resolve()
    zip_root_name = args.zip_root_name.strip("/") or "datasets"

    if not dataset_root.is_dir():
        raise SystemExit(f"dataset root not found: {dataset_root}")

    output_zip.parent.mkdir(parents=True, exist_ok=True)

    file_count = 0
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(dataset_root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(dataset_root)
            arcname = Path(zip_root_name) / rel
            zf.write(path, arcname.as_posix())
            file_count += 1

    print(f"[zip] dataset_root={dataset_root}")
    print(f"[zip] output={output_zip}")
    print(f"[zip] zip_root={zip_root_name}")
    print(f"[zip] files={file_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
