"""模型元信息 CLI."""
from __future__ import annotations

import argparse
import json

from .registry import find_release_checkpoint, inspect_checkpoint, list_model_versions


def main() -> None:
    parser = argparse.ArgumentParser(description="CAS OCR 模型元信息工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-versions", help="列出支持的模型版本")

    checkpoint_parser = subparsers.add_parser("checkpoint-metadata", help="打印 checkpoint 元信息")
    checkpoint_parser.add_argument("--checkpoint", required=True)
    checkpoint_parser.add_argument(
        "--field",
        default=None,
        choices=["version", "family", "display_name", "backbone", "asset_stem", "checkpoint"],
    )

    release_parser = subparsers.add_parser("release-checkpoint", help="解析 release 目录中的 PyTorch checkpoint")
    release_parser.add_argument("--release-root", required=True)

    args = parser.parse_args()
    if args.command == "list-versions":
        print("\n".join(list_model_versions()))
        return
    if args.command == "release-checkpoint":
        try:
            print(find_release_checkpoint(args.release_root))
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc
        return

    metadata = inspect_checkpoint(args.checkpoint)
    if args.field is not None:
        print(metadata[args.field])
        return
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
