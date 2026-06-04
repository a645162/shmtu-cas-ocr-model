"""采集器 CLI 入口.

调用:
    python -m cas_ocr_model.datasets.maker --backend restful \\
        --ocr-url http://127.0.0.1:21600 \\
        --output ./dataset --count 5000 --processes 4 --per-process 8
"""
from __future__ import annotations

from .config import build_arg_parser
from .pool import spawn_workers


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.processes < 1:
        raise SystemExit("--processes 必须 >= 1")
    if args.per_process < 1:
        raise SystemExit("--per-process 必须 >= 1")
    if args.count < 1:
        raise SystemExit("--count 必须 >= 1")
    spawn_workers(args)


if __name__ == "__main__":
    main()
