from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return int(value)


@dataclass(slots=True)
class ApiServerConfig:
    http_host: str = "0.0.0.0"
    http_port: int = 21600
    tcp_host: str = "0.0.0.0"
    tcp_port: int = 21601
    model_kind: str = "v2"
    checkpoint: Path | None = None
    v1_model_dir: Path | None = None
    device: str = "auto"
    worker_count: int = 0
    queue_capacity: int = 0
    server_name: str = ""
    image_size_h: int = 64
    image_size_w: int = 192
    threshold: int = 200
    binarize_mode: str = "min_channel_otsu"
    adaptive_block_size: int = 25
    adaptive_c: int = 15
    batch_size: int = 32
    max_input_bytes: int = 4 * 1024 * 1024

    def resolved_worker_count(self) -> int:
        if self.worker_count > 0:
            return self.worker_count
        if self.device == "cuda":
            return 1
        cpu_count = os.cpu_count() or 1
        return max(1, min(cpu_count, 4))

    def resolved_queue_capacity(self) -> int:
        if self.queue_capacity > 0:
            return self.queue_capacity
        return max(8, self.resolved_worker_count() * 16)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SHMTU CAS OCR PyTorch API server")
    parser.add_argument("--http-host", default=_env_str("SHMTU_HTTP_HOST", "0.0.0.0"))
    parser.add_argument("--http-port", type=int, default=_env_int("SHMTU_HTTP_PORT", 21600))
    parser.add_argument("--tcp-host", default=_env_str("SHMTU_TCP_HOST", "0.0.0.0"))
    parser.add_argument("--tcp-port", type=int, default=_env_int("SHMTU_TCP_PORT", 21601))
    parser.add_argument(
        "--model-kind",
        choices=["v1", "v2"],
        default=_env_str("SHMTU_MODEL_KIND", "v2"),
        help="v1 = 三模型旧版, v2 = TriSlot Decoder 新版",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("SHMTU_CHECKPOINT"),
        help="新版 TriSlot Decoder checkpoint, 通常是 best.pt",
    )
    parser.add_argument(
        "--v1-model-dir",
        default=os.environ.get("SHMTU_V1_MODEL_DIR"),
        help="v1 的 .pth 权重目录",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=_env_str("SHMTU_DEVICE", "auto"),
    )
    parser.add_argument("--workers", type=int, default=_env_int("SHMTU_WORKERS", 0))
    parser.add_argument(
        "--queue-capacity",
        type=int,
        default=_env_int("SHMTU_QUEUE_CAPACITY", 0),
    )
    parser.add_argument("--server-name", default=_env_str("SHMTU_SERVER_NAME", ""))
    parser.add_argument("--image-size-h", type=int, default=_env_int("SHMTU_IMAGE_SIZE_H", 64))
    parser.add_argument("--image-size-w", type=int, default=_env_int("SHMTU_IMAGE_SIZE_W", 192))
    parser.add_argument("--threshold", type=int, default=_env_int("SHMTU_THRESHOLD", 200))
    parser.add_argument(
        "--binarize-mode",
        default=_env_str("SHMTU_BINARIZE_MODE", "min_channel_otsu"),
    )
    parser.add_argument(
        "--adaptive-block-size",
        type=int,
        default=_env_int("SHMTU_ADAPTIVE_BLOCK_SIZE", 25),
    )
    parser.add_argument("--adaptive-c", type=int, default=_env_int("SHMTU_ADAPTIVE_C", 15))
    parser.add_argument("--batch-size", type=int, default=_env_int("SHMTU_BATCH_SIZE", 32))
    parser.add_argument(
        "--max-input-bytes",
        type=int,
        default=_env_int("SHMTU_MAX_INPUT_BYTES", 4 * 1024 * 1024),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> ApiServerConfig:
    args = build_parser().parse_args(argv)
    return ApiServerConfig(
        http_host=args.http_host,
        http_port=args.http_port,
        tcp_host=args.tcp_host,
        tcp_port=args.tcp_port,
        model_kind=args.model_kind,
        checkpoint=Path(args.checkpoint).resolve() if args.checkpoint else None,
        v1_model_dir=Path(args.v1_model_dir).resolve() if args.v1_model_dir else None,
        device=args.device,
        worker_count=args.workers,
        queue_capacity=args.queue_capacity,
        server_name=args.server_name,
        image_size_h=args.image_size_h,
        image_size_w=args.image_size_w,
        threshold=args.threshold,
        binarize_mode=args.binarize_mode,
        adaptive_block_size=args.adaptive_block_size,
        adaptive_c=args.adaptive_c,
        batch_size=args.batch_size,
        max_input_bytes=args.max_input_bytes,
    )
