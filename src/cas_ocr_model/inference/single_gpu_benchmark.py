"""单卡推理速度 benchmark.

测量单卡 PyTorch 上的延迟 / 吞吐量, 用于评估部署性能.

调用:
    python -m cas_ocr_model.inference.single_gpu_benchmark \\
        --checkpoint runs/exp1/best.pt \\
        --device cuda \\
        --num-samples 1000 --batch-sizes 1,8,32,128 \\
        --output bench.json

注: 多卡 DDP 精度请用 multi_gpu_benchmark.py.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from cas_ocr_model.common.console import print_benchmark_table, tag_print

from .inference import CaptchaInferencer, InferencerConfig


@dataclass
class LatencyStats:
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p99_ms: float = 0.0
    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0

    @staticmethod
    def from_samples(samples_ms: list[float]) -> "LatencyStats":
        if not samples_ms:
            return LatencyStats()
        arr = np.asarray(samples_ms, dtype=np.float64)
        return LatencyStats(
            p50_ms=float(np.percentile(arr, 50)),
            p90_ms=float(np.percentile(arr, 90)),
            p99_ms=float(np.percentile(arr, 99)),
            mean_ms=float(arr.mean()),
            std_ms=float(arr.std()),
            min_ms=float(arr.min()),
            max_ms=float(arr.max()),
        )


@dataclass
class SingleGpuReport:
    backend: str
    device: str
    image_size: tuple[int, int]
    n_samples: int
    warmup: int
    python_version: str = platform.python_version()
    torch_version: str = torch.__version__
    single: dict = field(default_factory=dict)
    batch_scan: dict = field(default_factory=dict)
    peak_memory_mb: float = 0.0


def build_backend(args: argparse.Namespace):
    from .backends.pytorch_backend import PyTorchBackend
    return PyTorchBackend(
        checkpoint=args.checkpoint,
        backbone=args.backbone,
        device=args.device,
    )


def _build_synthetic_batch(inferencer: CaptchaInferencer, batch_size: int) -> torch.Tensor:
    h, w = inferencer.config.image_size_h, inferencer.config.image_size_w
    return torch.zeros(batch_size, 1, h, w, dtype=torch.float32)


def run_benchmark(
    inferencer: CaptchaInferencer,
    n_samples: int = 500,
    warmup: int = 20,
    batch_sizes: tuple[int, ...] = (1, 8, 32, 128),
    backend_name: str = "unknown",
    device_str: str = "cpu",
) -> SingleGpuReport:
    h, w = inferencer.config.image_size_h, inferencer.config.image_size_w
    report = SingleGpuReport(
        backend=backend_name,
        device=device_str,
        image_size=(h, w),
        n_samples=n_samples,
        warmup=warmup,
    )

    one = _build_synthetic_batch(inferencer, 1)
    for _ in range(warmup):
        _ = inferencer.backend.infer(one)
    samples: list[float] = []
    tracemalloc.start()
    t_start = time.perf_counter()
    for _ in range(n_samples):
        t0 = time.perf_counter()
        _ = inferencer.backend.infer(one)
        samples.append((time.perf_counter() - t0) * 1000.0)
    elapsed = time.perf_counter() - t_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    report.peak_memory_mb = max(report.peak_memory_mb, peak / (1024 * 1024))

    stats = LatencyStats.from_samples(samples)
    report.single = {
        "batch_size": 1,
        "latency_ms": asdict(stats),
        "throughput_qps": n_samples / elapsed if elapsed > 0 else 0.0,
    }

    for bs in batch_sizes:
        batch = _build_synthetic_batch(inferencer, bs)
        n_iter = max(1, n_samples // bs)
        for _ in range(min(warmup, 5)):
            _ = inferencer.backend.infer(batch)
        samples: list[float] = []
        tracemalloc.start()
        t_start = time.perf_counter()
        for _ in range(n_iter):
            t0 = time.perf_counter()
            _ = inferencer.backend.infer(batch)
            samples.append((time.perf_counter() - t0) * 1000.0)
        elapsed = time.perf_counter() - t_start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        stats = LatencyStats.from_samples(samples)
        qps = (n_iter * bs) / elapsed if elapsed > 0 else 0.0
        report.batch_scan[str(bs)] = {
            "latency_ms": asdict(stats),
            "throughput_qps": qps,
        }
        report.peak_memory_mb = max(report.peak_memory_mb, peak / (1024 * 1024))

    return report


def print_report(r: SingleGpuReport) -> None:
    print_benchmark_table(
        title="Single-GPU Benchmark",
        backend=r.backend,
        device=r.device,
        image_size=r.image_size,
        python_version=r.python_version,
        torch_version=r.torch_version,
        single_stats=r.single["latency_ms"],
        single_qps=r.single["throughput_qps"],
        batch_scan={int(k): v["latency_ms"] for k, v in r.batch_scan.items()},
        batch_throughput={int(k): v["throughput_qps"] for k, v in r.batch_scan.items()},
        peak_memory_mb=r.peak_memory_mb,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="单卡推理速度 benchmark")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   choices=["cpu", "cuda"])
    p.add_argument("--num-samples", type=int, default=500)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--batch-sizes", default="1,8,32,128")
    p.add_argument("--image-size-h", type=int, default=64)
    p.add_argument("--image-size-w", type=int, default=192)
    p.add_argument("--threshold", type=int, default=200)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    backend = build_backend(args)
    cfg = InferencerConfig(
        image_size_h=args.image_size_h,
        image_size_w=args.image_size_w,
        threshold=args.threshold,
        batch_size=128,
    )
    inferencer = CaptchaInferencer(backend=backend, config=cfg)

    bs_list = tuple(int(x) for x in args.batch_sizes.split(",")) if args.batch_sizes else (1, 8, 32, 128)
    rpt = run_benchmark(
        inferencer,
        n_samples=args.num_samples,
        warmup=args.warmup,
        batch_sizes=bs_list,
        backend_name="pytorch",
        device_str=args.device,
    )
    print_report(rpt)

    if args.output:
        report_dict = {
            "backend": rpt.backend,
            "device": rpt.device,
            "image_size": list(rpt.image_size),
            "n_samples": rpt.n_samples,
            "warmup": rpt.warmup,
            "python_version": rpt.python_version,
            "torch_version": rpt.torch_version,
            "peak_memory_mb": rpt.peak_memory_mb,
            "single": rpt.single,
            "batch_scan": rpt.batch_scan,
        }
        Path(args.output).write_text(
            json.dumps(report_dict, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tag_print("saved", args.output)


if __name__ == "__main__":
    main()
