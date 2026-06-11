"""性能 benchmark (单进程通用): 延迟 / 吞吐量 / 内存 / 多 batch 扫描.

适用场景: 单机单卡或单机 CPU 的快速速度测量.
多卡 DDP 场景请用 multi_gpu_benchmark.py (精度) / single_gpu_benchmark.py (单卡速度).
"""
from __future__ import annotations

import platform
import time
import tracemalloc
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from cas_ocr_model.common.console import print_benchmark_table

from .inference import CaptchaInferencer


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
    def from_samples(samples_ms: list[float]) -> LatencyStats:
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
class BenchmarkReport:
    backend_name: str
    device: str
    image_size: tuple[int, int]
    n_samples: int
    warmup: int
    python_version: str = platform.python_version()
    torch_version: str = torch.__version__
    single_batch_size: int = 1
    single_latency: LatencyStats = field(default_factory=LatencyStats)
    single_throughput_qps: float = 0.0
    batch_scan: dict[int, LatencyStats] = field(default_factory=dict)
    batch_scan_throughput: dict[int, float] = field(default_factory=dict)
    peak_memory_mb: float = 0.0


def _build_synthetic_batch(inferencer: CaptchaInferencer, batch_size: int) -> torch.Tensor:
    h, w = inferencer.config.image_size_h, inferencer.config.image_size_w
    return torch.zeros(batch_size, 1, h, w, dtype=torch.float32)


def benchmark(
    inferencer: CaptchaInferencer,
    n_samples: int = 500,
    warmup: int = 20,
    batch_sizes: tuple[int, ...] = (1, 8, 32, 128),
    backend_name: str = "unknown",
) -> BenchmarkReport:
    h, w = inferencer.config.image_size_h, inferencer.config.image_size_w
    device_str = str(getattr(inferencer.backend, "device", "unknown")) if hasattr(inferencer.backend, "device") else "cpu"

    report = BenchmarkReport(
        backend_name=backend_name,
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

    report.single_batch_size = 1
    report.single_latency = LatencyStats.from_samples(samples)
    report.single_throughput_qps = n_samples / elapsed if elapsed > 0 else 0.0

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
        report.batch_scan[bs] = stats
        report.batch_scan_throughput[bs] = (n_iter * bs) / elapsed if elapsed > 0 else 0.0
        report.peak_memory_mb = max(report.peak_memory_mb, peak / (1024 * 1024))

    return report


def report_to_dict(r: BenchmarkReport) -> dict:
    return {
        "backend": r.backend_name,
        "device": r.device,
        "image_size": list(r.image_size),
        "n_samples": r.n_samples,
        "warmup": r.warmup,
        "python_version": r.python_version,
        "torch_version": r.torch_version,
        "peak_memory_mb": r.peak_memory_mb,
        "single": {
            "batch_size": r.single_batch_size,
            "latency_ms": asdict(r.single_latency),
            "throughput_qps": r.single_throughput_qps,
        },
        "batch_scan": {
            str(bs): {
                "latency_ms": asdict(stats),
                "throughput_qps": r.batch_scan_throughput[bs],
            }
            for bs, stats in r.batch_scan.items()
        },
    }


def print_report(r: BenchmarkReport) -> None:
    print_benchmark_table(
        title="Benchmark",
        backend=r.backend_name,
        device=r.device,
        image_size=r.image_size,
        python_version=r.python_version,
        torch_version=r.torch_version,
        single_stats=asdict(r.single_latency),
        single_qps=r.single_throughput_qps,
        batch_scan={bs: asdict(stats) for bs, stats in r.batch_scan.items()},
        batch_throughput=r.batch_scan_throughput,
        peak_memory_mb=r.peak_memory_mb,
    )
