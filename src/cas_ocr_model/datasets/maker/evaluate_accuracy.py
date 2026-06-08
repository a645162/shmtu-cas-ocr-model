"""基于真实 CAS 登录反馈统计 OCR 正确率."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from shmtu_cas import EpayAuth

from .cas_client import verify_one
from .config import build_eval_arg_parser
from .ocr_backends import build_backend


@dataclass(slots=True)
class AccuracyReport:
    backend: str
    model_name: str
    total: int
    correct: int
    incorrect: int
    failure: int
    error: int
    known_total: int
    accuracy: float
    elapsed_seconds: float


async def run_evaluation(args) -> AccuracyReport:
    backend = build_backend(args)
    backend.warmup()
    sem = asyncio.Semaphore(args.concurrency)
    counters = {"correct": 0, "incorrect": 0, "failure": 0, "error": 0}
    start = time.monotonic()

    async def one(index: int) -> None:
        auth = EpayAuth()
        try:
            async with sem:
                result = await verify_one(backend, auth, log_prefix=f"[eval#{index}]")
            counters[result.status] += 1
            hit_expr = result.hit.expression if result.hit else ""
            print(
                f"[eval] #{index:04d} status={result.status} variant={result.variant or '-'} expr={hit_expr}",
                flush=True,
            )
            if args.throttle > 0:
                await asyncio.sleep(args.throttle)
        finally:
            await auth.aclose()

    await asyncio.gather(*(one(i) for i in range(args.count)))
    elapsed = time.monotonic() - start
    known_total = counters["correct"] + counters["incorrect"]
    accuracy = (counters["correct"] / known_total) if known_total else 0.0
    return AccuracyReport(
        backend=args.backend,
        model_name=getattr(args, "model_name", ""),
        total=args.count,
        correct=counters["correct"],
        incorrect=counters["incorrect"],
        failure=counters["failure"],
        error=counters["error"],
        known_total=known_total,
        accuracy=accuracy,
        elapsed_seconds=elapsed,
    )


def main() -> None:
    args = build_eval_arg_parser().parse_args()
    if args.count < 1:
        raise SystemExit("--count 必须 >= 1")
    if args.concurrency < 1:
        raise SystemExit("--concurrency 必须 >= 1")

    report = asyncio.run(run_evaluation(args))
    payload = asdict(report)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
