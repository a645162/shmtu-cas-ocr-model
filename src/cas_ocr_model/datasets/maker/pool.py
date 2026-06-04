"""多进程 Pool 调度 + 进度监控."""
from __future__ import annotations

import argparse
import time
from multiprocessing import Manager, get_context
from pathlib import Path
from typing import Any

from .worker import worker_main


def _next_start_index(output_dir: Path) -> int:
    """扫描已有 8 位编号 jpg, 返回下一个可用序号 (断点续采)."""
    max_idx = -1
    for p in output_dir.glob("[0-9]" * 8 + ".jpg"):
        try:
            v = int(p.stem)
            if v > max_idx:
                max_idx = v
        except ValueError:
            pass
    return max_idx + 1


def spawn_workers(args: argparse.Namespace) -> None:
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # spawn 避免 fork torch/onnxruntime 子进程死锁
    ctx = get_context("spawn")
    mgr = ctx.Manager()
    counter = mgr.Value("i", _next_start_index(output_dir))
    progress_q: Any = mgr.Queue()

    print(
        f"[main] start from index={counter.value}, output={output_dir}, "
        f"backend={args.backend}, processes={args.processes}, "
        f"per_process={args.per_process}, target={args.count}",
        flush=True,
    )

    last_report = time.time()
    last_saved = 0

    with ctx.Pool(processes=args.processes) as pool:
        async_results = [
            pool.apply_async(
                worker_main, (i, args, counter, str(output_dir), progress_q),
            )
            for i in range(args.processes)
        ]

        try:
            while True:
                if counter.value >= args.count:
                    break
                crashed = False
                for i, ar in enumerate(async_results):
                    if ar.ready() and not ar.successful():
                        try:
                            ar.get()
                        except Exception as e:  # noqa: BLE001
                            print(f"[main] worker {i} crashed: {e}", flush=True)
                        crashed = True
                if crashed:
                    break
                while not progress_q.empty():
                    try:
                        progress_q.get_nowait()
                    except Exception:
                        break
                now = time.time()
                if now - last_report >= args.report_interval:
                    current = counter.value
                    rate = (current - last_saved) / max(1e-6, (now - last_report))
                    print(
                        f"[main] progress={current}/{args.count} "
                        f"({100*current/args.count:.1f}%) rate={rate:.1f}/s",
                        flush=True,
                    )
                    last_report = now
                    last_saved = current
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("[main] KeyboardInterrupt -> terminating pool", flush=True)
            pool.terminate()
        finally:
            pool.close()
            pool.join()

    print(f"[main] done. saved={counter.value} target={args.count} output={output_dir}")
