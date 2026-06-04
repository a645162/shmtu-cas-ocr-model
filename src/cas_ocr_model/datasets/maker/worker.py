"""单 worker 入口: 在信号量限制下循环采集, 计数由 manager.Value 共享."""
from __future__ import annotations

import argparse
import asyncio
import os
import time
import traceback
from pathlib import Path
from typing import Any

from shmtu_cas import EpayAuth

from .cas_client import collect_one
from .ocr_backends import build_backend


def worker_main(
    worker_id: int,
    args: argparse.Namespace,
    counter: Any,
    output_dir_str: str,
    progress_q: Any,
) -> dict[str, int]:
    """spawn 进程入口. 返回 {"saved","rejected","errors"}."""
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = build_backend(args)
    backend.warmup()
    sem = asyncio.Semaphore(args.per_process)

    saved = rejected = errors = 0
    log_prefix = f"[pid={os.getpid()} w={worker_id} b={backend.name}]"

    async def loop() -> None:
        nonlocal saved, rejected, errors
        auth = EpayAuth()
        try:
            while True:
                if counter.value >= args.count:
                    return
                async with sem:
                    status = await collect_one(backend, auth, counter, output_dir, log_prefix)
                if status == "saved":
                    saved += 1
                elif status == "rejected":
                    rejected += 1
                else:
                    errors += 1
                try:
                    progress_q.put_nowait(
                        {
                            "ts": time.time(),
                            "worker": worker_id,
                            "status": status,
                            "saved": saved,
                            "rejected": rejected,
                            "errors": errors,
                            "counter": counter.value,
                        }
                    )
                except Exception:  # noqa: BLE001
                    pass
                if args.throttle > 0:
                    await asyncio.sleep(args.throttle)
                if (saved + rejected + errors) % 200 == 0:
                    await auth.aclose()
                    auth = EpayAuth()
        finally:
            await auth.aclose()

    try:
        asyncio.run(loop())
    except KeyboardInterrupt:
        pass
    except Exception as e:  # noqa: BLE001
        print(f"{log_prefix} fatal: {e}\n{traceback.format_exc()}", flush=True)
    return {"saved": saved, "rejected": rejected, "errors": errors}
