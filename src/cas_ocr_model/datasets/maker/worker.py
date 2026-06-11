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
    """spawn 进程入口. 返回 {"saved","rejected","errors"}.

    所有 print 都改为 put 到 progress_q (与 collect_one 共享), 由主进程统一渲染,
    避免与 rich 进度条抢行. progress_q 为 None 时降级到 print.
    """
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = build_backend(args)
    backend.warmup()
    sem = asyncio.Semaphore(args.per_process)

    def _log(level: str, msg: str) -> None:
        prefix = f"[pid={os.getpid()} w={worker_id} b={backend.name}] {msg}"
        if progress_q is None:
            print(prefix, flush=True)
            return
        import contextlib

        with contextlib.suppress(Exception):
            progress_q.put_nowait({"level": level, "msg": prefix, "ts": time.time()})

    saved = rejected = errors = 0

    async def loop() -> None:
        nonlocal saved, rejected, errors
        auth = EpayAuth()
        try:
            while True:
                if counter.value >= args.count:
                    return
                async with sem:
                    status = await collect_one(backend, auth, counter, output_dir, "", progress_q)
                if status == "saved":
                    saved += 1
                elif status == "rejected":
                    rejected += 1
                else:
                    errors += 1
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
        _log("error", f"fatal: {e}\n{traceback.format_exc()}")
    return {"saved": saved, "rejected": rejected, "errors": errors}
