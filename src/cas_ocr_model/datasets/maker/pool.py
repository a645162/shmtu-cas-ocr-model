"""多进程 Pool 调度 + 进度监控.

主进程用 tqdm 进度条 (stderr) 显示进度, 子进程的 print (flush=True)
走 stdout 不会冲突 (终端是行式设备, 两路流并行不互相覆盖).

断点续采 (默认行为):
    启动时扫描 --output 已有 8 位 jpg, 从 max+1 继续, 达到 --count 自动停止.
    --resume 是显式开关, 行为完全一致, 仅在日志里多打 "explicit resume" 标记,
    便于审计 (例如在 cron / shell 脚本里明确表达"这是续采,不是覆盖").
"""
from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Manager, get_context
from pathlib import Path
from typing import Any

from .config import scan_existing_max_index
from .worker import worker_main


def spawn_workers(args: argparse.Namespace) -> None:
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 断点续采 (默认): 已有 N 张 -> 下一个可用编号 = N+1.
    # --resume 不改变行为, 仅在日志里多打 "explicit resume" 标记.
    existing_max = scan_existing_max_index(output_dir)
    start_idx = existing_max + 1

    # spawn 避免 fork torch/onnxruntime 子进程死锁
    ctx = get_context("spawn")
    mgr = ctx.Manager()
    counter = mgr.Value("i", start_idx)
    progress_q: Any = mgr.Queue()

    mode_tag = "explicit-resume" if args.resume else "default-resume"
    print(
        f"[main] mode={mode_tag} start_index={start_idx} "
        f"(existing={existing_max + 1} files scanned) output={output_dir} "
        f"backend={args.backend} processes={args.processes} "
        f"per_process={args.per_process} target={args.count}",
        flush=True,
    )

    # tqdm 走 stderr, 跟子进程 stdout 的 saved/error 日志互不干扰
    try:
        from tqdm import tqdm
        bar = tqdm(
            total=args.count,
            initial=counter.value,
            desc="collect",
            unit="img",
            ncols=80,
            file=sys.stderr,
            dynamic_ncols=True,
            mininterval=0.5,
            disable=not sys.stderr.isatty(),  # 非 tty (重定向到 log) 时关闭
        )
    except ImportError:
        # 没有 tqdm 时的兜底: 仍走原来的 [main] progress 报告
        bar = None
        last_report = time.time()
        last_saved = counter.value

    crashed_count = 0

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
                        crashed_count += 1
                if crashed:
                    break
                # 排空 progress 队列 (子进程向主进程报告, 当前未用, 排空防内存涨)
                while not progress_q.empty():
                    try:
                        progress_q.get_nowait()
                    except Exception:
                        break
                now = time.time()
                if bar is not None:
                    # tqdm: 增量更新到当前计数
                    delta = counter.value - bar.n
                    if delta > 0:
                        bar.update(delta)
                    bar.set_postfix_str(f"target={args.count}", refresh=False)
                else:
                    # 兜底: 定时打印
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
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("[main] KeyboardInterrupt -> terminating pool", flush=True)
            pool.terminate()
        finally:
            if bar is not None:
                # 收尾: 把进度对齐到真实计数, 关闭
                delta = counter.value - bar.n
                if delta > 0:
                    bar.update(delta)
                bar.close()
            pool.close()
            pool.join()

    suffix = ""
    if crashed_count:
        suffix = f" crashed_workers={crashed_count}"
    print(
        f"[main] done. saved={counter.value} target={args.count} "
        f"output={output_dir}{suffix}",
        flush=True,
    )
