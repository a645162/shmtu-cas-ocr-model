"""多进程 Pool 调度 + 进度监控.

主进程用 rich.progress 进度条 (stderr) 显示进度, 子进程的 print (flush=True)
走 stdout 不会冲突 (终端是行式设备, 两路流并行不互相覆盖).

断点续采 (默认行为):
    启动时扫描 --output 已有 8 位 jpg, 从 max+1 继续, 达到 --count 自动停止.
    --resume 是显式开关, 行为完全一致, 仅在日志里多打 "explicit resume" 标记,
    便于审计 (例如在 cron / shell 脚本里明确表达"这是续采,不是覆盖").

进度条组件 (rich):
    SpinnerColumn | 任务描述 | BarColumn | 计数 (m/n) | 百分比 | ETA (TimeRemainingColumn) | 自定义 fields (new/rate/resume)
    ETA 自动用 s / m / h / d 进位 (rich 内置 TimeRemainingColumn), 不再依赖 format_eta.
    没有 rich 时退化到 [main] progress 定时打印兜底日志.
"""
from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from .config import format_eta, scan_existing_max_index
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
    # 进度条语义: 断点续采时, 编号从 start_idx 增长到 start_idx + count
    final_total = start_idx + args.count
    print(
        f"[main] mode={mode_tag} start_index={start_idx} "
        f"(existing={existing_max + 1} files scanned) output={output_dir} "
        f"backend={args.backend} processes={args.processes} "
        f"per_process={args.per_process} target={args.count} "
        f"final_total={final_total}",
        flush=True,
    )

    # rich 进度条 (pip 风格): Spinner + Bar + 计数 + 百分比 + ETA (s/m/h/d) + 自定义字段
    # rich 没装时退回到 [main] progress 定时打印
    progress_ctx: Any = None
    task_id: Any = None
    try:
        from rich.console import Console
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TaskProgressColumn,
            TextColumn,
            TimeRemainingColumn,
        )

        # 非 tty 时禁用 rich (重定向到 log 文件会污染输出)
        console = Console(file=sys.stderr, force_terminal=sys.stderr.isatty(), soft_wrap=True)
        progress_ctx = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]collect[/bold cyan]"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),  # "N / M"
            TaskProgressColumn(),  # " 45%"
            TextColumn("eta"),
            TimeRemainingColumn(),  # rich 自带, 自动 s/m/h/d 进位
            TextColumn("{task.fields[info]}"),
            console=console,
            disable=not sys.stderr.isatty(),
            transient=False,
            redirect_stdout=False,
            redirect_stderr=False,
        )
    except ImportError:
        progress_ctx = None
        last_report = time.time()
        last_saved = counter.value

    # 启动时刻 (用 monotonic 不受系统时钟跳变影响)
    boot_time = time.monotonic()
    crashed_count = 0

    if progress_ctx is not None:
        progress_ctx.__enter__()
        task_id = progress_ctx.add_task(
            "collect",
            total=final_total,
            completed=start_idx,
            info="",
        )

    try:
        with ctx.Pool(processes=args.processes) as pool:
            async_results = [
                pool.apply_async(
                    worker_main, (i, args, counter, str(output_dir), progress_q),
                )
                for i in range(args.processes)
            ]

            try:
                while True:
                    if counter.value >= final_total:
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
                    # 排空 progress 队列 (子进程向主进程回传的日志/统计)
                    # 真正消费: 用 rich.log() 渲染 (自动绕开进度条不重绘区)
                    while not progress_q.empty():
                        try:
                            msg = progress_q.get_nowait()
                        except Exception:
                            break
                        if progress_ctx is not None and isinstance(msg, dict) and "level" in msg:
                            # 子进程日志: {"level": "info"|"error", "msg": "...", "ts": ...}
                            level = msg.get("level", "info")
                            text = msg.get("msg", "")
                            style = "red" if level == "error" else "cyan" if level == "warn" else "white"
                            progress_ctx.log(text, style=style)
                        # 旧格式 dict (status/saved/rejected/errors) 静默丢弃
                    now = time.time()
                    # ETA / 速率: 本次新采 / 已用时间; 至少 2 张样本 + 1s 才计算
                    new_this_run = counter.value - start_idx
                    elapsed = max(1e-6, time.monotonic() - boot_time)
                    rate = new_this_run / elapsed if new_this_run >= 2 and elapsed >= 1.0 else 0.0
                    eta_str_value = format_eta(
                        (max(0, args.count - new_this_run) / rate)
                        if rate > 0
                        else -1.0
                    )
                    info_str = (
                        f"new={new_this_run}/{args.count} "
                        f"rate={rate:.1f}/s "
                        f"manual_eta={eta_str_value} "
                        f"resume={'on' if args.resume else 'off'}"
                    )

                    if progress_ctx is not None:
                        progress_ctx.update(
                            task_id,
                            completed=counter.value,
                            info=info_str,
                            refresh=True,
                        )
                    else:
                        # 兜底: 定时打印
                        if now - last_report >= args.report_interval:
                            current = counter.value
                            period_rate = (current - last_saved) / max(1e-6, (now - last_report))
                            print(
                                f"[main] progress={current}/{final_total} "
                                f"(this_run={new_this_run}/{args.count}, "
                                f"{100*new_this_run/args.count:.1f}%) "
                                f"rate={period_rate:.1f}/s eta={eta_str_value}",
                                flush=True,
                            )
                            last_report = now
                            last_saved = current
                    time.sleep(0.2)
            except KeyboardInterrupt:
                print("[main] KeyboardInterrupt -> terminating pool", flush=True)
                pool.terminate()
            finally:
                pool.close()
                pool.join()
    finally:
        if progress_ctx is not None:
            # 收尾: 把进度对齐到真实计数, 关闭
            if task_id is not None:
                progress_ctx.update(task_id, completed=counter.value, refresh=True)
            progress_ctx.__exit__(None, None, None)

    elapsed_total = time.monotonic() - boot_time
    suffix = ""
    if crashed_count:
        suffix = f" crashed_workers={crashed_count}"
    new_total = counter.value - start_idx
    avg_rate = new_total / max(1e-6, elapsed_total)
    print(
        f"[main] done. written={counter.value}/{final_total} "
        f"(this_run={new_total}/{args.count}) "
        f"elapsed={elapsed_total:.1f}s avg_rate={avg_rate:.1f}/s "
        f"mode={mode_tag} output={output_dir}{suffix}",
        flush=True,
    )
