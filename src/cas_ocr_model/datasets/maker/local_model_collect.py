"""8 卡本地模型采集入口: 每卡 1 进程, 本地推理后提交 CAS 验证."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
import traceback
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from shmtu_cas import EpayAuth

from .cas_client import verify_one
from .config import format_eta, scan_existing_max_index
from .ocr_backends import ModelBackend, build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="8 卡本地模型验证码采集: 每卡 1 进程, 推理后提交 CAS 验证"
    )
    p.add_argument("--checkpoint", required=True, help="triple-head checkpoint (best.pt)")
    p.add_argument("--output", default="./dataset", help="输出目录 (含 .jpg + .json)")
    p.add_argument("--count", type=int, default=1000, help="目标成功保存数量 (在已有基础上累加)")
    p.add_argument("--gpu-ids", type=str, required=True, help="逗号分隔 GPU 编号, 例如 0,1,2,3,4,5,6,7")
    p.add_argument("--weights-dir", default="./weights", help="保留兼容参数, 传给模型构造器")
    p.add_argument("--throttle", type=float, default=0.0, help="每次验证后 sleep 秒数")
    p.add_argument("--report-interval", type=float, default=5.0, help="主进程兜底日志间隔")
    p.add_argument(
        "--resume",
        action="store_true",
        help="显式声明续采模式 (默认行为也是续采, 仅用于日志审计)",
    )
    return p.parse_args()


def parse_gpu_ids(raw: str) -> list[str]:
    gpu_ids = [item.strip() for item in raw.split(",") if item.strip()]
    if not gpu_ids:
        raise SystemExit("--gpu-ids 不能为空")
    return gpu_ids


def save_verified_sample(
    backend_name: str,
    verification,
    counter: Any,
    counter_lock: Any,
    final_total: int,
    output_dir: Path,
) -> bool:
    """验证成功后分配编号并原子落盘."""
    with counter_lock:
        if counter.value >= final_total:
            return False
        idx = counter.value
        counter.value = idx + 1

    stem = f"{idx:08d}"
    jpg_path = output_dir / f"{stem}.jpg"
    json_path = output_dir / f"{stem}.json"

    with tempfile.NamedTemporaryFile(
        delete=False, dir=output_dir, prefix=f".{stem}.", suffix=".jpg.tmp"
    ) as tmp_img:
        tmp_img.write(verification.image_bytes)
        tmp_img_path = Path(tmp_img.name)
    tmp_img_path.replace(jpg_path)

    payload = {
        "id": idx,
        "filename": jpg_path.name,
        "expression": verification.hit.expression,
        "answer": verification.hit.answer,
        "verification": "password_error_or_success",
        "probe_username": verification.probe_username,
        "backend": backend_name,
        "created_at": int(time.time()),
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        delete=False,
        dir=output_dir,
        prefix=f".{stem}.",
        suffix=".json.tmp",
        encoding="utf-8",
    ) as tmp_json:
        json.dump(payload, tmp_json, ensure_ascii=False, indent=2)
        tmp_json_path = Path(tmp_json.name)
    tmp_json_path.replace(json_path)
    return True


def worker_main(
    worker_id: int,
    gpu_id: str,
    args: argparse.Namespace,
    counter: Any,
    counter_lock: Any,
    final_total: int,
    output_dir_str: str,
    progress_q: Any,
) -> None:
    """单 GPU worker 入口."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    os.environ["SHMTU_COLLECT_GPU_ID"] = gpu_id
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _log(level: str, msg: str) -> None:
        prefix = f"[pid={os.getpid()} w={worker_id} gpu={gpu_id}] {msg}"
        try:
            progress_q.put_nowait({"level": level, "msg": prefix, "ts": time.time()})
        except Exception:  # noqa: BLE001
            print(prefix, flush=True)

    try:
        backend = ModelBackend(
            build_model(
                "pytorch_v2",
                weights_dir=Path(args.weights_dir),
                checkpoint=Path(args.checkpoint).resolve(),
            )
        )
        _log("info", "backend ready -> pytorch_v2")
        backend.warmup()

        saved = rejected = errors = attempts = 0

        async def loop() -> None:
            nonlocal saved, rejected, errors, attempts
            auth = EpayAuth()
            try:
                while True:
                    if counter.value >= final_total:
                        return
                    verification = await verify_one(
                        backend,
                        auth,
                        log_prefix=f"[gpu={gpu_id}]",
                        log_q=progress_q,
                    )
                    attempts += 1
                    if verification.status == "error":
                        errors += 1
                    elif verification.status != "correct" or verification.hit is None or verification.image_bytes is None:
                        rejected += 1
                    else:
                        wrote = save_verified_sample(
                            backend.name,
                            verification,
                            counter,
                            counter_lock,
                            final_total,
                            output_dir,
                        )
                        if wrote:
                            saved += 1
                            _log(
                                "info",
                                f"saved expr='{verification.hit.expression}' "
                                f"answer={verification.hit.answer} verify={verification.variant}",
                            )
                    if args.throttle > 0:
                        await asyncio.sleep(args.throttle)
                    if attempts % 200 == 0:
                        await auth.aclose()
                        auth = EpayAuth()
            finally:
                await auth.aclose()

        asyncio.run(loop())
        _log(
            "info",
            f"done saved={saved} rejected={rejected} errors={errors} attempts={attempts}",
        )
    except KeyboardInterrupt:
        _log("warn", "KeyboardInterrupt")
    except Exception as e:  # noqa: BLE001
        _log("error", f"fatal: {e}\n{traceback.format_exc()}")
        raise


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gpu_ids = parse_gpu_ids(args.gpu_ids)

    existing_max = scan_existing_max_index(output_dir)
    start_idx = existing_max + 1
    final_total = start_idx + args.count
    mode_tag = "explicit-resume" if args.resume else "default-resume"

    print(
        f"[local-collect] mode={mode_tag} output={output_dir} checkpoint={Path(args.checkpoint).resolve()} "
        f"gpu_ids={gpu_ids} workers={len(gpu_ids)} target={args.count} "
        f"start_index={start_idx} final_total={final_total}",
        flush=True,
    )

    ctx = get_context("spawn")
    mgr = ctx.Manager()
    counter = mgr.Value("i", start_idx)
    counter_lock = mgr.Lock()
    progress_q: Any = mgr.Queue()

    progress_ctx: Any = None
    task_id: Any = None
    last_report = time.time()
    last_saved = counter.value
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

        console = Console(file=sys.stderr, force_terminal=sys.stderr.isatty(), soft_wrap=True)
        progress_ctx = Progress(
            SpinnerColumn(),
            TextColumn("[bold magenta]local-collect[/bold magenta]"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TextColumn("eta"),
            TimeRemainingColumn(),
            TextColumn("{task.fields[info]}"),
            console=console,
            disable=not sys.stderr.isatty(),
            transient=False,
            redirect_stdout=False,
            redirect_stderr=False,
        )
    except ImportError:
        progress_ctx = None

    boot_time = time.monotonic()
    if progress_ctx is not None:
        progress_ctx.__enter__()
        task_id = progress_ctx.add_task(
            "local-collect",
            total=final_total,
            completed=start_idx,
            info="",
        )

    workers = [
        ctx.Process(
            target=worker_main,
            args=(i, gpu_id, args, counter, counter_lock, final_total, str(output_dir), progress_q),
            name=f"collect-gpu{gpu_id}",
        )
        for i, gpu_id in enumerate(gpu_ids)
    ]

    for proc in workers:
        proc.start()

    try:
        while True:
            if counter.value >= final_total:
                break

            crashed = []
            for proc in workers:
                if not proc.is_alive() and proc.exitcode not in (None, 0):
                    crashed.append((proc.name, proc.exitcode))
            if crashed:
                raise RuntimeError(f"worker crashed: {crashed}")

            while not progress_q.empty():
                try:
                    msg = progress_q.get_nowait()
                except Exception:
                    break
                if progress_ctx is not None and isinstance(msg, dict) and "level" in msg:
                    level = msg.get("level", "info")
                    text = msg.get("msg", "")
                    style = "red" if level == "error" else "cyan" if level == "warn" else "white"
                    progress_ctx.log(text, style=style)
                elif isinstance(msg, dict):
                    print(msg.get("msg", ""), flush=True)

            now = time.time()
            new_this_run = counter.value - start_idx
            elapsed = max(1e-6, time.monotonic() - boot_time)
            rate = new_this_run / elapsed if new_this_run >= 2 and elapsed >= 1.0 else 0.0
            eta_str_value = format_eta(
                (max(0, args.count - new_this_run) / rate) if rate > 0 else -1.0
            )
            info_str = (
                f"new={new_this_run}/{args.count} "
                f"rate={rate:.1f}/s "
                f"manual_eta={eta_str_value} "
                f"gpus={','.join(gpu_ids)}"
            )
            if progress_ctx is not None:
                progress_ctx.update(
                    task_id,
                    completed=counter.value,
                    info=info_str,
                    refresh=True,
                )
            elif now - last_report >= args.report_interval:
                current = counter.value
                period_rate = (current - last_saved) / max(1e-6, (now - last_report))
                print(
                    f"[local-collect] progress={current}/{final_total} "
                    f"(this_run={new_this_run}/{args.count}, {100 * new_this_run / args.count:.1f}%) "
                    f"rate={period_rate:.1f}/s eta={eta_str_value}",
                    flush=True,
                )
                last_report = now
                last_saved = current
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("[local-collect] KeyboardInterrupt -> terminating workers", flush=True)
        for proc in workers:
            proc.terminate()
    except Exception:
        for proc in workers:
            if proc.is_alive():
                proc.terminate()
        raise
    finally:
        for proc in workers:
            proc.join()
        if progress_ctx is not None:
            if task_id is not None:
                progress_ctx.update(task_id, completed=counter.value, refresh=True)
            progress_ctx.__exit__(None, None, None)

    elapsed_total = time.monotonic() - boot_time
    new_total = counter.value - start_idx
    avg_rate = new_total / max(1e-6, elapsed_total)
    print(
        f"[local-collect] done written={counter.value}/{final_total} "
        f"(this_run={new_total}/{args.count}) elapsed={elapsed_total:.1f}s "
        f"avg_rate={avg_rate:.1f}/s output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
