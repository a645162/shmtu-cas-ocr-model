"""上海海事大学 CAS 验证码图片数据集采集器.

通过 shmtu-cas-python (Lib/shmtu-cas-python) 调用 CAS 登录流程获取验证码图片,
使用三种可选推理后端 (RESTful HTTP / TCP / 本地 PyTorch) 识别算式,
再用随机学号/密码提交一次登录探测,根据服务器返回判定 OCR 是否正确:
    * PasswordError   -> 验证码正确,落盘图片 + json 标签
    * ValidateCodeError -> 验证码错误,丢弃
    * Success / Failure / 网络异常 -> Success 也视为正确 (对齐 tauri/captcha.rs)

并发模型:
    * 顶层: multiprocessing.Pool (多进程,每进程独立 httpx session + OCR 后端实例)
    * 每进程内: asyncio.Semaphore 限制并发请求数 (避免对 CAS/OCR 造成过载)

图片编号从 00000000 起严格单调递增, 跨进程安全: 由 manager.Value 共享原子计数器.
落盘使用 tempfile 原子重命名, 避免半截文件.

用法示例 (在 Model/shmtu-cas-ocr-model 根目录执行):
    # 1) 远端 RESTful OCR (默认 HTTP 端口 21600)
    python -m cas_ocr_model.dataset_collector \\
        --backend restful --ocr-url http://127.0.0.1:21600 \\
        --output ./dataset --count 5000 --processes 4 --per-process 8

    # 2) 远端 TCP OCR (默认 TCP 端口 21601)
    python -m cas_ocr_model.dataset_collector \\
        --backend tcp --ocr-host 127.0.0.1 --ocr-port 21601 \\
        --output ./dataset --count 5000 --processes 4 --per-process 8

    # 3) 本地 PyTorch 推理 (自动从 GitHub Release 下载权重到 ./weights)
    python -m cas_ocr_model.dataset_collector \\
        --backend pytorch --weights-dir ./weights \\
        --output ./dataset --count 5000 --processes 2 --per-process 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import string
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Literal

# 把仓库根的 Lib/shmtu-cas-python 加入 sys.path,
# 这样无论从仓库根或 Model 子目录运行都能 import shmtu_cas.
_HERE = Path(__file__).resolve().parent
_MODEL_PKG_ROOT = _HERE.parent.parent.parent  # .../Model/shmtu-cas-ocr-model
_REPO_ROOT = _MODEL_PKG_ROOT.parent.parent  # .../shmtu-terminal
for cand in (
    str(_REPO_ROOT / "Lib" / "shmtu-cas-python" / "src"),
    str(_MODEL_PKG_ROOT / "src"),
):
    if cand not in sys.path and Path(cand).is_dir():
        sys.path.insert(0, cand)

from shmtu_cas import EpayAuth  # noqa: E402
from shmtu_cas.captcha.http_ocr import CaptchaOcrHttp  # noqa: E402
from shmtu_cas.captcha.tcp_ocr import CaptchaOcr  # noqa: E402
from shmtu_cas.captcha.utils import get_expr_result  # noqa: E402

# ----------------------------------------------------------------------------
# 常量 (与 Rust CLI / cas.rs 保持一致)
# ----------------------------------------------------------------------------

PROBE_USERNAME_PREFIX = "ds_cap_"
PROBE_PASSWORD_LEN = 12
PROBE_PASSWORD_CHARS = string.ascii_letters + string.digits

# GitHub Release 直链 (公开 release, 无需认证)
GITHUB_RELEASE_TAG = "v1.0"
GITHUB_RELEASE_BASE = (
    f"https://github.com/a645162/shmtu-cas-ocr-model/releases/download/{GITHUB_RELEASE_TAG}"
)
WEIGHT_FILES = {
    "equal_symbol": f"{GITHUB_RELEASE_BASE}/resnet18_equal_symbol_latest.pth",
    "operator": f"{GITHUB_RELEASE_BASE}/resnet18_operator_latest.pth",
    "digit": f"{GITHUB_RELEASE_BASE}/resnet34_digit_latest.pth",
}

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


# ----------------------------------------------------------------------------
# 通用工具
# ----------------------------------------------------------------------------


def random_probe_account() -> tuple[str, str]:
    """生成随机学号/密码, 与 Rust 端 CAPTCHA_TEST_PROBE_* 等效但每次不同."""
    student_no = PROBE_USERNAME_PREFIX + "".join(
        random.choices(string.digits, k=max(0, 10 - len(PROBE_USERNAME_PREFIX)))
    )
    password = "".join(random.choices(PROBE_PASSWORD_CHARS, k=PROBE_PASSWORD_LEN))
    return student_no, password


def download_file(url: str, dest: Path) -> None:
    """流式下载, 进度提示; 已存在且大小 > 0 则跳过."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return

    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    print(f"[weights] downloading {url} -> {dest}", flush=True)
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as fh:
        total = int(resp.headers.get("Content-Length", "0") or 0)
        read = 0
        chunk = 64 * 1024
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            fh.write(buf)
            read += len(buf)
            if total:
                pct = read * 100 // total
                if read % (chunk * 16) == 0 or read == total:
                    print(f"  ... {read}/{total} bytes ({pct}%)", flush=True)
    tmp.replace(dest)


def ensure_pytorch_weights(weights_dir: Path) -> dict[str, Path]:
    """下载并校验 PyTorch 权重. 返回 {label: path} 字典."""
    weights_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for label, url in WEIGHT_FILES.items():
        dest = weights_dir / Path(url).name
        download_file(url, dest)
        paths[label] = dest
    return paths


# ----------------------------------------------------------------------------
# 推理后端抽象
# ----------------------------------------------------------------------------


@dataclass
class OcrHit:
    expression: str  # 例如 "12+34=46" 或 "12+34="
    answer: str  # get_expr_result(expression), 用于回填到 submit_login


class OcrBackend:
    """推理后端基类. ``recognize`` 协程: 输入图片字节, 输出 OcrHit."""

    name: str = "base"

    def warmup(self) -> None:
        """首次推理前可执行一次性初始化 (如模型加载). 可同步, 可异步."""

    async def recognize(self, image_bytes: bytes) -> OcrHit:
        raise NotImplementedError


class RestfulBackend(OcrBackend):
    name = "restful"

    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        self._client = CaptchaOcrHttp(base_url=base_url, timeout=timeout)

    def warmup(self) -> None:
        # CaptchaOcrHttp.health_check_async 本身是 async, 在没有运行中循环时跑一下.
        # 这里处于 worker 进程入口 (尚未进 asyncio.run), 可以直接 asyncio.run.
        try:
            asyncio.run(self._client.health_check_async())
        except Exception as e:  # noqa: BLE001
            print(f"[ocr:restful] health check failed (ignored): {e}", flush=True)

    async def recognize(self, image_bytes: bytes) -> OcrHit:
        expr = await self._client.ocr_auto_retry_async(image_bytes, max_retries=2)
        return OcrHit(expression=expr, answer=get_expr_result(expr))


class TcpBackend(OcrBackend):
    name = "tcp"

    def __init__(self, host: str, port: int) -> None:
        self._client = CaptchaOcr(host=host, port=port)

    async def recognize(self, image_bytes: bytes) -> OcrHit:
        # CaptchaOcr 是同步 socket 客户端, 丢到默认 executor 避免阻塞事件循环.
        expr = await asyncio.to_thread(
            self._client.ocr_auto_retry, image_bytes, 2
        )
        return OcrHit(expression=expr, answer=get_expr_result(expr))


class PytorchBackend(OcrBackend):
    name = "pytorch"

    def __init__(self, weights_dir: Path) -> None:
        self._weights = ensure_pytorch_weights(weights_dir)
        # 把权重文件镜像到 cas_ocr_model/v1/workdir/Models, 让 pth_save_dir_path 解析到
        from cas_ocr_model.v1.configs.paths import pth_save_dir_path  # type: ignore
        target = Path(pth_save_dir_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.mkdir(parents=True, exist_ok=True)
        for label, src in self._weights.items():
            dst = target / src.name
            if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                dst.write_bytes(src.read_bytes())

        import cv2  # type: ignore
        import numpy as np  # type: ignore
        from cas_ocr_model.v1.data_modules.device import get_recommended_device  # type: ignore
        from cas_ocr_model.v1.inference.predictor import (  # type: ignore
            load_models,
            predict_validate_code,
        )

        self._device = get_recommended_device()
        self._models = load_models(self._device, pth_dir=str(target))
        self._predict = predict_validate_code
        self._cv2 = cv2
        self._np = np

    def warmup(self) -> None:
        blank = self._np.full((30, 100, 3), 255, dtype=self._np.uint8)
        try:
            self._predict(blank, self._device, *self._models, print_result=False)
            print(f"[ocr:pytorch] warmup ok on {self._device}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[ocr:pytorch] warmup failed (ignored): {e}", flush=True)

    async def recognize(self, image_bytes: bytes) -> OcrHit:
        # CPU 密集, 走默认线程池.
        return await asyncio.to_thread(self._recognize_sync, image_bytes)

    def _recognize_sync(self, image_bytes: bytes) -> OcrHit:
        arr = self._np.frombuffer(image_bytes, dtype=self._np.uint8)
        img = self._cv2.imdecode(arr, self._cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("captcha image decode failed")
        calc_result, calc_str, *_ = self._predict(
            img, self._device, *self._models, print_result=False
        )
        normalized = calc_str.replace(" ", "")
        return OcrHit(expression=normalized, answer=str(calc_result))


def build_backend(args: argparse.Namespace) -> OcrBackend:
    if args.backend == "restful":
        return RestfulBackend(args.ocr_url, timeout=args.ocr_timeout)
    if args.backend == "tcp":
        return TcpBackend(args.ocr_host, args.ocr_port)
    if args.backend == "pytorch":
        return PytorchBackend(Path(args.weights_dir))
    raise ValueError(f"unknown backend: {args.backend}")


# ----------------------------------------------------------------------------
# 单次采集
# ----------------------------------------------------------------------------


async def collect_one(
    backend: OcrBackend,
    auth: EpayAuth,
    counter: Any,
    output_dir: Path,
    log_prefix: str,
) -> Literal["saved", "rejected", "error"]:
    """单次采集闭环: probe -> challenge -> ocr -> submit -> 落盘/丢弃."""
    try:
        probe = await auth.probe_login()
    except Exception as e:  # noqa: BLE001
        print(f"{log_prefix} probe failed: {e}", flush=True)
        return "error"
    if not probe.is_need_login:
        await auth.aclose()
        auth = EpayAuth()

    try:
        challenge = await auth.prepare_challenge()
    except Exception as e:  # noqa: BLE001
        print(f"{log_prefix} prepare_challenge failed: {e}", flush=True)
        return "error"

    try:
        hit = await backend.recognize(challenge.captcha_image)
    except Exception as e:  # noqa: BLE001
        print(f"{log_prefix} ocr failed: {e}", flush=True)
        return "error"

    if not hit.answer:
        return "rejected"

    student_no, password = random_probe_account()
    try:
        result = await auth.submit_login(
            student_no, password, hit.answer, challenge.execution
        )
    except Exception as e:  # noqa: BLE001
        print(f"{log_prefix} submit failed: {e}", flush=True)
        return "error"

    # 与 tauri/captcha.rs 同样的判定: PasswordError / Success 都视为验证码正确
    if not (result.is_password_error or result.is_success):
        return "rejected"

    # 验证码正确 -> 落盘 (原子写)
    # multiprocessing.Manager.Value 的 ValueProxy 内部已加锁 (RLock),
    # ``counter.value`` 的读/写在 RLock 下原子, 多 worker 安全.
    idx = counter.value
    counter.value = idx + 1
    stem = f"{idx:08d}"
    jpg_path = output_dir / f"{stem}.jpg"
    json_path = output_dir / f"{stem}.json"

    with tempfile.NamedTemporaryFile(
        delete=False, dir=output_dir, prefix=f".{stem}.", suffix=".jpg.tmp"
    ) as tmp_img:
        tmp_img.write(challenge.captcha_image)
        tmp_img_path = Path(tmp_img.name)
    tmp_img_path.replace(jpg_path)

    payload = {
        "id": idx,
        "filename": jpg_path.name,
        "expression": hit.expression,
        "answer": hit.answer,
        "verification": "password_error_or_success",
        "probe_username": student_no,
        "backend": backend.name,
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

    print(
        f"{log_prefix} saved #{stem}: expr='{hit.expression}' answer={hit.answer} "
        f"verify={result.variant}",
        flush=True,
    )
    return "saved"


# ----------------------------------------------------------------------------
# Worker 进程入口
# ----------------------------------------------------------------------------


def _worker_main(
    worker_id: int,
    args: argparse.Namespace,
    counter: Any,
    output_dir_str: str,
    progress_q: Any,
) -> dict[str, int]:
    """单进程入口: 启动 asyncio 循环, 在信号量限制下循环采集."""
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = build_backend(args)
    backend.warmup()
    sem = asyncio.Semaphore(args.per_process)

    saved = 0
    rejected = 0
    errors = 0
    log_prefix = f"[pid={os.getpid()} w={worker_id} b={backend.name}]"

    async def loop() -> None:
        nonlocal saved, rejected, errors
        auth = EpayAuth()
        try:
            while True:
                if counter.value >= args.count:
                    return
                async with sem:
                    status = await collect_one(
                        backend, auth, counter, output_dir, log_prefix
                    )
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


# ----------------------------------------------------------------------------
# 主进程: 启动 Pool + 进度监控
# ----------------------------------------------------------------------------


def _spawn_workers(args: argparse.Namespace) -> None:
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 用 'spawn' 避免 fork 出 torch / onnxruntime 子进程死锁
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
                _worker_main,
                (i, args, counter, str(output_dir), progress_q),
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

    final = counter.value
    print(f"[main] done. saved={final} target={args.count} output={output_dir}")


def _next_start_index(output_dir: Path) -> int:
    """扫描已有 8 位编号文件, 返回下一个可用序号 (断点续采)."""
    max_idx = -1
    for p in output_dir.glob("[0-9]" * 8 + ".jpg"):
        try:
            v = int(p.stem)
            if v > max_idx:
                max_idx = v
        except ValueError:
            pass
    return max_idx + 1


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="上海海事大学 CAS 验证码数据集采集器 (多进程 + 多会话)"
    )
    p.add_argument("--backend", choices=["restful", "tcp", "pytorch"], default="restful")
    p.add_argument("--output", default="./dataset", help="输出目录 (含 .jpg + .json)")
    p.add_argument("--count", type=int, default=1000, help="目标成功保存数量")
    p.add_argument("--processes", type=int, default=4, help="进程数")
    p.add_argument("--per-process", type=int, default=4, help="每进程内并发协程数")
    p.add_argument("--throttle", type=float, default=0.0, help="每次请求后睡眠秒")
    p.add_argument("--report-interval", type=float, default=5.0)

    p.add_argument("--ocr-url", default="http://127.0.0.1:21600",
                   help="RESTful OCR base url (HTTP 端口 21600, 见 shmtu-ocr-server)")
    p.add_argument("--ocr-timeout", type=float, default=10.0)

    p.add_argument("--ocr-host", default="127.0.0.1")
    p.add_argument("--ocr-port", type=int, default=21601,
                   help="TCP 端口 21601, 见 shmtu-ocr-server/main.rs tcp_port")

    p.add_argument("--weights-dir", default="./weights",
                   help="PyTorch 权重缓存目录 (自动从 GitHub release 下载)")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.processes < 1:
        raise SystemExit("--processes 必须 >= 1")
    if args.per_process < 1:
        raise SystemExit("--per-process 必须 >= 1")
    if args.count < 1:
        raise SystemExit("--count 必须 >= 1")
    _spawn_workers(args)


if __name__ == "__main__":
    main()
