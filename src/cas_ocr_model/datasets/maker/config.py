"""采集器配置: argparse + 公共常量."""
from __future__ import annotations

import argparse
import random
import string
import urllib.request
from pathlib import Path


PROBE_USERNAME_PREFIX = "ds_cap_"
PROBE_PASSWORD_LEN = 12
PROBE_PASSWORD_CHARS = string.ascii_letters + string.digits

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


def random_probe_account() -> tuple[str, str]:
    """生成随机学号/密码, 与 Rust 端 CAPTCHA_TEST_PROBE_* 等效但每次不同."""
    student_no = PROBE_USERNAME_PREFIX + "".join(
        random.choices(string.digits, k=max(0, 10 - len(PROBE_USERNAME_PREFIX)))
    )
    password = "".join(random.choices(PROBE_PASSWORD_CHARS, k=PROBE_PASSWORD_LEN))
    return student_no, password


def download_file(url: str, dest: Path) -> None:
    """流式下载, 进度提示; 已存在且大小 > 0 则跳过."""
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


def build_arg_parser() -> argparse.ArgumentParser:
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
                   help="RESTful OCR base url (对齐 shmtu-ocr-server HTTP 默认端口 21600)")
    p.add_argument("--ocr-timeout", type=float, default=10.0)

    p.add_argument("--ocr-host", default="127.0.0.1")
    p.add_argument("--ocr-port", type=int, default=21601,
                   help="对齐 shmtu-ocr-server TCP 默认端口 21601")

    p.add_argument("--weights-dir", default="./weights",
                   help="PyTorch 权重缓存目录 (自动从 GitHub release 下载)")

    return p
