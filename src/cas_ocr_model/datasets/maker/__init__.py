"""数据集采集器 (maker) — 多进程 + 多会话 CAS 验证码图片 + json 标签采集.

入口:
    python -m cas_ocr_model.datasets.maker --backend restful ...

模块拆分 (便于扩展与测试):
    config        argparse + 公共常量 (GitHub Release URL / USER_AGENT 等)
    ocr_backends  Restful / Tcp / Pytorch 三种 OCR 后端
    cas_client    EpayAuth 三阶段流程 (probe -> challenge -> submit) 的薄包装
    worker        单 worker 入口 (asyncio 循环 + 共享 EpayAuth)
    pool          多进程 Pool 调度 + 进度监控
    cli           命令行入口
"""
from .cas_client import collect_one
from .config import (
    GITHUB_RELEASE_BASE,
    GITHUB_RELEASE_TAG,
    PROBE_PASSWORD_CHARS,
    PROBE_PASSWORD_LEN,
    PROBE_USERNAME_PREFIX,
    USER_AGENT,
    WEIGHT_FILES,
    build_arg_parser,
    random_probe_account,
)
from .ocr_backends import OcrBackend, PytorchBackend, RestfulBackend, TcpBackend, build_backend
from .pool import spawn_workers

__all__ = [
    "build_arg_parser",
    "random_probe_account",
    "USER_AGENT",
    "PROBE_USERNAME_PREFIX",
    "PROBE_PASSWORD_CHARS",
    "PROBE_PASSWORD_LEN",
    "GITHUB_RELEASE_BASE",
    "GITHUB_RELEASE_TAG",
    "WEIGHT_FILES",
    "OcrBackend",
    "RestfulBackend",
    "TcpBackend",
    "PytorchBackend",
    "build_backend",
    "collect_one",
    "spawn_workers",
]
