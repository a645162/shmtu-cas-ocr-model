"""数据集采集器 (maker) — 多进程 + 多会话 CAS 验证码图片 + json 标签采集.

入口:
    python -m cas_ocr_model.datasets.maker --backend restful ...

模块拆分 (便于扩展与测试):
    config        argparse + 公共常量 (GitHub Release URL / USER_AGENT 等)
                  + scan_existing_max_index (断点续采扫描工具)
    ocr_backends  Restful / Tcp / Pytorch 三种 OCR 后端
    cas_client    EpayAuth 三阶段流程 (probe -> challenge -> submit) 的薄包装
    worker        单 worker 入口 (asyncio 循环 + 共享 EpayAuth)
    pool          多进程 Pool 调度 + 进度监控
    cli           命令行入口

断点续采: 默认从 --output 已有最大编号 + 1 继续, 不会覆盖已有数据.
        --resume 显式开关, 行为一致, 仅日志多打 "explicit resume" 标记.

子模块懒加载 (PEP 562): cas_client / ocr_backends / pool / worker 都在第一次
访问对应符号时才真正 import, 这样 config 层的工具 (scan_existing_max_index /
build_arg_parser) 可被测试和静态分析在缺少 shmtu_cas / torch 的环境下直接使用.
"""
from .config import (  # config 不依赖外部重量级库, 顶层 import 安全
    GITHUB_RELEASE_BASE,
    GITHUB_RELEASE_TAG,
    PROBE_PASSWORD_CHARS,
    PROBE_PASSWORD_LEN,
    PROBE_USERNAME_PREFIX,
    USER_AGENT,
    WEIGHT_FILES,
    build_arg_parser,
    format_eta,
    random_probe_account,
    scan_existing_max_index,
)


_LAZY_EXPORTS: dict[str, str] = {
    # name -> "module.attr"
    "OcrBackend": ".ocr_backends:OcrBackend",
    "RestfulBackend": ".ocr_backends:RestfulBackend",
    "TcpBackend": ".ocr_backends:TcpBackend",
    "PytorchBackend": ".ocr_backends:PytorchBackend",
    "build_backend": ".ocr_backends:build_backend",
    "collect_one": ".cas_client:collect_one",
    "spawn_workers": ".pool:spawn_workers",
}


def __getattr__(name: str):  # PEP 562
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, _, attr = target.partition(":")
    import importlib

    mod = importlib.import_module(mod_name, __name__)
    value = getattr(mod, attr)
    globals()[name] = value  # 缓存, 后续访问不再走 __getattr__
    return value


__all__ = [
    "build_arg_parser",
    "random_probe_account",
    "scan_existing_max_index",
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
