"""数据集采集器 (maker) — 多进程 + 多会话 CAS 验证码图片 + json 标签采集."""
from .config import (
    GITHUB_RELEASE_BASE,
    GITHUB_RELEASE_TAG,
    PROBE_PASSWORD_CHARS,
    PROBE_PASSWORD_LEN,
    PROBE_USERNAME_PREFIX,
    USER_AGENT,
    WEIGHT_FILES,
    build_arg_parser,
    build_eval_arg_parser,
    format_eta,
    random_probe_account,
    scan_existing_max_index,
)

_LAZY_EXPORTS: dict[str, str] = {
    "OcrBackend": ".ocr_backends:OcrBackend",
    "OcrModel": ".ocr_backends:OcrModel",
    "RestfulBackend": ".ocr_backends:RestfulBackend",
    "TcpBackend": ".ocr_backends:TcpBackend",
    "ModelBackend": ".ocr_backends:ModelBackend",
    "PytorchV1Model": ".ocr_backends:PytorchV1Model",
    "PytorchV2Model": ".ocr_backends:PytorchV2Model",
    "LOCAL_MODEL_BUILDERS": ".ocr_backends:LOCAL_MODEL_BUILDERS",
    "build_model": ".ocr_backends:build_model",
    "build_backend": ".ocr_backends:build_backend",
    "verify_one": ".cas_client:verify_one",
    "collect_one": ".cas_client:collect_one",
    "spawn_workers": ".pool:spawn_workers",
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, _, attr = target.partition(":")
    import importlib

    mod = importlib.import_module(mod_name, __name__)
    value = getattr(mod, attr)
    globals()[name] = value
    return value


__all__ = [
    "build_arg_parser",
    "build_eval_arg_parser",
    "format_eta",
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
    "OcrModel",
    "RestfulBackend",
    "TcpBackend",
    "ModelBackend",
    "PytorchV1Model",
    "PytorchV2Model",
    "LOCAL_MODEL_BUILDERS",
    "build_model",
    "build_backend",
    "verify_one",
    "collect_one",
    "spawn_workers",
]
