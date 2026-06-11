"""收集运行时与模型元信息的工具函数.

提供:
 - collect_run_metadata(model): 返回 model_size_m 与 environment 信息
 - collect_pip_snapshot(): 封装已有的 pip 列表采集
"""
from __future__ import annotations

import getpass
import platform
import subprocess
from typing import Any

from cas_ocr_model.model.stats import count_parameters
from cas_ocr_model.common.checkpoint_pip import capture_pip_list_snapshot


def collect_run_metadata(model: Any) -> dict[str, Any]:
    """收集模型尺寸与运行环境信息。"""
    meta: dict[str, Any] = {}
    try:
        total_params, trainable = count_parameters(model)
        model_size_m = round(float(total_params) / 1_000_000.0, 2)
        meta["model_size_m"] = model_size_m
    except Exception:
        meta["model_size_m"] = None

    try:
        training_user = getpass.getuser()
    except Exception:
        training_user = None
    try:
        training_host = platform.node()
    except Exception:
        training_host = None

    n_gpus = 0
    main_device = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            n_gpus = int(torch.cuda.device_count())
            try:
                main_device = torch.cuda.get_device_name(0)
            except Exception:
                main_device = "cuda:0"
    except Exception:
        pass

    git_commit = None
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        git_commit = None

    versions = {"python": platform.python_version()}
    try:
        import torch as _torch

        versions["torch"] = str(getattr(_torch, "__version__", None))
        try:
            versions["cuda"] = _torch.version.cuda if hasattr(_torch, "version") else None
        except Exception:
            versions["cuda"] = None
    except Exception:
        versions.setdefault("torch", None)
        versions.setdefault("cuda", None)

    meta["environment"] = {
        "training_user": training_user,
        "training_host": training_host,
        "main_device": main_device,
        "n_gpus": int(n_gpus),
        "git_commit": git_commit,
        "versions": versions,
    }

    try:
        pip_list, pip_list_metadata = capture_pip_list_snapshot()
        meta["pip_list"] = pip_list
        meta["pip_list_metadata"] = pip_list_metadata
    except Exception:
        meta.setdefault("pip_list", None)
        meta.setdefault("pip_list_metadata", None)

    return meta


def collect_pip_snapshot() -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None]:
    try:
        return capture_pip_list_snapshot()
    except Exception:
        return None, None
