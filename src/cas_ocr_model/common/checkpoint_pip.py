"""PyTorch checkpoint 里的 pip list 元数据读写工具."""
from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch


@lru_cache(maxsize=1)
def capture_pip_list_snapshot() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--format=json"],
        check=True,
        capture_output=True,
        text=True,
    )
    raw = json.loads(proc.stdout)
    if not isinstance(raw, list):
        raise RuntimeError("pip list --format=json did not return a JSON list")

    packages: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        package: dict[str, Any] = {}
        for key, value in item.items():
            if isinstance(key, str):
                package[key] = value
        if package:
            packages.append(package)

    metadata = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "pip_command": [sys.executable, "-m", "pip", "list", "--format=json"],
    }
    return packages, metadata


def extract_checkpoint_pip_list(raw_checkpoint: Any) -> list[dict[str, Any]] | None:
    if not isinstance(raw_checkpoint, Mapping):
        return None

    raw = raw_checkpoint.get("pip_list")
    if not isinstance(raw, list):
        environment = raw_checkpoint.get("environment")
        if isinstance(environment, Mapping):
            raw = environment.get("pip_list")
    if not isinstance(raw, list):
        return None

    packages: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        package: dict[str, Any] = {}
        for key, value in item.items():
            if isinstance(key, str):
                package[key] = value
        if package:
            packages.append(package)
    return packages


def extract_checkpoint_pip_metadata(raw_checkpoint: Any) -> dict[str, Any]:
    if not isinstance(raw_checkpoint, Mapping):
        return {}

    raw = raw_checkpoint.get("pip_list_metadata")
    if not isinstance(raw, Mapping):
        environment = raw_checkpoint.get("environment")
        if isinstance(environment, Mapping):
            raw = environment.get("pip_list_metadata")
    if not isinstance(raw, Mapping):
        return {}

    metadata: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(key, str):
            metadata[key] = value
    return metadata


def load_checkpoint_pip_snapshot(checkpoint: str | Path) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    raw = torch.load(checkpoint, map_location="cpu")
    return extract_checkpoint_pip_list(raw), extract_checkpoint_pip_metadata(raw)


def write_pip_list_json(
    output_path: str | Path,
    packages: list[dict[str, Any]],
) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(packages, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
