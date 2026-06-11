"""数据集 manifest 摘要工具.

提供:
 - collect_manifest_summary(dataset_root) -> dict
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .format import DatasetManifest, MANIFEST_FILENAME


def collect_manifest_summary(dataset_root: str | Path) -> dict[str, Any]:
    path = Path(dataset_root) / MANIFEST_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    m = DatasetManifest.load(dataset_root)
    # compute sha256
    d = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            d.update(chunk)
    return {
        "created_at": int(getattr(m, "created_at", 0)),
        "manifest_sha256": d.hexdigest(),
        "stats": getattr(m, "stats", {}),
        "label_set": getattr(m, "label_set", {}),
        "version": getattr(m, "version", None),
    }
