#!/usr/bin/env python3
"""Simulate GitHub Action extraction of model assets.

Scans a run directory and its release/pytorch folder for .pt files, safely loads
each checkpoint (with fallbacks for PyTorch weights-only loading changes),
extracts `model_metadata` and `pip_list` info, and writes a combined
`model-assets-simulated.json` in the release directory for inspection.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def safe_torch_load(path: Path) -> Any:
    import torch
    from torch.serialization import add_safe_globals

    # Try normal load first (explicitly set weights_only=False to allow metadata)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        # Try to add known safe globals suggested by PyTorch error messages
        try:
            add_safe_globals([torch.torch_version.TorchVersion])
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            # As a last resort, try weights_only=True (may lose metadata)
            try:
                return torch.load(path, map_location="cpu", weights_only=True)
            except Exception:
                raise


def extract_from_checkpoint(raw: Any) -> dict[str, Any]:
    # Import here to ensure repo src is on PYTHONPATH
    from cas_ocr_model.model.registry import extract_checkpoint_metadata
    from cas_ocr_model.common.checkpoint_pip import extract_checkpoint_pip_list

    meta = extract_checkpoint_metadata(raw)
    pip_list = None
    try:
        pip_list = extract_checkpoint_pip_list(raw)
    except Exception:
        pip_list = None
    return {"model_metadata": meta, "pip_list": pip_list}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", help="Path to run directory (where best.pt/last.pt live)")
    args = p.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    release_root = run_dir / "release"
    pytorch_dir = release_root / "pytorch"

    # Collect candidate checkpoints
    candidates: list[Path] = []
    for name in ("best.pt", "last.pt"):
        path = run_dir / name
        if path.is_file():
            candidates.append(path)

    if pytorch_dir.is_dir():
        for path in sorted(pytorch_dir.glob("*.pt")):
            candidates.append(path)

    if not candidates:
        print(f"no .pt files found under {run_dir} or {pytorch_dir}")
        raise SystemExit(2)

    results: list[dict[str, Any]] = []
    for ckpt_path in candidates:
        print(f"processing {ckpt_path}")
        try:
            raw = safe_torch_load(ckpt_path)
        except Exception as exc:
            print(f"WARN: failed to load {ckpt_path}: {exc}")
            continue
        try:
            info = extract_from_checkpoint(raw)
        except Exception as exc:
            print(f"WARN: failed to extract metadata from {ckpt_path}: {exc}")
            continue
        entry = {
            "path": str(ckpt_path.relative_to(release_root).as_posix()) if release_root in ckpt_path.parents else str(ckpt_path.as_posix()),
            "model_metadata": info.get("model_metadata"),
            "has_pip_list": info.get("pip_list") is not None,
        }
        results.append(entry)

    out = {
        "run_dir": str(run_dir),
        "artifacts": results,
    }

    release_root.mkdir(parents=True, exist_ok=True)
    out_path = release_root / "model-assets-simulated.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote simulated manifest to {out_path}")


if __name__ == "__main__":
    # Ensure repo src is on sys.path when running from project root
    repo_root = Path(__file__).resolve().parents[2]
    src = repo_root / "src"
    sys.path.insert(0, str(src))
    main()
