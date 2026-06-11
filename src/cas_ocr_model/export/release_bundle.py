"""从一个或多个 PyTorch checkpoint 生成 release 资产.

输出目录结构:
    output_root/
      model-assets.json
      pytorch/
      onnx/
      ncnn/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from cas_ocr_model.common.checkpoint_pip import (
    load_checkpoint_pip_snapshot,
    write_pip_list_json,
)
from cas_ocr_model.common.console import tag_print
from cas_ocr_model.common.release_manifest import build_release_manifest
from cas_ocr_model.model import inspect_checkpoint

SUPPORTED_ENGINES = ("pytorch", "onnx", "ncnn")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="批量导出 release 用的 pytorch/onnx/ncnn 资产")
    p.add_argument("--checkpoint", action="append", default=[], help="可重复指定多个 .pt checkpoint")
    p.add_argument("--checkpoint-dir", default=None, help="自动扫描目录内的 .pt/.pth")
    p.add_argument("--output-root", required=True, help="导出根目录")
    p.add_argument("--engines", default="pytorch onnx ncnn", help="空格或逗号分隔, 可选 pytorch/onnx/ncnn")
    p.add_argument("--precisions", default="fp16 fp32", help="空格或逗号分隔, 默认同时导出 fp16 和 fp32")
    p.add_argument("--image-size-h", type=int, default=None)
    p.add_argument("--image-size-w", type=int, default=None)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--legacy-exporter", action="store_true")
    p.add_argument("--dynamic-batch", action="store_true")
    p.add_argument("--onnx-device", choices=("auto", "cpu", "cuda"), default="cpu")
    p.add_argument("--ncnn-optlevel", type=int, default=2)
    p.add_argument("--finalize-only", action="store_true", help="不导出, 仅基于现有资产生成 manifest/digest")
    p.add_argument("--skip-manifest", action="store_true", help="导出后不生成 model-assets.json")
    p.add_argument("--skip-digest", action="store_true", help="导出后不生成 SHA256SUMS.txt")
    return p.parse_args()


def normalize_precisions(raw: str) -> list[str]:
    items = [item.strip() for item in raw.replace(",", " ").split() if item.strip()]
    if not items:
        raise SystemExit("未解析到任何 precision")
    normalized: list[str] = []
    for item in items:
        if item in {"fp16", "float16", "half"}:
            normalized.append("fp16")
        elif item in {"fp32", "float32"}:
            normalized.append("fp32")
        else:
            raise SystemExit(f"不支持的 precision: {item}")
    return normalized


def normalize_engines(raw: str) -> list[str]:
    items = [item.strip().lower() for item in raw.replace(",", " ").split() if item.strip()]
    if not items:
        raise SystemExit("未解析到任何 engine")
    normalized: list[str] = []
    for item in items:
        if item not in SUPPORTED_ENGINES:
            raise SystemExit(f"不支持的 engine: {item}")
        if item not in normalized:
            normalized.append(item)
    return normalized


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sha256sums(root: Path, *, files: list[Path]) -> Path:
    digest_path = root / "SHA256SUMS.txt"
    lines: list[str] = []
    seen: set[Path] = set()
    for path in sorted(files):
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        if not path.is_file() or path.resolve() == digest_path.resolve():
            continue
        lines.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    digest_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return digest_path


def cleanup_release_directory(directory: Path) -> None:
    for pycache_dir in directory.rglob("__pycache__"):
        if pycache_dir.is_dir():
            shutil.rmtree(pycache_dir, ignore_errors=True)

    for path in directory.rglob("*.pyc"):
        path.unlink(missing_ok=True)

    for path in directory.rglob("*.pyo"):
        path.unlink(missing_ok=True)

    for path in directory.rglob("*.py"):
        if path.name.endswith("_pnnx.py"):
            path.unlink(missing_ok=True)


def resolve_checkpoints(args: argparse.Namespace) -> list[Path]:
    checkpoints = [Path(item).expanduser().resolve() for item in args.checkpoint]
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
        checkpoints.extend(sorted(checkpoint_dir.glob("*.pt")))
        checkpoints.extend(sorted(checkpoint_dir.glob("*.pth")))
    deduped: list[Path] = []
    seen: set[Path] = set()
    for checkpoint in checkpoints:
        if checkpoint in seen:
            continue
        seen.add(checkpoint)
        deduped.append(checkpoint)
    if not deduped:
        raise SystemExit("未找到任何 checkpoint")
    for checkpoint in deduped:
        if not checkpoint.is_file():
            raise SystemExit(f"checkpoint 不存在: {checkpoint}")
    return deduped


def infer_image_sizes(checkpoint: Path, *, override_h: int | None, override_w: int | None) -> tuple[int, int]:
    raw = torch.load(checkpoint, map_location="cpu")
    cfg = raw.get("config", {}) if isinstance(raw, dict) else {}
    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    image_size_h = override_h if override_h is not None else int(data_cfg.get("image_size_h", 64))
    image_size_w = override_w if override_w is not None else int(data_cfg.get("image_size_w", 192))
    return image_size_h, image_size_w


def run_command(args: list[str]) -> None:
    tag_print("release-export", f"run: {' '.join(args)}")
    subprocess.run(args, check=True)


def copy_pytorch_checkpoint(src: Path, dst: Path) -> dict[str, Any]:
    shutil.copy2(src, dst)
    return {
        "engine": "pytorch",
        "precision": "fp32",
        "format": "checkpoint",
        "files": [
            {
                "path": str(dst),
                "sha256": sha256_file(dst),
            }
        ],
    }


def collect_pytorch_artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"缺少 pytorch release 资产: {path}")
    files = [
        {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    ]
    pip_list, _ = load_checkpoint_pip_snapshot(path)
    if pip_list is not None:
        pip_list_path = path.with_name(f"{path.stem}.pip-list.json")
        write_pip_list_json(pip_list_path, pip_list)
        files.append(
            {
                "path": str(pip_list_path),
                "sha256": sha256_file(pip_list_path),
            }
        )
    else:
        tag_print("release-export", f"checkpoint missing pip_list metadata: {path.name}")
    return {
        "engine": "pytorch",
        "precision": "fp32",
        "format": "checkpoint",
        "files": files,
    }


def export_onnx_bundle(
    *,
    checkpoint: Path,
    asset_stem: str,
    output_dir: Path,
    precisions: list[str],
    image_size_h: int,
    image_size_w: int,
    opset: int,
    legacy_exporter: bool,
    dynamic_batch: bool,
    onnx_device: str,
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for precision in precisions:
        output = output_dir / f"{asset_stem}.{precision}.onnx"
        cmd = [
            sys.executable,
            "-m",
            "cas_ocr_model.trainer.export",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--image-size-h",
            str(image_size_h),
            "--image-size-w",
            str(image_size_w),
            "--opset",
            str(opset),
            "--precision",
            precision,
            "--device",
            onnx_device,
        ]
        if legacy_exporter:
            cmd.append("--legacy-exporter")
        if dynamic_batch:
            cmd.append("--dynamic-batch")
        run_command(cmd)
        artifacts.append(
            {
                "engine": "onnx",
                "precision": precision,
                "format": "onnx",
                "files": [
                    {
                        "path": str(output),
                        "sha256": sha256_file(output),
                    }
                ],
            }
        )
    return artifacts


def collect_onnx_artifacts(*, asset_stem: str, output_dir: Path, precisions: list[str]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for precision in precisions:
        output = output_dir / f"{asset_stem}.{precision}.onnx"
        if not output.is_file():
            raise SystemExit(f"缺少 onnx release 资产: {output}")
        artifacts.append(
            {
                "engine": "onnx",
                "precision": precision,
                "format": "onnx",
                "files": [
                    {
                        "path": str(output),
                        "sha256": sha256_file(output),
                    }
                ],
            }
        )
    return artifacts


def export_ncnn_bundle(
    *,
    checkpoint: Path,
    asset_stem: str,
    output_dir: Path,
    precisions: list[str],
    image_size_h: int,
    image_size_w: int,
    optlevel: int,
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for precision in precisions:
        output_pt = output_dir / f"{asset_stem}.{precision}.pt"
        output_param = output_dir / f"{asset_stem}.{precision}.param"
        output_bin = output_dir / f"{asset_stem}.{precision}.bin"
        pnnx_param = output_dir / f"{asset_stem}.{precision}.pnnx.param"
        pnnx_bin = output_dir / f"{asset_stem}.{precision}.pnnx.bin"
        pnnx_py = output_dir / f"{asset_stem}.{precision}_pnnx.py"
        pnnx_onnx = output_dir / f"{asset_stem}.{precision}.pnnx.onnx"
        cmd = [
            sys.executable,
            "-m",
            "cas_ocr_model.export.export_ncnn",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output_pt),
            "--image-size-h",
            str(image_size_h),
            "--image-size-w",
            str(image_size_w),
            "--precision",
            precision,
            "--optlevel",
            str(optlevel),
            "--ncnn-param",
            str(output_param),
            "--ncnn-bin",
            str(output_bin),
            "--pnnx-param",
            str(pnnx_param),
            "--pnnx-bin",
            str(pnnx_bin),
            "--pnnx-py",
            str(pnnx_py),
            "--pnnx-onnx",
            str(pnnx_onnx),
        ]
        run_command(cmd)
        for intermediate in (output_pt, pnnx_param, pnnx_bin, pnnx_py, pnnx_onnx):
            intermediate.unlink(missing_ok=True)
        cleanup_release_directory(output_dir)
        artifacts.append(
            {
                "engine": "ncnn",
                "precision": precision,
                "format": "ncnn",
                "files": [
                    {
                        "path": str(output_param),
                        "sha256": sha256_file(output_param),
                    },
                    {
                        "path": str(output_bin),
                        "sha256": sha256_file(output_bin),
                    },
                ],
            }
        )
    return artifacts


def collect_ncnn_artifacts(*, asset_stem: str, output_dir: Path, precisions: list[str]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for precision in precisions:
        output_param = output_dir / f"{asset_stem}.{precision}.param"
        output_bin = output_dir / f"{asset_stem}.{precision}.bin"
        if not output_param.is_file():
            raise SystemExit(f"缺少 ncnn release 资产: {output_param}")
        if not output_bin.is_file():
            raise SystemExit(f"缺少 ncnn release 资产: {output_bin}")
        artifacts.append(
            {
                "engine": "ncnn",
                "precision": precision,
                "format": "ncnn",
                "files": [
                    {
                        "path": str(output_param),
                        "sha256": sha256_file(output_param),
                    },
                    {
                        "path": str(output_bin),
                        "sha256": sha256_file(output_bin),
                    },
                ],
            }
        )
    return artifacts


def relativize_artifact_paths(output_root: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    converted = dict(artifact)
    converted["files"] = [
        {
            **file_info,
            "path": Path(file_info["path"]).resolve().relative_to(output_root.resolve()).as_posix(),
            "release_asset_name": Path(file_info["path"]).name,
        }
        for file_info in artifact["files"]
    ]
    return converted


def main() -> None:
    args = parse_args()
    checkpoints = resolve_checkpoints(args)
    engines = normalize_engines(args.engines)
    precisions = normalize_precisions(args.precisions)

    output_root = Path(args.output_root).expanduser().resolve()
    pytorch_dir = output_root / "pytorch"
    onnx_dir = output_root / "onnx"
    ncnn_dir = output_root / "ncnn"
    engine_dirs = {
        "pytorch": pytorch_dir,
        "onnx": onnx_dir,
        "ncnn": ncnn_dir,
    }
    for directory in (pytorch_dir, onnx_dir, ncnn_dir):
        directory.mkdir(parents=True, exist_ok=True)

    used_asset_stems: dict[str, str] = {}
    resolved_models: list[tuple[Path, dict[str, Any]]] = []
    for checkpoint in checkpoints:
        metadata = inspect_checkpoint(checkpoint)
        asset_stem = metadata["asset_stem"]
        checkpoint_sha256 = sha256_file(checkpoint)
        if asset_stem in used_asset_stems:
            if used_asset_stems[asset_stem] == checkpoint_sha256:
                tag_print("release-export", f"skip duplicate checkpoint: {checkpoint} -> {asset_stem}")
                continue
            raise SystemExit(f"重复的 release asset_stem 且内容不同: {asset_stem}")
        used_asset_stems[asset_stem] = checkpoint_sha256
        resolved_models.append((checkpoint, metadata))

        image_size_h, image_size_w = infer_image_sizes(
            checkpoint,
            override_h=args.image_size_h,
            override_w=args.image_size_w,
        )

        if args.finalize_only:
            continue

        if "pytorch" in engines:
            pytorch_path = pytorch_dir / f"{asset_stem}.pt"
            copy_pytorch_checkpoint(checkpoint, pytorch_path)
        if "onnx" in engines:
            export_onnx_bundle(
                checkpoint=checkpoint,
                asset_stem=asset_stem,
                output_dir=onnx_dir,
                precisions=precisions,
                image_size_h=image_size_h,
                image_size_w=image_size_w,
                opset=args.opset,
                legacy_exporter=args.legacy_exporter,
                dynamic_batch=args.dynamic_batch,
                onnx_device=args.onnx_device,
            )
        if "ncnn" in engines:
            export_ncnn_bundle(
                checkpoint=checkpoint,
                asset_stem=asset_stem,
                output_dir=ncnn_dir,
                precisions=precisions,
                image_size_h=image_size_h,
                image_size_w=image_size_w,
                optlevel=args.ncnn_optlevel,
            )

    cleanup_targets = list(engine_dirs.values()) if args.finalize_only else [engine_dirs[engine] for engine in engines]
    for directory in cleanup_targets:
        cleanup_release_directory(directory)

    if args.skip_manifest and args.skip_digest:
        return

    manifest_artifacts: list[dict[str, Any]] = []
    release_files: list[Path] = []
    for _, metadata in resolved_models:
        asset_stem = metadata["asset_stem"]

        if "pytorch" in engines:
            artifact = collect_pytorch_artifact(pytorch_dir / f"{asset_stem}.pt")
            release_files.extend(Path(file_info["path"]) for file_info in artifact["files"])
            manifest_artifacts.append(
                {
                    **metadata,
                    **relativize_artifact_paths(output_root, artifact),
                }
            )
        if "onnx" in engines:
            for artifact in collect_onnx_artifacts(asset_stem=asset_stem, output_dir=onnx_dir, precisions=precisions):
                release_files.extend(Path(file_info["path"]) for file_info in artifact["files"])
                manifest_artifacts.append(
                    {
                        **metadata,
                        **relativize_artifact_paths(output_root, artifact),
                    }
                )
        if "ncnn" in engines:
            for artifact in collect_ncnn_artifacts(asset_stem=asset_stem, output_dir=ncnn_dir, precisions=precisions):
                release_files.extend(Path(file_info["path"]) for file_info in artifact["files"])
                manifest_artifacts.append(
                    {
                        **metadata,
                        **relativize_artifact_paths(output_root, artifact),
                    }
                )

    manifest_digests: list[dict[str, Any]] = []
    if not args.skip_digest:
        digest_path = write_sha256sums(output_root, files=release_files)
        manifest_digests.append(
            {
                "engine": "release",
                "path": digest_path.relative_to(output_root).as_posix(),
                "release_asset_name": digest_path.name,
                "sha256": sha256_file(digest_path),
            }
        )

    manifest = build_release_manifest(
        model_entries=[metadata for _, metadata in resolved_models],
        artifacts=manifest_artifacts,
        digests=manifest_digests,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        schema_version=2,
    )

    if args.skip_manifest:
        return

    manifest_path = output_root / "model-assets.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tag_print("release-export", f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
