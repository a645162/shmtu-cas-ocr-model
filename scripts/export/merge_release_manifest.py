#!/usr/bin/env python3
"""合并 per-engine 元数据 JSON 为最终 release manifest.

本脚本是自包含的 (仅依赖 stdlib), 不需要 PyTorch 或任何项目依赖.
专为 CI finalize 阶段设计, 无需安装重依赖即可运行.

用法:
    python3 scripts/export/merge_release_manifest.py \\
        --output-root release_export \\
        --upload-list release_upload_list.tsv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_artifacts_by_model(
    artifacts: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    grouped: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for artifact in artifacts:
        asset_stem = str(artifact.get("asset_stem", "")).strip()
        engine = str(artifact.get("engine", "")).strip()
        precision = str(artifact.get("precision", "")).strip()
        if not asset_stem or not engine or not precision:
            continue
        grouped.setdefault(asset_stem, {}).setdefault(engine, {})[precision] = {
            "engine": engine,
            "precision": precision,
            "format": artifact.get("format"),
            "files": artifact.get("files", []),
        }
    return grouped


def build_release_manifest(
    *,
    model_entries: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    digests: list[dict[str, Any]],
    generated_at_utc: str | None = None,
    schema_version: int = 2,
) -> dict[str, Any]:
    grouped = group_artifacts_by_model(artifacts)
    deduped_models: list[dict[str, Any]] = []
    seen_asset_stems: set[str] = set()
    for entry in model_entries:
        asset_stem = str(entry.get("asset_stem", "")).strip()
        if not asset_stem or asset_stem in seen_asset_stems:
            continue
        seen_asset_stems.add(asset_stem)
        deduped_models.append(
            {
                **entry,
                "artifacts": grouped.get(asset_stem, {}),
            }
        )

    return {
        "schema_version": schema_version,
        "generated_at_utc": generated_at_utc or datetime.now(timezone.utc).isoformat(),
        "model_count": len(deduped_models),
        "modellist": [entry["asset_stem"] for entry in deduped_models],
        "models": deduped_models,
        "artifacts": artifacts,
        "digests": digests,
    }


def compute_sha256sums(output_root: Path) -> tuple[Path, int]:
    """计算 SHA256SUMS.txt, 排除 _meta_*.json 中间文件."""
    digest_path = output_root / "SHA256SUMS.txt"
    files = sorted(
        path
        for path in output_root.rglob("*")
        if path.is_file()
        and path.resolve() != digest_path.resolve()
        and "__pycache__" not in path.parts
        and not path.name.startswith("_meta_")
    )
    lines: list[str] = []
    for path in files:
        rel = path.relative_to(output_root).as_posix()
        digest = sha256_file(path)
        lines.append(f"{digest}  {rel}")
    digest_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return digest_path, len(files)


ALLOWED_SUFFIXES = {".pt", ".pth", ".onnx", ".param", ".bin", ".txt", ".json"}


def collect_release_uploads(output_root: Path, manifest: dict[str, Any]) -> list[tuple[Path, str]]:
    """从 manifest 生成 release 上传清单."""
    uploads: list[tuple[Path, str]] = []
    manifest_path = output_root / "model-assets.json"
    uploads.append((manifest_path, manifest_path.name))

    for artifact in manifest.get("artifacts", []):
        for file_info in artifact.get("files", []):
            rel_path = Path(file_info["path"])
            abs_path = (output_root / rel_path).resolve()
            uploads.append((abs_path, file_info.get("release_asset_name", abs_path.name)))

    for digest in manifest.get("digests", []):
        rel_path = Path(digest["path"])
        abs_path = (output_root / rel_path).resolve()
        uploads.append((abs_path, digest.get("release_asset_name", abs_path.name)))

    digest_path = output_root / "SHA256SUMS.txt"
    if digest_path.is_file():
        uploads.append((digest_path, digest_path.name))

    deduped: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for abs_path, release_name in uploads:
        if abs_path in seen:
            continue
        seen.add(abs_path)
        deduped.append((abs_path, release_name))

    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(description="合并 per-engine 元数据为最终 release manifest (stdlib only)")
    parser.add_argument("--output-root", required=True, help="Release export 根目录")
    parser.add_argument("--upload-list", default=None, help="输出上传清单 TSV 路径")
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser().resolve()

    # 读取所有 _meta_*.json 文件
    meta_files = sorted(output_root.glob("_meta_*.json"))
    if not meta_files:
        print("[merge] 未找到 _meta_*.json 文件", file=sys.stderr)
        sys.exit(1)

    all_model_entries: list[dict[str, Any]] = []
    all_artifacts: list[dict[str, Any]] = []
    seen_stems: set[str] = set()

    for meta_file in meta_files:
        print(f"[merge] 读取 {meta_file.name}")
        data = json.loads(meta_file.read_text(encoding="utf-8"))

        for entry in data.get("model_entries", []):
            stem = entry.get("asset_stem", "")
            if stem and stem not in seen_stems:
                seen_stems.add(stem)
                all_model_entries.append(entry)

        all_artifacts.extend(data.get("artifacts", []))

    print(f"[merge] 模型条目: {len(all_model_entries)}")
    print(f"[merge] 资产条目: {len(all_artifacts)}")

    # 构建 manifest
    manifest = build_release_manifest(
        model_entries=all_model_entries,
        artifacts=all_artifacts,
        digests=[],
    )

    # 写出 manifest
    manifest_path = output_root / "model-assets.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[merge] Manifest -> {manifest_path}")

    # 计算 SHA256SUMS.txt
    digest_path, file_count = compute_sha256sums(output_root)
    print(f"[merge] Digest -> {digest_path} ({file_count} 文件)")

    # 更新 manifest 中的 digest 条目
    digest_sha256 = sha256_file(digest_path)
    manifest["digests"] = [
        {
            "engine": "release",
            "path": digest_path.relative_to(output_root).as_posix(),
            "release_asset_name": digest_path.name,
            "sha256": digest_sha256,
        }
    ]
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # 生成上传清单
    if args.upload_list:
        uploads = collect_release_uploads(output_root, manifest)
        lines: list[str] = []
        for abs_path, release_name in uploads:
            if not abs_path.is_file():
                print(f"[merge] 警告: release 资产缺失: {abs_path}", file=sys.stderr)
                continue
            if abs_path.suffix not in ALLOWED_SUFFIXES:
                print(f"[merge] 警告: 意外的资产后缀: {abs_path.name}", file=sys.stderr)
            lines.append(f"{abs_path}\t{release_name}")
        upload_path = Path(args.upload_list).expanduser().resolve()
        upload_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        print(f"[merge] 上传清单 -> {upload_path} ({len(lines)} 条)")

    print("[merge] 完成!")


if __name__ == "__main__":
    main()
