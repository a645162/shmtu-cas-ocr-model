#!/usr/bin/env python3
"""Repair release manifest/digest assets for a GitHub release via gh CLI.

This script only fetches release metadata and the current ``model-assets.json``
asset. It rebuilds ``SHA256SUMS.txt`` from the actual uploaded release assets,
refreshes ``model-assets.json`` digest metadata, and uploads the repaired files
back with ``gh release upload --clobber``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


MANIFEST_NAME = "model-assets.json"
DIGEST_NAME = "SHA256SUMS.txt"
V2_TAG_PATTERN = re.compile(r"^v2\.\d+\.\d+(?:[-+].*)?$")


def log(message: str) -> None:
    print(message, flush=True)


def run(cmd: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(
            f"command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"{result.stdout}{result.stderr}"
        )
    return result.stdout


def run_binary(cmd: list[str], *, output_path: Path) -> None:
    with output_path.open("wb") as handle:
        result = subprocess.run(cmd, stdout=handle, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise SystemExit(
            f"command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"{result.stderr.decode('utf-8', errors='replace')}"
        )


def infer_repo_from_git() -> str | None:
    result = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    remote = result.stdout.strip()
    if remote.endswith(".git"):
        remote = remote[:-4]
    if remote.startswith("git@"):
        _, _, path = remote.partition(":")
        parts = [part for part in path.split("/") if part]
    else:
        from urllib.parse import urlparse

        parsed = urlparse(remote)
        parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return None
    return f"{parts[-2]}/{parts[-1]}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair release manifest assets with gh CLI.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tag", help="Single release tag to repair")
    mode.add_argument(
        "--all-v2",
        action="store_true",
        help="Repair every published v2.x release",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="GitHub repo in owner/name form (defaults to GITHUB_REPOSITORY or git origin)",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Directory for downloaded assets; defaults to a temporary directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Rebuild files locally but do not upload them back",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of pruning manifest entries whose release assets are missing",
    )
    return parser.parse_args()


def resolve_repo(repo: str | None) -> str:
    resolved = repo or infer_repo_from_git() or None
    if not resolved:
        import os

        resolved = os.environ.get("GITHUB_REPOSITORY")
    if not resolved or "/" not in resolved:
        raise SystemExit("--repo is required unless GITHUB_REPOSITORY or git origin can be inferred")
    return resolved


def resolve_release_name(file_info: dict[str, Any]) -> str:
    return str(file_info.get("release_asset_name") or Path(file_info["path"]).name)


def is_v2_tag(tag: str) -> bool:
    return bool(V2_TAG_PATTERN.match(tag))


def list_v2_release_tags(repo: str) -> list[str]:
    output = run(
        [
            "gh",
            "release",
            "list",
            "--repo",
            repo,
            "--limit",
            "200",
            "--json",
            "tagName,isDraft",
        ]
    )
    items = json.loads(output)
    tags: list[str] = []
    for item in items:
        tag = str(item.get("tagName") or "").strip()
        if not tag:
            continue
        if item.get("isDraft"):
            continue
        if is_v2_tag(tag):
            tags.append(tag)
    return tags


def fetch_release(repo: str, tag: str) -> dict[str, Any]:
    output = run(
        [
            "gh",
            "api",
            f"repos/{repo}/releases/tags/{tag}",
        ]
    )
    return json.loads(output)


def asset_digest(asset: dict[str, Any]) -> str:
    raw = str(asset.get("digest") or "").strip()
    if not raw:
        raise SystemExit(f"release asset is missing digest metadata: {asset.get('name')}")
    prefix = "sha256:"
    return raw[len(prefix) :] if raw.startswith(prefix) else raw


def download_manifest_asset(repo: str, asset_id: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"[fetch] manifest asset -> {output_path}")
    run_binary(
        [
            "gh",
            "api",
            f"repos/{repo}/releases/assets/{asset_id}",
            "-H",
            "Accept: application/octet-stream",
        ],
        output_path=output_path,
    )


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"missing {MANIFEST_NAME}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def rebuild_manifest(
    manifest: dict[str, Any],
    release_assets: dict[str, dict[str, Any]],
    work_dir: Path,
    *,
    prune_missing_artifacts: bool,
) -> tuple[dict[str, Any], int]:
    kept_files = 0

    for artifact in manifest.get("artifacts", []):
        new_files: list[dict[str, Any]] = []
        for file_info in artifact.get("files", []):
            release_name = resolve_release_name(file_info)
            asset = release_assets.get(release_name)
            if asset is None:
                message = f"missing release asset referenced by manifest: {release_name}"
                if prune_missing_artifacts:
                    log(f"[prune] {message}")
                    continue
                raise SystemExit(message)
            updated = dict(file_info)
            updated["release_asset_name"] = release_name
            updated["sha256"] = asset_digest(asset)
            new_files.append(updated)
            kept_files += 1
        artifact["files"] = new_files

    digest_lines: list[str] = []
    for artifact in manifest.get("artifacts", []):
        for file_info in artifact.get("files", []):
            release_name = resolve_release_name(file_info)
            digest_lines.append(f"{file_info['sha256']}  {release_name}")

    digest_path = work_dir / DIGEST_NAME
    digest_path.write_text("\n".join(digest_lines) + ("\n" if digest_lines else ""), encoding="utf-8")
    digest_sha = sha256_file(digest_path)
    manifest["digests"] = [
        {
            "engine": "release",
            "path": DIGEST_NAME,
            "release_asset_name": DIGEST_NAME,
            "sha256": digest_sha,
        }
    ]
    return manifest, kept_files


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def upload_repaired_assets(repo: str, tag: str, work_dir: Path) -> None:
    manifest_path = work_dir / MANIFEST_NAME
    digest_path = work_dir / DIGEST_NAME
    run(
        [
            "gh",
            "release",
            "upload",
            tag,
            f"{manifest_path}#{MANIFEST_NAME}",
            f"{digest_path}#{DIGEST_NAME}",
            "--repo",
            repo,
            "--clobber",
        ]
    )


def repair_one_tag(
    *,
    repo: str,
    tag: str,
    work_dir_arg: str | None,
    dry_run: bool,
    prune_missing_artifacts: bool,
) -> int:
    if not is_v2_tag(tag):
        raise SystemExit(f"refusing to repair non-v2 tag: {tag}")

    cleanup: tempfile.TemporaryDirectory[str] | None = None
    if work_dir_arg:
        work_dir = Path(work_dir_arg).expanduser().resolve() / tag
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        cleanup = tempfile.TemporaryDirectory(prefix=f"repair-release-{tag}-")
        work_dir = Path(cleanup.name).resolve()

    try:
        release = fetch_release(repo, tag)
        assets = list(release.get("assets", []))
        release_assets_by_name = {str(asset.get("name")): asset for asset in assets if asset.get("name")}
        manifest_asset = release_assets_by_name.get(MANIFEST_NAME)
        if manifest_asset is None:
            raise SystemExit(f"missing {MANIFEST_NAME} in release {tag}")

        manifest_path = work_dir / MANIFEST_NAME
        download_manifest_asset(repo, int(manifest_asset["id"]), manifest_path)
        manifest = load_manifest(manifest_path)
        manifest, kept_files = rebuild_manifest(
            manifest,
            release_assets_by_name,
            work_dir,
            prune_missing_artifacts=prune_missing_artifacts,
        )
        write_manifest(manifest_path, manifest)
        log(f"[write] {manifest_path}")
        log(f"[write] {work_dir / DIGEST_NAME}")
        log(f"[summary] {tag}: kept artifact files: {kept_files}")
        if dry_run:
            log(f"[dry-run] {tag}: skip upload")
            return 0
        upload_repaired_assets(repo, tag, work_dir)
        log(f"[done] uploaded repaired {MANIFEST_NAME} and {DIGEST_NAME} to {repo}@{tag}")
        return 0
    finally:
        if cleanup is not None:
            cleanup.cleanup()


def main() -> int:
    if shutil.which("gh") is None:
        raise SystemExit("gh CLI is required")

    args = parse_args()
    repo = resolve_repo(args.repo)
    if args.tag:
        return repair_one_tag(
            repo=repo,
            tag=args.tag,
            work_dir_arg=args.work_dir,
            dry_run=args.dry_run,
            prune_missing_artifacts=not args.strict,
        )

    tags = list_v2_release_tags(repo)
    if not tags:
        log("[summary] no published v2.x releases found")
        return 0
    log(f"[summary] found {len(tags)} published v2.x release(s)")
    for index, tag in enumerate(tags, start=1):
        log(f"[release] {index}/{len(tags)} {tag}")
        repair_one_tag(
            repo=repo,
            tag=tag,
            work_dir_arg=args.work_dir,
            dry_run=args.dry_run,
            prune_missing_artifacts=not args.strict,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
