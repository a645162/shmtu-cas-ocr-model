#!/usr/bin/env python3
"""Download a GitHub release via gh CLI, then sync it to Gitee via API."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
from pathlib import Path
from typing import Any

from sync_gitee_releases import (
    ApiError,
    DesiredAsset,
    GiteeClient,
    GitHubClient,
    log,
    MANIFEST_ASSET_NAME,
    DIGEST_ASSET_NAME,
    plan_gitee_v2_slim_assets,
    purge_obsolete_gitee_v2_releases,
    require_env,
    select_latest_release,
    split_repo,
    sync_release,
    write_gitee_v2_slim_bundle,
    is_v2_tag,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use gh CLI to download a GitHub release locally, then sync it to Gitee."
    )
    parser.add_argument("--tag", help="GitHub release tag to sync. Defaults to the latest published release.")
    parser.add_argument(
        "--github-repo",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help="GitHub repo in owner/name form. Defaults to GITHUB_REPOSITORY.",
    )
    parser.add_argument(
        "--gitee-repo",
        default=os.environ.get("GITEE_REPO"),
        help="Gitee repo in owner/name form. Defaults to GITEE_REPO.",
    )
    parser.add_argument(
        "--download-dir",
        help="Directory to store downloaded assets. Defaults to a temporary directory.",
    )
    parser.add_argument(
        "--keep-downloads",
        action="store_true",
        help="Keep the temporary download directory when --download-dir is not provided.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print changes without mutating Gitee releases.",
    )
    return parser.parse_args()


def run_gh(args: list[str], *, expect_json: bool = False) -> Any:
    command = ["gh", *args]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise SystemExit("gh CLI not found in PATH") from exc

    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        detail = stderr or stdout or "unknown gh CLI error"
        raise ApiError(f"gh {' '.join(args)} failed: {detail}")

    if not expect_json:
        return completed.stdout

    text = completed.stdout.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ApiError(f"gh {' '.join(args)} returned invalid JSON") from exc


def fetch_github_release(repo: str, tag: str) -> dict[str, Any]:
    payload = run_gh(
        [
            "api",
            f"repos/{repo}/releases/tags/{urllib.parse.quote(tag, safe='')}",
        ],
        expect_json=True,
    )
    if not isinstance(payload, dict):
        raise ApiError("gh api returned an unexpected payload type")
    return payload


def fetch_github_releases(repo: str) -> list[dict[str, Any]]:
    payload = run_gh(
        [
            "api",
            f"repos/{repo}/releases",
        ],
        expect_json=True,
    )
    if not isinstance(payload, list):
        raise ApiError("gh api returned an unexpected releases payload type")
    return payload


def resolve_release(repo: str, tag: str | None) -> dict[str, Any]:
    if tag:
        return fetch_github_release(repo, tag)
    releases = fetch_github_releases(repo)
    release = select_latest_release(releases)
    log(f"[info] selected latest published release: {release['tag_name']}")
    return release


def download_release_assets(repo: str, release: dict[str, Any], dest_dir: Path) -> list[DesiredAsset]:
    tag = str(release["tag_name"])
    assets = release.get("assets", [])
    if not isinstance(assets, list):
        raise ApiError("release assets payload is not a list")
    if not assets:
        return []

    run_gh(
        [
            "release",
            "download",
            tag,
            "--repo",
            repo,
            "--dir",
            str(dest_dir),
            "--clobber",
        ]
    )

    desired_assets: list[DesiredAsset] = []
    for asset in assets:
        name = str(asset["name"])
        local_path = dest_dir / name
        if not local_path.is_file():
            raise ApiError(f"gh download completed but asset is missing locally: {name}")
        actual_size = local_path.stat().st_size
        expected_size = int(asset["size"])
        if actual_size != expected_size:
            raise ApiError(
                f"downloaded asset size mismatch for {name}: expected {expected_size}, got {actual_size}"
            )
        desired_assets.append(
            DesiredAsset(
                name=name,
                size=expected_size,
                local_path=local_path,
            )
        )
    return desired_assets


def download_release_assets_by_name(repo: str, tag: str, asset_names: list[str], dest_dir: Path) -> dict[str, Path]:
    if not asset_names:
        return {}
    args = [
        "release",
        "download",
        tag,
        "--repo",
        repo,
        "--dir",
        str(dest_dir),
        "--clobber",
    ]
    for name in asset_names:
        args.extend(["--pattern", name])
    run_gh(args)

    staged_paths: dict[str, Path] = {}
    for name in asset_names:
        local_path = dest_dir / name
        if not local_path.is_file():
            raise ApiError(f"gh download completed but asset is missing locally: {name}")
        staged_paths[name] = local_path
    return staged_paths


def prepare_local_gitee_release_assets(repo: str, release: dict[str, Any], dest_dir: Path) -> list[DesiredAsset]:
    tag = str(release["tag_name"])
    if not is_v2_tag(tag):
        return download_release_assets(repo, release, dest_dir)

    source_manifest_paths = download_release_assets_by_name(repo, tag, [MANIFEST_ASSET_NAME], dest_dir)
    source_manifest_path = source_manifest_paths[MANIFEST_ASSET_NAME]
    raw_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    _, _, selected_asset_names = plan_gitee_v2_slim_assets(raw_manifest)
    log(f"[filter] {tag}: downloading {len(selected_asset_names)} slim asset(s) via gh")
    staged_paths = download_release_assets_by_name(repo, tag, selected_asset_names, dest_dir)
    manifest_path, digest_path = write_gitee_v2_slim_bundle(
        raw_manifest,
        staged_asset_paths=staged_paths,
        output_dir=dest_dir,
    )

    release_assets = {
        str(asset["name"]): asset
        for asset in release.get("assets", [])
        if asset.get("name")
    }
    desired_assets: list[DesiredAsset] = []
    for name in selected_asset_names:
        asset = release_assets.get(name)
        if asset is None:
            raise ApiError(f"{tag}: release asset referenced by slim manifest is missing: {name}")
        local_path = staged_paths[name]
        if local_path.stat().st_size != int(asset["size"]):
            raise ApiError(
                f"downloaded asset size mismatch for {name}: expected {int(asset['size'])}, got {local_path.stat().st_size}"
            )
        desired_assets.append(
            DesiredAsset(
                name=name,
                size=int(asset["size"]),
                local_path=local_path,
            )
        )
    desired_assets.append(
        DesiredAsset(
            name=MANIFEST_ASSET_NAME,
            size=manifest_path.stat().st_size,
            local_path=manifest_path,
        )
    )
    desired_assets.append(
        DesiredAsset(
            name=DIGEST_ASSET_NAME,
            size=digest_path.stat().st_size,
            local_path=digest_path,
        )
    )
    return desired_assets


def resolve_download_root(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.download_dir:
        root = Path(args.download_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root, False
    root = Path(tempfile.mkdtemp(prefix=f"gh-release-{args.tag}-")).resolve()
    return root, not args.keep_downloads


def main() -> int:
    args = parse_args()
    github_owner, github_repo = split_repo(args.github_repo, "GITHUB_REPOSITORY")
    gitee_owner, gitee_repo = split_repo(args.gitee_repo, "GITEE_REPO")
    gitee_token = require_env("GITEE_TOKEN")

    gh_repo = f"{github_owner}/{github_repo}"
    download_root, cleanup_downloads = resolve_download_root(args)
    if args.download_dir:
        log(f"[info] download dir: {download_root}")
    elif cleanup_downloads:
        log(f"[info] temporary download dir: {download_root}")
    else:
        log(f"[info] keeping temporary download dir: {download_root}")

    try:
        release = resolve_release(gh_repo, args.tag)
        desired_assets = prepare_local_gitee_release_assets(gh_repo, release, download_root)
        github = GitHubClient(github_owner, github_repo, token=None)
        gitee = GiteeClient(gitee_owner, gitee_repo, gitee_token)
        gitee_releases_by_tag = {
            item["tag_name"]: item for item in gitee.list_releases() if item.get("tag_name")
        }
        status = sync_release(
            github,
            gitee,
            release,
            dry_run=args.dry_run,
            manifest_assets=desired_assets,
            gitee_releases_by_tag=gitee_releases_by_tag,
        )
        deleted_releases = purge_obsolete_gitee_v2_releases(
            gitee,
            keep_tag=str(release["tag_name"]),
            gitee_releases_by_tag=gitee_releases_by_tag,
            dry_run=args.dry_run,
        )
        log(f"[summary] {json.dumps({'status': status, 'deleted_releases': deleted_releases}, ensure_ascii=False)}")
        return 0
    finally:
        if cleanup_downloads:
            shutil.rmtree(download_root, ignore_errors=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ApiError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
