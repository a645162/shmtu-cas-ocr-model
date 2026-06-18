#!/usr/bin/env python3
"""Synchronize GitHub releases to Gitee for shmtu-cas-ocr-model."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


GITHUB_API = "https://api.github.com"
GITEE_API = "https://gitee.com/api/v5"


class ApiError(RuntimeError):
    """Raised when an API request fails."""


@dataclass(frozen=True)
class DesiredAsset:
    name: str
    size: int
    download_url: str | None = None
    local_path: Path | None = None


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync GitHub release metadata/assets to Gitee releases."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tag", help="Sync a single GitHub release tag.")
    mode.add_argument(
        "--all-releases",
        action="store_true",
        help="Sync every non-draft GitHub release.",
    )
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
        "--asset-manifest",
        help="TSV file with '<local path>\\t<release asset name>' rows.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print changes without mutating Gitee releases.",
    )
    return parser.parse_args()


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"missing required environment variable: {name}")
    return value


def split_repo(repo: str | None, env_name: str) -> tuple[str, str]:
    if not repo or "/" not in repo:
        raise SystemExit(f"{env_name} must be set to owner/repo")
    owner, name = repo.split("/", 1)
    if not owner or not name:
        raise SystemExit(f"{env_name} must be set to owner/repo")
    return owner, name


def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)


def _decode_error_payload(payload: bytes | None) -> str:
    if not payload:
        return ""
    text = payload.decode("utf-8", errors="replace").strip()
    if not text:
        return ""
    try:
        body = json.loads(text)
    except json.JSONDecodeError:
        return text[:500]
    if isinstance(body, dict):
        for key in ("message", "error_description", "error", "errors"):
            if key in body:
                return f"{body[key]}"
    return text[:500]


def request(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
) -> tuple[Any, int]:
    req = urllib.request.Request(url, headers=headers or {}, method=method, data=data)
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            raw = response.read()
            content_type = response.headers.get("Content-Type", "")
            if raw and "application/json" in content_type:
                return json.loads(raw.decode("utf-8")), response.status
            if raw:
                return raw, response.status
            return None, response.status
    except urllib.error.HTTPError as exc:
        payload = exc.read()
        detail = _decode_error_payload(payload)
        suffix = f": {detail}" if detail else ""
        raise ApiError(f"{method} {url} failed with {exc.code}{suffix}") from exc


def encode_form(data: dict[str, Any]) -> bytes:
    items: list[tuple[str, str]] = []
    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, bool):
            items.append((key, "true" if value else "false"))
        else:
            items.append((key, str(value)))
    return urllib.parse.urlencode(items).encode("utf-8")


def encode_multipart(fields: dict[str, str], file_field: str, file_path: Path) -> tuple[bytes, str]:
    boundary = "----CodexGiteeSyncBoundary7MA4YWxkTrZu0gW"
    content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(f"--{boundary}\r\n".encode("utf-8"))
        parts.append(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode("utf-8")
        )
    parts.append(f"--{boundary}\r\n".encode("utf-8"))
    parts.append(
        (
            f'Content-Disposition: form-data; name="{file_field}"; filename="{file_path.name}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8")
    )
    parts.append(file_path.read_bytes())
    parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


class GitHubClient:
    def __init__(self, owner: str, repo: str, token: str | None) -> None:
        self.owner = owner
        self.repo = repo
        self.token = token

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "shmtu-cas-ocr-model-gitee-sync",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _url(self, path: str, **params: Any) -> str:
        query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        base = f"{GITHUB_API}/repos/{self.owner}/{self.repo}{path}"
        return f"{base}?{query}" if query else base

    def list_releases(self) -> list[dict[str, Any]]:
        releases: list[dict[str, Any]] = []
        page = 1
        while True:
            payload, _ = request(
                "GET",
                self._url("/releases", per_page=100, page=page),
                headers=self._headers(),
            )
            assert isinstance(payload, list)
            if not payload:
                return releases
            releases.extend(payload)
            if len(payload) < 100:
                return releases
            page += 1

    def get_release_by_tag(self, tag: str) -> dict[str, Any]:
        payload, _ = request(
            "GET",
            self._url(f"/releases/tags/{urllib.parse.quote(tag, safe='')}"),
            headers=self._headers(),
        )
        assert isinstance(payload, dict)
        return payload

    def download_asset(self, asset: DesiredAsset, dest_dir: Path) -> Path:
        if asset.local_path is not None:
            return asset.local_path
        if not asset.download_url:
            raise ApiError(f"asset {asset.name} is missing a download URL")
        dest = dest_dir / asset.name
        headers = {"User-Agent": "shmtu-cas-ocr-model-gitee-sync"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request_obj = urllib.request.Request(asset.download_url, headers=headers, method="GET")
        with urllib.request.urlopen(request_obj, timeout=300) as response:
            dest.write_bytes(response.read())
        actual_size = dest.stat().st_size
        if actual_size != asset.size:
            raise ApiError(
                f"downloaded asset size mismatch for {asset.name}: expected {asset.size}, got {actual_size}"
            )
        return dest


class GiteeClient:
    def __init__(self, owner: str, repo: str, token: str) -> None:
        self.owner = owner
        self.repo = repo
        self.token = token

    def _url(self, path: str, **params: Any) -> str:
        query = {"access_token": self.token}
        query.update({k: v for k, v in params.items() if v is not None})
        encoded = urllib.parse.urlencode(query)
        return f"{GITEE_API}/repos/{self.owner}/{self.repo}{path}?{encoded}"

    def list_releases(self) -> list[dict[str, Any]]:
        releases: list[dict[str, Any]] = []
        page = 1
        while True:
            payload, _ = request(
                "GET",
                self._url("/releases", per_page=100, page=page),
                headers={"Accept": "application/json", "User-Agent": "shmtu-cas-ocr-model-gitee-sync"},
            )
            assert isinstance(payload, list)
            if not payload:
                return releases
            releases.extend(payload)
            if len(payload) < 100:
                return releases
            page += 1

    def find_release_by_tag(self, tag: str) -> dict[str, Any] | None:
        for release in self.list_releases():
            if release.get("tag_name") == tag:
                return release
        return None

    def create_release(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = encode_form(payload)
        body, _ = request(
            "POST",
            self._url("/releases"),
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "User-Agent": "shmtu-cas-ocr-model-gitee-sync",
            },
            data=data,
        )
        assert isinstance(body, dict)
        return body

    def update_release(self, release_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        errors: list[str] = []
        for method in ("PATCH", "PUT"):
            try:
                body, _ = request(
                    method,
                    self._url(f"/releases/{release_id}"),
                    headers={
                        "Content-Type": "application/x-www-form-urlencoded",
                        "Accept": "application/json",
                        "User-Agent": "shmtu-cas-ocr-model-gitee-sync",
                    },
                    data=encode_form(payload),
                )
                assert isinstance(body, dict)
                return body
            except ApiError as exc:
                errors.append(str(exc))
        raise ApiError(" / ".join(errors))

    def list_attach_files(self, release_id: int) -> list[dict[str, Any]]:
        body, _ = request(
            "GET",
            self._url(f"/releases/{release_id}/attach_files"),
            headers={"Accept": "application/json", "User-Agent": "shmtu-cas-ocr-model-gitee-sync"},
        )
        assert isinstance(body, list)
        return body

    def delete_attach_file(self, release_id: int, attach_file_id: int) -> None:
        request(
            "DELETE",
            self._url(f"/releases/{release_id}/attach_files/{attach_file_id}"),
            headers={"Accept": "application/json", "User-Agent": "shmtu-cas-ocr-model-gitee-sync"},
        )

    def upload_attach_file(self, release_id: int, file_path: Path) -> dict[str, Any]:
        data, content_type = encode_multipart({}, "file", file_path)
        body, _ = request(
            "POST",
            self._url(f"/releases/{release_id}/attach_files"),
            headers={
                "Content-Type": content_type,
                "Accept": "application/json",
                "User-Agent": "shmtu-cas-ocr-model-gitee-sync",
            },
            data=data,
        )
        assert isinstance(body, dict)
        return body


def load_desired_assets_from_manifest(manifest_path: Path) -> list[DesiredAsset]:
    assets: list[DesiredAsset] = []
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        file_path_str, release_name = line.split("\t", 1)
        file_path = Path(file_path_str).resolve()
        if not file_path.is_file():
            raise SystemExit(f"asset listed in manifest not found: {file_path}")
        assets.append(
            DesiredAsset(
                name=release_name,
                size=file_path.stat().st_size,
                local_path=file_path,
            )
        )
    return assets


def load_desired_assets_from_github(release: dict[str, Any]) -> list[DesiredAsset]:
    assets: list[DesiredAsset] = []
    for asset in release.get("assets", []):
        assets.append(
            DesiredAsset(
                name=asset["name"],
                size=int(asset["size"]),
                download_url=asset["browser_download_url"],
            )
        )
    return assets


def build_release_payload(release: dict[str, Any]) -> dict[str, Any]:
    return {
        "tag_name": release["tag_name"],
        "target_commitish": release.get("target_commitish") or "",
        "prerelease": bool(release.get("prerelease")),
        "name": release.get("name") or release["tag_name"],
        "body": release.get("body") or "",
    }


def same_release_metadata(github_release: dict[str, Any], gitee_release: dict[str, Any]) -> bool:
    github_name = github_release.get("name") or github_release["tag_name"]
    github_body = github_release.get("body") or ""
    github_commitish = github_release.get("target_commitish") or ""
    return (
        (gitee_release.get("name") or "") == github_name
        and (gitee_release.get("body") or "") == github_body
        and bool(gitee_release.get("prerelease")) == bool(github_release.get("prerelease"))
        and (gitee_release.get("target_commitish") or "") == github_commitish
    )


def same_asset_set(desired_assets: list[DesiredAsset], gitee_assets: list[dict[str, Any]]) -> bool:
    desired = sorted((asset.name, asset.size) for asset in desired_assets)
    actual = sorted((asset["name"], int(asset["size"])) for asset in gitee_assets)
    return desired == actual


def sync_release(
    github: GitHubClient,
    gitee: GiteeClient,
    github_release: dict[str, Any],
    *,
    dry_run: bool,
    manifest_assets: list[DesiredAsset] | None = None,
    gitee_releases_by_tag: dict[str, dict[str, Any]] | None = None,
) -> str:
    tag = github_release["tag_name"]
    if github_release.get("draft"):
        log(f"[skip] {tag}: draft GitHub release is not mirrored")
        return "skipped"

    desired_assets = manifest_assets if manifest_assets is not None else load_desired_assets_from_github(github_release)
    if gitee_releases_by_tag is None:
        gitee_release = gitee.find_release_by_tag(tag)
    else:
        gitee_release = gitee_releases_by_tag.get(tag)
    payload = build_release_payload(github_release)

    if gitee_release is None:
        if dry_run:
            log(f"[plan] create Gitee release {tag}")
            if desired_assets:
                for asset in desired_assets:
                    log(f"[plan] upload {tag}: {asset.name} ({asset.size} bytes)")
            return "planned"
        log(f"[sync] create Gitee release {tag}")
        gitee_release = gitee.create_release(payload)
        if gitee_releases_by_tag is not None:
            gitee_releases_by_tag[tag] = gitee_release
    elif same_release_metadata(github_release, gitee_release):
        log(f"[ok] metadata up-to-date for {tag}")
    else:
        if dry_run:
            log(f"[plan] update Gitee release metadata for {tag}")
        else:
            log(f"[sync] update Gitee release metadata for {tag}")
            gitee_release = gitee.update_release(int(gitee_release["id"]), payload)
            if gitee_releases_by_tag is not None:
                gitee_releases_by_tag[tag] = gitee_release

    assert gitee_release is not None
    release_id = int(gitee_release["id"])
    gitee_assets = gitee.list_attach_files(release_id)
    if same_asset_set(desired_assets, gitee_assets):
        log(f"[ok] assets up-to-date for {tag}")
        return "unchanged"

    if dry_run:
        if gitee_assets:
            for asset in gitee_assets:
                log(f"[plan] delete {tag}: {asset['name']} ({asset['size']} bytes)")
        if desired_assets:
            for asset in desired_assets:
                log(f"[plan] upload {tag}: {asset.name} ({asset.size} bytes)")
        return "planned"

    for asset in gitee_assets:
        log(f"[sync] delete {tag}: {asset['name']}")
        gitee.delete_attach_file(release_id, int(asset["id"]))

    if not desired_assets:
        log(f"[ok] {tag}: release has no uploaded assets")
        return "updated"

    with tempfile.TemporaryDirectory(prefix=f"gitee-sync-{tag}-") as tmp_dir:
        tmp_root = Path(tmp_dir)
        for asset in desired_assets:
            local_path = github.download_asset(asset, tmp_root)
            log(f"[sync] upload {tag}: {asset.name}")
            gitee.upload_attach_file(release_id, local_path)

    return "updated"


def main() -> int:
    args = parse_args()
    github_owner, github_repo = split_repo(args.github_repo, "GITHUB_REPOSITORY")
    gitee_owner, gitee_repo = split_repo(args.gitee_repo, "GITEE_REPO")
    gh_token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    gitee_token = require_env("GITEE_TOKEN")

    github = GitHubClient(github_owner, github_repo, gh_token)
    gitee = GiteeClient(gitee_owner, gitee_repo, gitee_token)
    gitee_releases_by_tag = {
        release["tag_name"]: release for release in gitee.list_releases() if release.get("tag_name")
    }

    manifest_assets: list[DesiredAsset] | None = None
    if args.asset_manifest:
        manifest_path = Path(args.asset_manifest).resolve()
        if not manifest_path.is_file():
            raise SystemExit(f"asset manifest not found: {manifest_path}")
        manifest_assets = load_desired_assets_from_manifest(manifest_path)

    if args.all_releases and manifest_assets is not None:
        raise SystemExit("--asset-manifest only supports single-tag sync")

    statuses: list[str] = []
    if args.tag:
        github_release = github.get_release_by_tag(args.tag)
        statuses.append(
            sync_release(
                github,
                gitee,
                github_release,
                dry_run=args.dry_run,
                manifest_assets=manifest_assets,
                gitee_releases_by_tag=gitee_releases_by_tag,
            )
        )
    else:
        for release in github.list_releases():
            if release.get("draft"):
                continue
            statuses.append(
                sync_release(
                    github,
                    gitee,
                    release,
                    dry_run=args.dry_run,
                    gitee_releases_by_tag=gitee_releases_by_tag,
                )
            )

    summary = {
        "planned": statuses.count("planned"),
        "updated": statuses.count("updated"),
        "unchanged": statuses.count("unchanged"),
        "skipped": statuses.count("skipped"),
    }
    log(f"[summary] {json_dumps(summary)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ApiError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
