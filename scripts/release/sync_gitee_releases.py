#!/usr/bin/env python3
"""Synchronize GitHub releases to Gitee for shmtu-cas-ocr-model."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import mimetypes
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar


GITHUB_API = "https://api.github.com"
GITEE_API = "https://gitee.com/api/v5"
MANIFEST_ASSET_NAME = "model-assets.json"
DIGEST_ASSET_NAME = "SHA256SUMS.txt"
SEMVER_TAG_PATTERN = re.compile(r"^v(\d+)\.(\d+)\.(\d+)(?:[-+].*)?$")
GITEE_V2_ALLOWED_BACKBONES = {
    "mobilenet_v3_small",
    "mobilenetv4_conv_small",
}


class ApiError(RuntimeError):
    """Raised when an API request fails."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class DesiredAsset:
    name: str
    size: int
    download_url: str | None = None
    local_path: Path | None = None


@dataclass(frozen=True)
class AssetSyncPlan:
    keep: list[dict[str, Any]]
    delete: list[dict[str, Any]]
    upload: list[DesiredAsset]


RETRYABLE_HTTP_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
RETRYABLE_DOWNLOAD_EXCEPTIONS = (
    urllib.error.URLError,
    TimeoutError,
    socket.timeout,
    http.client.IncompleteRead,
)
DEFAULT_UPLOAD_MAX_ATTEMPTS = 3
DEFAULT_UPLOAD_RETRY_DELAY_SECONDS = 2.0
DEFAULT_DOWNLOAD_MAX_ATTEMPTS = 3
DEFAULT_DOWNLOAD_RETRY_DELAY_SECONDS = 2.0
DEFAULT_API_READ_MAX_ATTEMPTS = 5
DEFAULT_API_READ_RETRY_DELAY_SECONDS = 2.0
T = TypeVar("T")


def log(message: str) -> None:
    print(message, flush=True)


def format_bytes(size: int) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{size} B"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_download_cache_dir(repo_slug: str, tag: str) -> Path:
    root = os.environ.get("SYNC_GITHUB_CACHE_DIR")
    if root:
        base = Path(root).expanduser().resolve()
    else:
        base = Path(tempfile.gettempdir()).resolve() / "shmtu-cas-ocr-model-release-cache"
    safe_repo = repo_slug.replace("/", "__")
    return base / safe_repo / tag


def run_with_heartbeat(label: str, action: Callable[[], T], *, interval_seconds: float = 15.0) -> T:
    stop_event = threading.Event()

    def heartbeat() -> None:
        start = time.monotonic()
        while not stop_event.wait(interval_seconds):
            elapsed = time.monotonic() - start
            log(f"[wait] {label} still running ({elapsed:.0f}s elapsed)")

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        return action()
    finally:
        stop_event.set()
        thread.join(timeout=0.1)


def retry_read_operation(label: str, action: Callable[[], T]) -> T:
    max_attempts = max(
        1,
        int(os.environ.get("API_READ_MAX_ATTEMPTS", str(DEFAULT_API_READ_MAX_ATTEMPTS))),
    )
    base_delay_seconds = float(
        os.environ.get(
            "API_READ_RETRY_DELAY_SECONDS",
            str(DEFAULT_API_READ_RETRY_DELAY_SECONDS),
        )
    )

    for attempt in range(1, max_attempts + 1):
        try:
            return action()
        except ApiError as exc:
            if not exc.retryable or attempt >= max_attempts:
                raise
            delay_seconds = min(30.0, base_delay_seconds * (2 ** (attempt - 1)))
            log(
                f"[retry] {label} attempt {attempt}/{max_attempts} failed; "
                f"retrying in {delay_seconds:.1f}s"
            )
            time.sleep(delay_seconds)

    raise ApiError(f"{label} failed unexpectedly")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync GitHub release metadata/assets to Gitee releases.",
        epilog=(
            "Examples:\n"
            "  sync_gitee_releases.py                 # latest published release\n"
            "  sync_gitee_releases.py --tag v2.0.5\n"
            "  sync_gitee_releases.py --all-releases --dry-run\n"
            "  RELEASE_TAG=v2.0.5 sync_gitee_releases.py"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument("--tag", help="Sync a single GitHub release tag.")
    mode.add_argument(
        "--latest",
        action="store_true",
        help="Sync the latest published GitHub release (default when no mode is specified).",
    )
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


def infer_tag_from_env() -> str | None:
    candidates = (
        "RELEASE_TAG",
        "GITHUB_REF_NAME",
        "CI_COMMIT_TAG",
    )
    for name in candidates:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    ref = (os.environ.get("GITHUB_REF") or "").strip()
    prefix = "refs/tags/"
    if ref.startswith(prefix):
        return ref[len(prefix) :]
    return None


def resolve_sync_mode(args: argparse.Namespace) -> tuple[str, str | None]:
    if args.tag:
        return "tag", args.tag
    if args.latest:
        return "latest", None
    if args.all_releases:
        return "all", None

    inferred_tag = infer_tag_from_env()
    if inferred_tag:
        log(f"[info] inferred release tag from environment: {inferred_tag}")
        return "tag", inferred_tag

    log("[info] no explicit mode provided; defaulting to latest published GitHub release")
    return "latest", None


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"missing required environment variable: {name}")
    return value


def _run_git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def infer_repo_from_git_remote(remote_name: str) -> str | None:
    remote_url = _run_git(["remote", "get-url", remote_name])
    if not remote_url:
        return None

    normalized = remote_url.strip()
    if normalized.endswith(".git"):
        normalized = normalized[:-4]

    if normalized.startswith("git@"):
        _, _, path = normalized.partition(":")
        parts = [part for part in path.split("/") if part]
    else:
        parsed = urllib.parse.urlparse(normalized)
        parts = [part for part in parsed.path.split("/") if part]

    if len(parts) < 2:
        return None
    return f"{parts[-2]}/{parts[-1]}"


def split_repo(repo: str | None, env_name: str, *, remote_name: str | None = None) -> tuple[str, str]:
    if not repo and remote_name:
        repo = infer_repo_from_git_remote(remote_name)
        if repo:
            log(f"[info] inferred {env_name} from git remote '{remote_name}': {repo}")
    if not repo or "/" not in repo:
        hint = f" or discoverable from git remote '{remote_name}'" if remote_name else ""
        raise SystemExit(f"{env_name} must be set to owner/repo{hint}")
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
        raise ApiError(
            f"{method} {url} failed with {exc.code}{suffix}",
            retryable=exc.code in RETRYABLE_HTTP_STATUS_CODES,
        ) from exc
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        raise ApiError(f"{method} {url} failed: {exc}", retryable=True) from exc


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
            payload, _ = retry_read_operation(
                f"GitHub releases page {page}",
                lambda page=page: request(
                    "GET",
                    self._url("/releases", per_page=100, page=page),
                    headers=self._headers(),
                ),
            )
            assert isinstance(payload, list)
            if not payload:
                return releases
            releases.extend(payload)
            if len(payload) < 100:
                return releases
            page += 1

    def get_release_by_tag(self, tag: str) -> dict[str, Any]:
        payload, _ = retry_read_operation(
            f"GitHub release tag {tag}",
            lambda: request(
                "GET",
                self._url(f"/releases/tags/{urllib.parse.quote(tag, safe='')}"),
                headers=self._headers(),
            ),
        )
        assert isinstance(payload, dict)
        return payload

    def download_asset(self, asset: DesiredAsset, dest_dir: Path) -> Path:
        if asset.local_path is not None:
            return asset.local_path
        if not asset.download_url:
            raise ApiError(f"asset {asset.name} is missing a download URL")
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / asset.name
        part = dest_dir / f"{asset.name}.part"
        base_headers = {
            "User-Agent": "shmtu-cas-ocr-model-gitee-sync",
            "Accept-Encoding": "identity",
        }
        if self.token:
            base_headers["Authorization"] = f"Bearer {self.token}"
        max_attempts = max(
            1,
            int(os.environ.get("GITHUB_DOWNLOAD_MAX_ATTEMPTS", str(DEFAULT_DOWNLOAD_MAX_ATTEMPTS))),
        )
        base_delay_seconds = float(
            os.environ.get(
                "GITHUB_DOWNLOAD_RETRY_DELAY_SECONDS",
                str(DEFAULT_DOWNLOAD_RETRY_DELAY_SECONDS),
            )
        )

        if dest.exists():
            final_size = dest.stat().st_size
            if final_size == asset.size:
                return dest
            if final_size > asset.size:
                dest.unlink()
            else:
                if part.exists():
                    part.unlink()
                dest.replace(part)

        if part.exists() and part.stat().st_size > asset.size:
            part.unlink()

        for attempt in range(1, max_attempts + 1):
            try:
                existing_size = part.stat().st_size if part.exists() else 0
                headers = dict(base_headers)
                if 0 < existing_size < asset.size:
                    headers["Range"] = f"bytes={existing_size}-"
                    log(
                        f"[resume] {asset.name}: resume from {format_bytes(existing_size)} "
                        f"/ {format_bytes(asset.size)}"
                    )

                request_obj = urllib.request.Request(asset.download_url, headers=headers, method="GET")
                with urllib.request.urlopen(request_obj, timeout=300) as response:
                    response_status = getattr(response, "status", response.getcode())
                    if existing_size > 0 and response_status == 206:
                        mode = "ab"
                    else:
                        if existing_size > 0 and response_status == 200:
                            log(f"[resume] {asset.name}: server ignored range, restarting full download")
                        mode = "wb"
                        existing_size = 0

                    with part.open(mode) as handle:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            handle.write(chunk)

                actual_size = part.stat().st_size
                if actual_size != asset.size:
                    if actual_size > asset.size:
                        part.unlink()
                        raise ApiError(
                            f"downloaded asset size mismatch for {asset.name}: "
                            f"expected {asset.size}, got {actual_size}",
                            retryable=True,
                        )
                    log(
                        f"[retry] download {asset.name} incomplete after attempt {attempt}/{max_attempts}: "
                        f"{format_bytes(actual_size)} / {format_bytes(asset.size)}"
                    )
                    if attempt >= max_attempts:
                        raise ApiError(
                            f"downloaded asset size mismatch for {asset.name}: "
                            f"expected {asset.size}, got {actual_size}",
                            retryable=True,
                        )
                    delay_seconds = min(30.0, base_delay_seconds * (2 ** (attempt - 1)))
                    time.sleep(delay_seconds)
                    continue
                part.replace(dest)
                return dest
            except RETRYABLE_DOWNLOAD_EXCEPTIONS as exc:
                current_size = part.stat().st_size if part.exists() else 0
                if attempt >= max_attempts:
                    raise ApiError(
                        f"download {asset.name} failed after {attempt} attempts: {exc}",
                        retryable=True,
                    ) from exc
                delay_seconds = min(30.0, base_delay_seconds * (2 ** (attempt - 1)))
                log(
                    f"[retry] download {asset.name} attempt {attempt}/{max_attempts} failed; "
                    f"retrying in {delay_seconds:.1f}s "
                    f"(partial={format_bytes(current_size)}/{format_bytes(asset.size)})"
                )
                time.sleep(delay_seconds)
            except ApiError as exc:
                if part.exists() and part.stat().st_size > asset.size:
                    part.unlink()
                if dest.exists() and dest.stat().st_size > asset.size:
                    dest.unlink()
                if not exc.retryable or attempt >= max_attempts:
                    raise
                delay_seconds = min(30.0, base_delay_seconds * (2 ** (attempt - 1)))
                log(
                    f"[retry] download {asset.name} attempt {attempt}/{max_attempts} failed; "
                    f"retrying in {delay_seconds:.1f}s"
                )
                time.sleep(delay_seconds)

        raise ApiError(f"download {asset.name} failed unexpectedly")


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
            payload, _ = retry_read_operation(
                f"Gitee releases page {page}",
                lambda page=page: request(
                    "GET",
                    self._url("/releases", per_page=100, page=page),
                    headers={"Accept": "application/json", "User-Agent": "shmtu-cas-ocr-model-gitee-sync"},
                ),
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
        body, _ = retry_read_operation(
            f"Gitee release {release_id} attach_files",
            lambda: request(
                "GET",
                self._url(f"/releases/{release_id}/attach_files"),
                headers={"Accept": "application/json", "User-Agent": "shmtu-cas-ocr-model-gitee-sync"},
            ),
        )
        assert isinstance(body, list)
        return body

    def delete_attach_file(self, release_id: int, attach_file_id: int) -> None:
        request(
            "DELETE",
            self._url(f"/releases/{release_id}/attach_files/{attach_file_id}"),
            headers={"Accept": "application/json", "User-Agent": "shmtu-cas-ocr-model-gitee-sync"},
        )

    def delete_release(self, release_id: int) -> None:
        request(
            "DELETE",
            self._url(f"/releases/{release_id}"),
            headers={"Accept": "application/json", "User-Agent": "shmtu-cas-ocr-model-gitee-sync"},
        )

    def _find_existing_attach_file(
        self, release_id: int, file_name: str, expected_size: int
    ) -> dict[str, Any] | None:
        for asset in self.list_attach_files(release_id):
            if str(asset.get("name")) != file_name:
                continue
            if int(asset.get("size", -1)) != expected_size:
                continue
            return asset
        return None

    def upload_attach_file(self, release_id: int, file_path: Path) -> dict[str, Any]:
        data, content_type = encode_multipart({}, "file", file_path)
        max_attempts = max(
            1,
            int(os.environ.get("GITEE_UPLOAD_MAX_ATTEMPTS", str(DEFAULT_UPLOAD_MAX_ATTEMPTS))),
        )
        base_delay_seconds = float(
            os.environ.get(
                "GITEE_UPLOAD_RETRY_DELAY_SECONDS",
                str(DEFAULT_UPLOAD_RETRY_DELAY_SECONDS),
            )
        )
        expected_size = file_path.stat().st_size
        last_error: ApiError | None = None

        for attempt in range(1, max_attempts + 1):
            try:
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
            except ApiError as exc:
                last_error = exc
                try:
                    existing = self._find_existing_attach_file(release_id, file_path.name, expected_size)
                except ApiError:
                    existing = None
                if existing is not None:
                    log(f"[ok] upload already present on Gitee after failure: {file_path.name}")
                    return existing
                if not exc.retryable or attempt >= max_attempts:
                    raise
                delay_seconds = min(30.0, base_delay_seconds * (2 ** (attempt - 1)))
                log(
                    f"[retry] upload {file_path.name} attempt {attempt}/{max_attempts} failed; "
                    f"retrying in {delay_seconds:.1f}s"
                )
                time.sleep(delay_seconds)

        assert last_error is not None
        raise last_error


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


def parse_semver_tag(tag: str) -> tuple[int, int, int] | None:
    match = SEMVER_TAG_PATTERN.match(tag.strip())
    if match is None:
        return None
    return tuple(int(part) for part in match.groups())


def is_v2_tag(tag: str) -> bool:
    version = parse_semver_tag(tag)
    return version is not None and version[0] == 2


def select_latest_release(releases: list[dict[str, Any]]) -> dict[str, Any]:
    published = [
        release
        for release in releases
        if not release.get("draft") and not release.get("prerelease")
    ]
    if not published:
        published = [release for release in releases if not release.get("draft")]
    if not published:
        raise ApiError("no published GitHub releases found")

    semver_releases = [
        release
        for release in published
        if parse_semver_tag(str(release.get("tag_name") or "")) is not None
    ]
    if semver_releases:
        return max(
            semver_releases,
            key=lambda release: (
                parse_semver_tag(str(release.get("tag_name") or "")) or (-1, -1, -1),
                str(release.get("published_at") or release.get("created_at") or ""),
            ),
        )
    return published[0]


def parse_sha256sums(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise ApiError(f"invalid SHA256SUMS line in {path.name!r}: {raw_line!r}")
        digest, rel_path = parts
        rel_path = rel_path.strip().lstrip("*").strip()
        name = Path(rel_path).name
        if not name:
            raise ApiError(f"invalid SHA256SUMS entry path in {path.name!r}: {raw_line!r}")
        entries[name] = digest.lower()
    return entries


def parse_manifest_expected_digests(path: Path) -> dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected: dict[str, str] = {}

    for artifact in raw.get("artifacts", []):
        for file_info in artifact.get("files", []):
            release_name = file_info.get("release_asset_name") or Path(file_info["path"]).name
            digest = str(file_info["sha256"]).lower()
            expected[str(release_name)] = digest

    for digest_info in raw.get("digests", []):
        release_name = digest_info.get("release_asset_name") or Path(digest_info["path"]).name
        digest = str(digest_info["sha256"]).lower()
        expected[str(release_name)] = digest

    return expected


def _normalize_artifact_files(raw_files: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_files, list):
        return []
    files: list[dict[str, Any]] = []
    for item in raw_files:
        if not isinstance(item, dict):
            continue
        files.append(dict(item))
    return files


def _normalize_artifact(raw_artifact: Any) -> dict[str, Any] | None:
    if not isinstance(raw_artifact, dict):
        return None
    artifact = dict(raw_artifact)
    artifact["files"] = _normalize_artifact_files(raw_artifact.get("files"))
    return artifact


def iter_manifest_artifacts(raw_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    raw_artifacts = raw_manifest.get("artifacts")
    if isinstance(raw_artifacts, list):
        artifacts: list[dict[str, Any]] = []
        for raw_artifact in raw_artifacts:
            artifact = _normalize_artifact(raw_artifact)
            if artifact is not None:
                artifacts.append(artifact)
        if artifacts:
            return artifacts

    artifacts = []
    raw_models = raw_manifest.get("models")
    if not isinstance(raw_models, list):
        return artifacts
    for raw_model in raw_models:
        if not isinstance(raw_model, dict):
            continue
        grouped = raw_model.get("artifacts")
        if not isinstance(grouped, dict):
            continue
        for engine, precision_map in grouped.items():
            if not isinstance(engine, str) or not isinstance(precision_map, dict):
                continue
            for precision, raw_artifact in precision_map.items():
                artifact = _normalize_artifact(raw_artifact)
                if artifact is None:
                    continue
                artifact.setdefault("engine", engine)
                if isinstance(precision, str):
                    artifact.setdefault("precision", precision)
                for key, value in raw_model.items():
                    if key == "artifacts":
                        continue
                    artifact.setdefault(key, value)
                artifacts.append(artifact)
    return artifacts


def build_release_manifest(
    *,
    model_entries: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    digests: list[dict[str, Any]],
    generated_at_utc: str | None,
    schema_version: int,
) -> dict[str, Any]:
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
            "files": _normalize_artifact_files(artifact.get("files")),
        }

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
        "generated_at_utc": generated_at_utc or raw_manifest_timestamp_now(),
        "model_count": len(deduped_models),
        "modellist": [entry["asset_stem"] for entry in deduped_models],
        "models": deduped_models,
        "artifacts": artifacts,
        "digests": digests,
    }


def raw_manifest_timestamp_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def resolve_release_asset_name(file_info: dict[str, Any]) -> str:
    return str(file_info.get("release_asset_name") or Path(str(file_info["path"])).name)


def is_allowed_gitee_v2_model(entry: dict[str, Any]) -> bool:
    backbone = str(entry.get("backbone") or "").strip()
    asset_stem = str(entry.get("asset_stem") or "").strip()
    stem_backbone = asset_stem.split(".", 1)[0] if asset_stem else ""
    return backbone in GITEE_V2_ALLOWED_BACKBONES or stem_backbone in GITEE_V2_ALLOWED_BACKBONES


def plan_gitee_v2_slim_assets(raw_manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    raw_models = raw_manifest.get("models")
    if not isinstance(raw_models, list):
        raise ApiError(f"{MANIFEST_ASSET_NAME} is missing models[]")

    model_entries: list[dict[str, Any]] = []
    allowed_stems: set[str] = set()
    for raw_model in raw_models:
        if not isinstance(raw_model, dict):
            continue
        if not is_allowed_gitee_v2_model(raw_model):
            continue
        entry = dict(raw_model)
        entry.pop("artifacts", None)
        asset_stem = str(entry.get("asset_stem") or "").strip()
        if not asset_stem or asset_stem in allowed_stems:
            continue
        allowed_stems.add(asset_stem)
        model_entries.append(entry)

    if not model_entries:
        raise ApiError("no allowed Gitee v2 models were found in model-assets.json")

    artifacts: list[dict[str, Any]] = []
    asset_names: list[str] = []
    seen_names: set[str] = set()
    for raw_artifact in iter_manifest_artifacts(raw_manifest):
        asset_stem = str(raw_artifact.get("asset_stem") or "").strip()
        if asset_stem not in allowed_stems:
            continue
        artifact = dict(raw_artifact)
        files: list[dict[str, Any]] = []
        for raw_file in _normalize_artifact_files(raw_artifact.get("files")):
            release_name = resolve_release_asset_name(raw_file)
            updated_file = dict(raw_file)
            updated_file["release_asset_name"] = release_name
            files.append(updated_file)
            if release_name not in seen_names:
                seen_names.add(release_name)
                asset_names.append(release_name)
        artifact["files"] = files
        artifacts.append(artifact)

    if not asset_names:
        raise ApiError("no slim Gitee assets were resolved from model-assets.json")
    return model_entries, artifacts, asset_names


def write_gitee_v2_slim_bundle(
    raw_manifest: dict[str, Any],
    *,
    staged_asset_paths: dict[str, Path],
    output_dir: Path,
) -> tuple[Path, Path]:
    model_entries, artifacts, _ = plan_gitee_v2_slim_assets(raw_manifest)
    digest_records: list[tuple[str, str]] = []
    updated_artifacts: list[dict[str, Any]] = []

    for raw_artifact in artifacts:
        artifact = dict(raw_artifact)
        files: list[dict[str, Any]] = []
        for raw_file in _normalize_artifact_files(raw_artifact.get("files")):
            release_name = resolve_release_asset_name(raw_file)
            local_path = staged_asset_paths.get(release_name)
            if local_path is None or not local_path.is_file():
                raise ApiError(f"missing staged slim asset for manifest rebuild: {release_name}")
            digest = sha256_file(local_path)
            updated_file = dict(raw_file)
            updated_file["path"] = release_name
            updated_file["release_asset_name"] = release_name
            updated_file["sha256"] = digest
            files.append(updated_file)
            digest_records.append((release_name, digest))
        artifact["files"] = files
        updated_artifacts.append(artifact)

    digest_path = output_dir / DIGEST_ASSET_NAME
    deduped_digest_records: dict[str, str] = {}
    for release_name, digest in digest_records:
        deduped_digest_records[release_name] = digest
    digest_lines = [
        f"{deduped_digest_records[release_name]}  {release_name}"
        for release_name in sorted(deduped_digest_records)
    ]
    digest_path.write_text("\n".join(digest_lines) + ("\n" if digest_lines else ""), encoding="utf-8")

    manifest = build_release_manifest(
        model_entries=model_entries,
        artifacts=updated_artifacts,
        digests=[
            {
                "engine": "release",
                "path": DIGEST_ASSET_NAME,
                "release_asset_name": DIGEST_ASSET_NAME,
                "sha256": sha256_file(digest_path),
            }
        ],
        generated_at_utc=str(raw_manifest.get("generated_at_utc") or ""),
        schema_version=int(raw_manifest.get("schema_version") or 2),
    )
    manifest_path = output_dir / MANIFEST_ASSET_NAME
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path, digest_path


def prepare_gitee_release_assets(github: GitHubClient, github_release: dict[str, Any]) -> list[DesiredAsset]:
    tag = str(github_release["tag_name"])
    if not is_v2_tag(tag):
        return load_desired_assets_from_github(github_release)

    log(f"[filter] {tag}: building slim Gitee release bundle")
    release_assets = {
        str(asset["name"]): asset
        for asset in github_release.get("assets", [])
        if asset.get("name")
    }
    source_manifest_asset = release_assets.get(MANIFEST_ASSET_NAME)
    if source_manifest_asset is None:
        raise ApiError(f"{tag}: missing {MANIFEST_ASSET_NAME} in GitHub release assets")

    cache_root = get_download_cache_dir(f"{github.owner}/{github.repo}", tag)
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path = github.download_asset(
        DesiredAsset(
            name=MANIFEST_ASSET_NAME,
            size=int(source_manifest_asset["size"]),
            download_url=str(source_manifest_asset["browser_download_url"]),
        ),
        cache_root,
    )
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _, _, selected_asset_names = plan_gitee_v2_slim_assets(raw_manifest)
    log(
        f"[filter] {tag}: keeping {len(selected_asset_names)} asset(s) "
        f"for backbones={sorted(GITEE_V2_ALLOWED_BACKBONES)}"
    )

    staged_asset_paths: dict[str, Path] = {}
    desired_assets: list[DesiredAsset] = []
    for name in selected_asset_names:
        asset = release_assets.get(name)
        if asset is None:
            raise ApiError(f"{tag}: release asset referenced by slim manifest is missing: {name}")
        desired = DesiredAsset(
            name=name,
            size=int(asset["size"]),
            download_url=str(asset["browser_download_url"]),
        )
        local_path = github.download_asset(desired, cache_root)
        staged_asset_paths[name] = local_path
        desired_assets.append(
            DesiredAsset(
                name=name,
                size=int(asset["size"]),
                local_path=local_path,
            )
        )

    manifest_path, digest_path = write_gitee_v2_slim_bundle(
        raw_manifest,
        staged_asset_paths=staged_asset_paths,
        output_dir=cache_root,
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


def collect_release_digest_errors(tag: str, staged_paths: dict[str, Path]) -> dict[str, str]:
    errors: dict[str, str] = {}
    skipped_missing: list[str] = []
    if not staged_paths:
        return errors
    manifest_path = staged_paths.get("model-assets.json")
    digest_path = staged_paths.get("SHA256SUMS.txt")
    manifest_expected: dict[str, str] = {}
    digest_expected: dict[str, str] = {}

    if manifest_path is not None:
        log(f"[check] {tag}: verifying model-assets.json digests")
        manifest_expected = parse_manifest_expected_digests(manifest_path)

    if digest_path is not None:
        log(f"[check] {tag}: verifying SHA256SUMS.txt entries")
        digest_expected = parse_sha256sums(digest_path)

    if manifest_expected and digest_expected:
        for name, digest in digest_expected.items():
            if name == "model-assets.json":
                continue
            manifest_digest = manifest_expected.get(name)
            if manifest_digest is not None and manifest_digest != digest:
                errors[name] = (
                    f"{tag}: digest mismatch between model-assets.json and SHA256SUMS.txt for {name}"
                )

    expected_names = set(manifest_expected) | set(digest_expected)
    verified = 0
    for name in sorted(expected_names):
        path = staged_paths.get(name)
        if path is None:
            skipped_missing.append(name)
            continue
        if name == "model-assets.json" and name in digest_expected:
            log(f"[check] {tag}: skip SHA256SUMS verification for model-assets.json (generator order dependent)")
            continue
        actual = sha256_file(path)
        expected = manifest_expected.get(name) or digest_expected.get(name)
        if expected is None:
            continue
        if actual.lower() != expected.lower():
            errors[name] = (
                f"{tag}: sha256 mismatch for {name}: expected {expected.lower()}, got {actual.lower()}"
            )
            continue
        verified += 1

    if verified:
        log(f"[check] {tag}: verified {verified} digest entr{'y' if verified == 1 else 'ies'}")
    if skipped_missing:
        log(
            f"[warn] {tag}: skip {len(skipped_missing)} digest entr"
            f"{'y' if len(skipped_missing) == 1 else 'ies'} not present in release assets"
        )
        for name in skipped_missing:
            log(f"[warn] {tag}: ignore non-release digest entry {name}")
    return errors


def verify_release_digests(tag: str, staged_paths: dict[str, Path]) -> None:
    errors = collect_release_digest_errors(tag, staged_paths)
    if errors:
        message = "\n".join(errors[name] for name in sorted(errors))
        raise ApiError(message)


def stage_upload_assets(
    github: GitHubClient,
    assets: list[DesiredAsset],
    cache_root: Path,
    *,
    tag: str,
) -> list[tuple[DesiredAsset, Path]]:
    staged: list[tuple[DesiredAsset, Path]] = []
    total = len(assets)
    for index, asset in enumerate(assets, start=1):
        size_text = format_bytes(asset.size)
        if asset.local_path is not None:
            log(f"[stage] {tag}: {index}/{total} use local {asset.name} ({size_text})")
            staged.append((asset, asset.local_path))
            continue

        log(f"[download] {tag}: {index}/{total} {asset.name} ({size_text})")
        local_path = run_with_heartbeat(
            f"download {tag} {asset.name}",
            lambda asset=asset: github.download_asset(asset, cache_root),
        )
        staged.append((asset, local_path))
    return staged


def purge_cached_assets(cache_root: Path, asset_names: list[str]) -> None:
    for name in asset_names:
        for path in (cache_root / name, cache_root / f"{name}.part"):
            if path.exists():
                path.unlink()


def build_release_payload(release: dict[str, Any]) -> dict[str, Any]:
    tag = release["tag_name"]
    body = (release.get("body") or "").strip()
    if not body:
        body = f"Synced from GitHub release {tag}."
    return {
        "tag_name": tag,
        "target_commitish": release.get("target_commitish") or "",
        "prerelease": bool(release.get("prerelease")),
        "name": release.get("name") or tag,
        "body": body,
    }


def same_release_metadata(github_release: dict[str, Any], gitee_release: dict[str, Any]) -> bool:
    github_name = github_release.get("name") or github_release["tag_name"]
    github_body = (github_release.get("body") or "").strip()
    if not github_body:
        github_body = f"Synced from GitHub release {github_release['tag_name']}."
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


def plan_asset_sync(desired_assets: list[DesiredAsset], gitee_assets: list[dict[str, Any]]) -> AssetSyncPlan:
    seen_names: set[str] = set()
    actual_by_name: dict[str, list[dict[str, Any]]] = {}
    for asset in gitee_assets:
        actual_by_name.setdefault(str(asset["name"]), []).append(asset)

    keep: list[dict[str, Any]] = []
    delete: list[dict[str, Any]] = []
    upload: list[DesiredAsset] = []

    for desired in desired_assets:
        if desired.name in seen_names:
            raise ApiError(f"duplicate desired asset name: {desired.name}")
        seen_names.add(desired.name)

        candidates = actual_by_name.pop(desired.name, [])
        if not candidates:
            upload.append(desired)
            continue

        matched = False
        for candidate in candidates:
            if not matched and int(candidate["size"]) == desired.size:
                keep.append(candidate)
                matched = True
            else:
                delete.append(candidate)
        if not matched:
            upload.append(desired)

    for extras in actual_by_name.values():
        delete.extend(extras)

    return AssetSyncPlan(keep=keep, delete=delete, upload=upload)


def sync_release(
    github: GitHubClient,
    gitee: GiteeClient,
    github_release: dict[str, Any],
    *,
    dry_run: bool,
    manifest_assets: list[DesiredAsset] | None = None,
    gitee_releases_by_tag: dict[str, dict[str, Any]] | None = None,
    release_index: int | None = None,
    release_total: int | None = None,
) -> str:
    tag = github_release["tag_name"]
    if release_index is not None and release_total is not None:
        log(f"[release] {release_index}/{release_total} {tag}")
    else:
        log(f"[release] {tag}")
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
    asset_plan = plan_asset_sync(desired_assets, gitee_assets)
    log(
        f"[plan] {tag}: keep={len(asset_plan.keep)} "
        f"delete={len(asset_plan.delete)} upload={len(asset_plan.upload)}"
    )
    if not asset_plan.delete and not asset_plan.upload:
        log(f"[ok] assets up-to-date for {tag}")
        return "unchanged"

    if dry_run:
        if asset_plan.delete:
            for asset in asset_plan.delete:
                log(f"[plan] delete {tag}: {asset['name']} ({asset['size']} bytes)")
        if asset_plan.upload:
            for asset in asset_plan.upload:
                log(f"[plan] upload {tag}: {asset.name} ({asset.size} bytes)")
        return "planned"

    for asset in asset_plan.delete:
        log(f"[sync] delete {tag}: {asset['name']}")
        gitee.delete_attach_file(release_id, int(asset["id"]))

    if not desired_assets:
        log(f"[ok] {tag}: release has no uploaded assets")
        return "updated"

    if not asset_plan.upload:
        log(f"[ok] {tag}: kept {len(asset_plan.keep)} existing asset(s)")
        return "updated"

    cache_root = get_download_cache_dir(f"{github.owner}/{github.repo}", tag)
    cache_root.mkdir(parents=True, exist_ok=True)
    log(f"[cache] {tag}: {cache_root}")

    staged_assets: list[tuple[DesiredAsset, Path]] = []
    for verify_round in range(1, 3):
        staged_assets = stage_upload_assets(github, desired_assets, cache_root, tag=tag)
        staged_paths = {asset.name: path for asset, path in staged_assets}
        digest_errors = collect_release_digest_errors(tag, staged_paths)
        if not digest_errors:
            break
        bad_names = sorted(digest_errors)
        if verify_round >= 2:
            raise ApiError("\n".join(digest_errors[name] for name in bad_names))
        log(f"[retry] {tag}: digest check failed for {len(bad_names)} asset(s), purging bad cache entries")
        for name in bad_names:
            log(f"[retry] {tag}: purge cache for {name}")
        purge_cached_assets(cache_root, bad_names)
    verify_release_digests(tag, {asset.name: path for asset, path in staged_assets})

    upload_total = len(asset_plan.upload)
    upload_lookup = {asset.name for asset in asset_plan.upload}
    upload_index = 0
    for asset, local_path in staged_assets:
        if asset.name not in upload_lookup:
            continue
        upload_index += 1
        log(
            f"[upload] {tag}: {upload_index}/{upload_total} "
            f"{asset.name} ({format_bytes(local_path.stat().st_size)})"
        )
        run_with_heartbeat(
            f"upload {tag} {asset.name}",
            lambda release_id=release_id, local_path=local_path: gitee.upload_attach_file(
                release_id, local_path
            ),
        )

    return "updated"


def purge_obsolete_gitee_v2_releases(
    gitee: GiteeClient,
    *,
    keep_tag: str,
    gitee_releases_by_tag: dict[str, dict[str, Any]],
    dry_run: bool,
) -> int:
    if not is_v2_tag(keep_tag):
        return 0

    obsolete_tags = [
        tag
        for tag in sorted(gitee_releases_by_tag)
        if tag != keep_tag and is_v2_tag(tag)
    ]
    if not obsolete_tags:
        log(f"[ok] {keep_tag}: no obsolete Gitee v2 releases to delete")
        return 0

    for tag in obsolete_tags:
        release = gitee_releases_by_tag[tag]
        if dry_run:
            log(f"[plan] delete obsolete Gitee v2 release {tag}")
            continue
        log(f"[sync] delete obsolete Gitee v2 release {tag}")
        gitee.delete_release(int(release["id"]))
        gitee_releases_by_tag.pop(tag, None)
    return len(obsolete_tags)


def main() -> int:
    args = parse_args()
    sync_mode, tag = resolve_sync_mode(args)
    github_owner, github_repo = split_repo(
        args.github_repo,
        "GITHUB_REPOSITORY",
        remote_name="origin",
    )
    resolved_gitee_repo = args.gitee_repo
    if not resolved_gitee_repo:
        resolved_gitee_repo = f"{github_owner}/{github_repo}"
        log(f"[info] defaulting GITEE_REPO to GitHub repo path: {resolved_gitee_repo}")
    gitee_owner, gitee_repo = split_repo(
        resolved_gitee_repo,
        "GITEE_REPO",
        remote_name="gitee",
    )
    gh_token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    gitee_token = require_env("GITEE_TOKEN")

    github = GitHubClient(github_owner, github_repo, gh_token)
    gitee = GiteeClient(gitee_owner, gitee_repo, gitee_token)
    log(f"[step] fetch existing Gitee releases from {gitee_owner}/{gitee_repo}")
    existing_gitee_releases = run_with_heartbeat(
        f"fetch Gitee releases {gitee_owner}/{gitee_repo}",
        gitee.list_releases,
    )
    gitee_releases_by_tag = {
        release["tag_name"]: release for release in existing_gitee_releases if release.get("tag_name")
    }

    manifest_assets: list[DesiredAsset] | None = None
    if args.asset_manifest:
        manifest_path = Path(args.asset_manifest).resolve()
        if not manifest_path.is_file():
            raise SystemExit(f"asset manifest not found: {manifest_path}")
        manifest_assets = load_desired_assets_from_manifest(manifest_path)

    if sync_mode == "all" and manifest_assets is not None:
        raise SystemExit("--asset-manifest only supports single-tag sync")

    statuses: list[str] = []
    deleted_releases = 0
    if sync_mode == "tag":
        assert tag is not None
        log(f"[step] fetch GitHub release {tag} from {github_owner}/{github_repo}")
        github_release = run_with_heartbeat(
            f"fetch GitHub release {tag}",
            lambda: github.get_release_by_tag(tag),
        )
        if manifest_assets is None:
            manifest_assets = prepare_gitee_release_assets(github, github_release)
        statuses.append(
            sync_release(
                github,
                gitee,
                github_release,
                dry_run=args.dry_run,
                manifest_assets=manifest_assets,
                gitee_releases_by_tag=gitee_releases_by_tag,
                release_index=1,
                release_total=1,
            )
        )
        deleted_releases += purge_obsolete_gitee_v2_releases(
            gitee,
            keep_tag=tag,
            gitee_releases_by_tag=gitee_releases_by_tag,
            dry_run=args.dry_run,
        )
    elif sync_mode == "latest":
        log(f"[step] fetch GitHub releases from {github_owner}/{github_repo}")
        github_releases = run_with_heartbeat(
            f"fetch GitHub releases {github_owner}/{github_repo}",
            github.list_releases,
        )
        latest_release = select_latest_release(github_releases)
        latest_tag = str(latest_release["tag_name"])
        log(f"[info] selected latest published release: {latest_tag}")
        desired_assets = manifest_assets if manifest_assets is not None else prepare_gitee_release_assets(github, latest_release)
        statuses.append(
            sync_release(
                github,
                gitee,
                latest_release,
                dry_run=args.dry_run,
                manifest_assets=desired_assets,
                gitee_releases_by_tag=gitee_releases_by_tag,
                release_index=1,
                release_total=1,
            )
        )
        deleted_releases += purge_obsolete_gitee_v2_releases(
            gitee,
            keep_tag=latest_tag,
            gitee_releases_by_tag=gitee_releases_by_tag,
            dry_run=args.dry_run,
        )
    else:
        log(f"[step] fetch GitHub releases from {github_owner}/{github_repo}")
        github_releases = run_with_heartbeat(
            f"fetch GitHub releases {github_owner}/{github_repo}",
            github.list_releases,
        )
        github_releases = [release for release in github_releases if not release.get("draft")]
        release_total = len(github_releases)
        for release_index, release in enumerate(github_releases, start=1):
            if release.get("draft"):
                continue
            statuses.append(
                sync_release(
                    github,
                    gitee,
                    release,
                    dry_run=args.dry_run,
                    manifest_assets=prepare_gitee_release_assets(github, release),
                    gitee_releases_by_tag=gitee_releases_by_tag,
                    release_index=release_index,
                    release_total=release_total,
                )
            )

    summary = {
        "planned": statuses.count("planned"),
        "updated": statuses.count("updated"),
        "unchanged": statuses.count("unchanged"),
        "skipped": statuses.count("skipped"),
        "deleted_releases": deleted_releases,
    }
    log(f"[summary] {json_dumps(summary)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ApiError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
