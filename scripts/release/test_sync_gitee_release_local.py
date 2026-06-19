from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


RELEASE_DIR = Path(__file__).resolve().parent
SYNC_MODULE_PATH = RELEASE_DIR / "sync_gitee_releases.py"
LOCAL_MODULE_PATH = RELEASE_DIR / "sync_gitee_release_local.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sync_gitee_releases = _load_module("sync_gitee_releases", SYNC_MODULE_PATH)
sync_gitee_release_local = _load_module("sync_gitee_release_local", LOCAL_MODULE_PATH)

download_release_assets = sync_gitee_release_local.download_release_assets
prepare_local_gitee_release_assets = sync_gitee_release_local.prepare_local_gitee_release_assets
resolve_release = sync_gitee_release_local.resolve_release
resolve_download_root = sync_gitee_release_local.resolve_download_root
DesiredAsset = sync_gitee_releases.DesiredAsset


def test_download_release_assets_builds_local_manifest(tmp_path, monkeypatch) -> None:
    def fake_run_gh(args, *, expect_json=False):
        assert args[:4] == ["release", "download", "v2.0.5", "--repo"]
        assert expect_json is False
        asset_path = tmp_path / "model.bin"
        asset_path.write_bytes(b"1234")
        return ""

    monkeypatch.setattr(sync_gitee_release_local, "run_gh", fake_run_gh)
    release = {
        "tag_name": "v2.0.5",
        "assets": [{"name": "model.bin", "size": 4}],
    }

    desired_assets = download_release_assets("owner/repo", release, tmp_path)

    assert desired_assets == [DesiredAsset(name="model.bin", size=4, local_path=tmp_path / "model.bin")]


def test_resolve_download_root_keeps_temp_dir_when_requested(tmp_path) -> None:
    args = argparse.Namespace(tag="v2.0.5", download_dir=None, keep_downloads=True)

    root, cleanup = resolve_download_root(args)

    assert root.is_dir()
    assert cleanup is False


def test_resolve_release_uses_latest_when_tag_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        sync_gitee_release_local,
        "fetch_github_releases",
        lambda repo: [
            {"tag_name": "v2.0.4", "draft": False, "prerelease": False},
            {"tag_name": "v2.0.5", "draft": False, "prerelease": False},
        ],
    )

    release = resolve_release("owner/repo", None)

    assert release["tag_name"] == "v2.0.5"


def test_prepare_local_gitee_release_assets_builds_slim_bundle(tmp_path, monkeypatch) -> None:
    release = {
        "tag_name": "v2.0.5",
        "assets": [
            {"name": "model-assets.json", "size": 1},
            {"name": "mobilenet_v3_small.pt", "size": 2},
            {"name": "mobilenetv4_conv_small.onnx", "size": 2},
            {"name": "resnet18.pt", "size": 3},
        ],
    }
    manifest = {
        "schema_version": 2,
        "generated_at_utc": "2026-06-19T00:00:00Z",
        "models": [
            {"asset_stem": "mobilenet_v3_small.trislot_decoder.v2_0", "backbone": "mobilenet_v3_small"},
            {"asset_stem": "mobilenetv4_conv_small.trislot_decoder.v2_0", "backbone": "mobilenetv4_conv_small"},
            {"asset_stem": "resnet18.trislot_decoder.v2_0", "backbone": "resnet18"},
        ],
        "artifacts": [
            {
                "asset_stem": "mobilenet_v3_small.trislot_decoder.v2_0",
                "engine": "pytorch",
                "precision": "fp32",
                "files": [{"path": "pytorch/v3.pt", "release_asset_name": "mobilenet_v3_small.pt"}],
            },
            {
                "asset_stem": "mobilenetv4_conv_small.trislot_decoder.v2_0",
                "engine": "onnx",
                "precision": "fp16",
                "files": [{"path": "onnx/v4.onnx", "release_asset_name": "mobilenetv4_conv_small.onnx"}],
            },
            {
                "asset_stem": "resnet18.trislot_decoder.v2_0",
                "engine": "pytorch",
                "precision": "fp32",
                "files": [{"path": "pytorch/resnet18.pt", "release_asset_name": "resnet18.pt"}],
            },
        ],
    }

    def fake_download_by_name(repo, tag, asset_names, dest_dir):
        staged = {}
        for name in asset_names:
            path = dest_dir / name
            if name == "model-assets.json":
                path.write_text(json.dumps(manifest), encoding="utf-8")
            elif name == "mobilenet_v3_small.pt":
                path.write_bytes(b"v3")
            elif name == "mobilenetv4_conv_small.onnx":
                path.write_bytes(b"v4")
            else:
                raise AssertionError(f"unexpected asset download: {name}")
            staged[name] = path
        return staged

    monkeypatch.setattr(sync_gitee_release_local, "download_release_assets_by_name", fake_download_by_name)

    desired_assets = prepare_local_gitee_release_assets("owner/repo", release, tmp_path)

    names = [asset.name for asset in desired_assets]
    assert names == [
        "mobilenet_v3_small.pt",
        "mobilenetv4_conv_small.onnx",
        "model-assets.json",
        "SHA256SUMS.txt",
    ]
