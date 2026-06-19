from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


MODULE_PATH = Path(__file__).resolve().parent / "sync_gitee_releases.py"
SPEC = importlib.util.spec_from_file_location("sync_gitee_releases", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
sync_gitee_releases = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sync_gitee_releases
SPEC.loader.exec_module(sync_gitee_releases)

DesiredAsset = sync_gitee_releases.DesiredAsset
plan_asset_sync = sync_gitee_releases.plan_asset_sync
ApiError = sync_gitee_releases.ApiError
GiteeClient = sync_gitee_releases.GiteeClient
select_latest_release = sync_gitee_releases.select_latest_release
plan_gitee_v2_slim_assets = sync_gitee_releases.plan_gitee_v2_slim_assets
write_gitee_v2_slim_bundle = sync_gitee_releases.write_gitee_v2_slim_bundle
purge_obsolete_gitee_v2_releases = sync_gitee_releases.purge_obsolete_gitee_v2_releases
MANIFEST_ASSET_NAME = sync_gitee_releases.MANIFEST_ASSET_NAME
DIGEST_ASSET_NAME = sync_gitee_releases.DIGEST_ASSET_NAME


def test_plan_asset_sync_keeps_matching_uploaded_files() -> None:
    desired_assets = [
        DesiredAsset(name="a.bin", size=10),
        DesiredAsset(name="b.bin", size=20),
        DesiredAsset(name="c.bin", size=30),
    ]
    gitee_assets = [
        {"id": 1, "name": "a.bin", "size": 10},
        {"id": 2, "name": "b.bin", "size": 20},
    ]

    plan = plan_asset_sync(desired_assets, gitee_assets)

    assert [asset["name"] for asset in plan.keep] == ["a.bin", "b.bin"]
    assert plan.delete == []
    assert [asset.name for asset in plan.upload] == ["c.bin"]


def test_plan_asset_sync_deletes_only_mismatched_or_extra_files() -> None:
    desired_assets = [
        DesiredAsset(name="a.bin", size=10),
        DesiredAsset(name="b.bin", size=20),
    ]
    gitee_assets = [
        {"id": 1, "name": "a.bin", "size": 999},
        {"id": 2, "name": "b.bin", "size": 20},
        {"id": 3, "name": "extra.bin", "size": 5},
        {"id": 4, "name": "b.bin", "size": 20},
    ]

    plan = plan_asset_sync(desired_assets, gitee_assets)

    assert [asset["name"] for asset in plan.keep] == ["b.bin"]
    assert [(asset["name"], asset["id"]) for asset in plan.delete] == [
        ("a.bin", 1),
        ("b.bin", 4),
        ("extra.bin", 3),
    ]
    assert [asset.name for asset in plan.upload] == ["a.bin"]


def test_upload_attach_file_retries_until_success(tmp_path, monkeypatch) -> None:
    file_path = tmp_path / "model.bin"
    file_path.write_bytes(b"1234")
    client = GiteeClient("owner", "repo", "token")
    calls = {"post": 0, "list": 0, "sleep": 0}

    def fake_request(method, url, *, headers=None, data=None):
        if method == "POST":
            calls["post"] += 1
            if calls["post"] == 1:
                raise ApiError("timeout", retryable=True)
            return ({"id": 7, "name": "model.bin", "size": 4}, 201)
        if method == "GET":
            calls["list"] += 1
            return ([], 200)
        raise AssertionError(f"unexpected request method: {method}")

    monkeypatch.setattr(sync_gitee_releases, "request", fake_request)
    monkeypatch.setattr(sync_gitee_releases.time, "sleep", lambda _: calls.__setitem__("sleep", calls["sleep"] + 1))

    result = client.upload_attach_file(123, file_path)

    assert result["id"] == 7
    assert calls == {"post": 2, "list": 1, "sleep": 1}


def test_upload_attach_file_accepts_remote_success_after_timeout(tmp_path, monkeypatch) -> None:
    file_path = tmp_path / "model.bin"
    file_path.write_bytes(b"1234")
    client = GiteeClient("owner", "repo", "token")
    calls = {"post": 0, "list": 0, "sleep": 0}

    def fake_request(method, url, *, headers=None, data=None):
        if method == "POST":
            calls["post"] += 1
            raise ApiError("timeout", retryable=True)
        if method == "GET":
            calls["list"] += 1
            return ([{"id": 9, "name": "model.bin", "size": 4}], 200)
        raise AssertionError(f"unexpected request method: {method}")

    monkeypatch.setattr(sync_gitee_releases, "request", fake_request)
    monkeypatch.setattr(sync_gitee_releases.time, "sleep", lambda _: calls.__setitem__("sleep", calls["sleep"] + 1))

    result = client.upload_attach_file(123, file_path)

    assert result["id"] == 9
    assert calls == {"post": 1, "list": 1, "sleep": 0}


def test_select_latest_release_prefers_highest_semver() -> None:
    release = select_latest_release(
        [
            {"tag_name": "v1.0-NCNN", "draft": False, "prerelease": False},
            {"tag_name": "v2.0.4", "draft": False, "prerelease": False},
            {"tag_name": "v2.0.5", "draft": False, "prerelease": False},
        ]
    )

    assert release["tag_name"] == "v2.0.5"


def test_plan_gitee_v2_slim_assets_keeps_only_two_mobilenet_models() -> None:
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
                "files": [{"path": "pytorch/mobilenet_v3_small.pt", "release_asset_name": "mobilenet_v3_small.pt"}],
            },
            {
                "asset_stem": "mobilenetv4_conv_small.trislot_decoder.v2_0",
                "engine": "onnx",
                "precision": "fp16",
                "files": [{"path": "onnx/mobilenetv4_conv_small.onnx", "release_asset_name": "mobilenetv4_conv_small.onnx"}],
            },
            {
                "asset_stem": "resnet18.trislot_decoder.v2_0",
                "engine": "pytorch",
                "precision": "fp32",
                "files": [{"path": "pytorch/resnet18.pt", "release_asset_name": "resnet18.pt"}],
            },
        ],
    }

    model_entries, artifacts, asset_names = plan_gitee_v2_slim_assets(manifest)

    assert [entry["asset_stem"] for entry in model_entries] == [
        "mobilenet_v3_small.trislot_decoder.v2_0",
        "mobilenetv4_conv_small.trislot_decoder.v2_0",
    ]
    assert [artifact["asset_stem"] for artifact in artifacts] == [
        "mobilenet_v3_small.trislot_decoder.v2_0",
        "mobilenetv4_conv_small.trislot_decoder.v2_0",
    ]
    assert asset_names == ["mobilenet_v3_small.pt", "mobilenetv4_conv_small.onnx"]


def test_write_gitee_v2_slim_bundle_rebuilds_manifest_and_digest(tmp_path) -> None:
    v3 = tmp_path / "mobilenet_v3_small.pt"
    v4 = tmp_path / "mobilenetv4_conv_small.onnx"
    v3.write_bytes(b"v3")
    v4.write_bytes(b"v4")
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
        ],
    }

    manifest_path, digest_path = write_gitee_v2_slim_bundle(
        manifest,
        staged_asset_paths={
            "mobilenet_v3_small.pt": v3,
            "mobilenetv4_conv_small.onnx": v4,
        },
        output_dir=tmp_path,
    )

    rebuilt_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest_text = digest_path.read_text(encoding="utf-8")

    assert manifest_path.name == MANIFEST_ASSET_NAME
    assert digest_path.name == DIGEST_ASSET_NAME
    assert rebuilt_manifest["modellist"] == [
        "mobilenet_v3_small.trislot_decoder.v2_0",
        "mobilenetv4_conv_small.trislot_decoder.v2_0",
    ]
    assert rebuilt_manifest["model_count"] == 2
    assert rebuilt_manifest["artifacts"][0]["files"][0]["path"] == "mobilenet_v3_small.pt"
    assert rebuilt_manifest["artifacts"][1]["files"][0]["path"] == "mobilenetv4_conv_small.onnx"
    assert "mobilenet_v3_small.pt" in digest_text
    assert "mobilenetv4_conv_small.onnx" in digest_text


def test_purge_obsolete_gitee_v2_releases_deletes_only_other_v2_tags() -> None:
    deleted_ids: list[int] = []

    class FakeGitee:
        def delete_release(self, release_id: int) -> None:
            deleted_ids.append(release_id)

    releases_by_tag = {
        "v2.0.3": {"id": 3, "tag_name": "v2.0.3"},
        "v2.0.4": {"id": 4, "tag_name": "v2.0.4"},
        "v2.0.5": {"id": 5, "tag_name": "v2.0.5"},
        "v1.0-NCNN": {"id": 10, "tag_name": "v1.0-NCNN"},
    }

    deleted = purge_obsolete_gitee_v2_releases(
        FakeGitee(),
        keep_tag="v2.0.5",
        gitee_releases_by_tag=releases_by_tag,
        dry_run=False,
    )

    assert deleted == 2
    assert deleted_ids == [3, 4]
    assert sorted(releases_by_tag) == ["v1.0-NCNN", "v2.0.5"]


def test_plan_asset_sync_prunes_extra_assets_from_latest_gitee_tag() -> None:
    desired_assets = [
        DesiredAsset(name="mobilenet_v3_small.pt", size=10),
        DesiredAsset(name="mobilenetv4_conv_small.onnx", size=20),
        DesiredAsset(name="model-assets.json", size=30),
        DesiredAsset(name="SHA256SUMS.txt", size=40),
    ]
    gitee_assets = [
        {"id": 1, "name": "mobilenet_v3_small.pt", "size": 10},
        {"id": 2, "name": "mobilenetv4_conv_small.onnx", "size": 20},
        {"id": 3, "name": "model-assets.json", "size": 30},
        {"id": 4, "name": "SHA256SUMS.txt", "size": 40},
        {"id": 5, "name": "resnet18.pt", "size": 999},
        {"id": 6, "name": "resnet34.onnx", "size": 888},
    ]

    plan = plan_asset_sync(desired_assets, gitee_assets)

    assert [asset["name"] for asset in plan.keep] == [
        "mobilenet_v3_small.pt",
        "mobilenetv4_conv_small.onnx",
        "model-assets.json",
        "SHA256SUMS.txt",
    ]
    assert [(asset["name"], asset["id"]) for asset in plan.delete] == [
        ("resnet18.pt", 5),
        ("resnet34.onnx", 6),
    ]
    assert plan.upload == []
