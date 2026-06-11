from __future__ import annotations

import json

from cas_ocr_model.export.release_upload_list import collect_release_uploads


def test_collect_release_uploads_includes_root_digest_even_if_manifest_omits_it(tmp_path):
    output_root = tmp_path / "release_export"
    output_root.mkdir()
    (output_root / "pytorch").mkdir()

    manifest = {
        "schema_version": 2,
        "artifacts": [
            {
                "asset_stem": "mobilenet_v3_small.trislot_decoder.v2_0",
                "engine": "pytorch",
                "precision": "fp32",
                "format": "checkpoint",
                "files": [
                    {
                        "path": "pytorch/mobilenet_v3_small.trislot_decoder.v2_0.pt",
                        "release_asset_name": "mobilenet_v3_small.trislot_decoder.v2_0.pt",
                    }
                ],
            }
        ],
        "digests": [],
    }
    (output_root / "model-assets.json").write_text(json.dumps(manifest), encoding="utf-8")
    (output_root / "pytorch" / "mobilenet_v3_small.trislot_decoder.v2_0.pt").write_bytes(b"pt")
    (output_root / "SHA256SUMS.txt").write_text("dummy\n", encoding="utf-8")

    uploads = collect_release_uploads(output_root)
    upload_names = [release_name for _, release_name in uploads]

    assert "model-assets.json" in upload_names
    assert "mobilenet_v3_small.trislot_decoder.v2_0.pt" in upload_names
    assert "SHA256SUMS.txt" in upload_names


def test_collect_release_uploads_deduplicates_digest_when_manifest_already_lists_it(tmp_path):
    output_root = tmp_path / "release_export"
    output_root.mkdir()

    manifest = {
        "schema_version": 2,
        "artifacts": [],
        "digests": [
            {
                "path": "SHA256SUMS.txt",
                "release_asset_name": "SHA256SUMS.txt",
            }
        ],
    }
    (output_root / "model-assets.json").write_text(json.dumps(manifest), encoding="utf-8")
    (output_root / "SHA256SUMS.txt").write_text("dummy\n", encoding="utf-8")

    uploads = collect_release_uploads(output_root)
    digest_uploads = [release_name for _, release_name in uploads if release_name == "SHA256SUMS.txt"]

    assert len(digest_uploads) == 1
