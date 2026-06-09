from __future__ import annotations

from cas_ocr_model.common.release_manifest import (
    build_release_manifest,
    iter_manifest_artifacts,
)


def test_build_release_manifest_includes_modellist_and_grouped_artifacts():
    manifest = build_release_manifest(
        model_entries=[
            {
                "asset_stem": "mobilenet_v3_small.trislot_decoder.v2_0",
                "backbone": "mobilenet_v3_small",
            },
            {
                "asset_stem": "mobilenetv4_conv_small.trislot_decoder.v2_0",
                "backbone": "mobilenetv4_conv_small",
            },
        ],
        artifacts=[
            {
                "asset_stem": "mobilenet_v3_small.trislot_decoder.v2_0",
                "engine": "pytorch",
                "precision": "fp32",
                "format": "checkpoint",
                "files": [{"path": "pytorch/a.pt", "release_asset_name": "a.pt"}],
            },
            {
                "asset_stem": "mobilenet_v3_small.trislot_decoder.v2_0",
                "engine": "onnx",
                "precision": "fp16",
                "format": "onnx",
                "files": [{"path": "onnx/a.fp16.onnx", "release_asset_name": "a.fp16.onnx"}],
            },
            {
                "asset_stem": "mobilenetv4_conv_small.trislot_decoder.v2_0",
                "engine": "ncnn",
                "precision": "fp32",
                "format": "ncnn",
                "files": [{"path": "ncnn/b.fp32.param", "release_asset_name": "b.fp32.param"}],
            },
        ],
        digests=[],
    )

    assert manifest["modellist"] == [
        "mobilenet_v3_small.trislot_decoder.v2_0",
        "mobilenetv4_conv_small.trislot_decoder.v2_0",
    ]
    assert manifest["model_count"] == 2
    assert manifest["models"][0]["artifacts"]["pytorch"]["fp32"]["files"][0]["path"] == "pytorch/a.pt"
    assert manifest["models"][0]["artifacts"]["onnx"]["fp16"]["files"][0]["path"] == "onnx/a.fp16.onnx"
    assert manifest["models"][1]["artifacts"]["ncnn"]["fp32"]["files"][0]["path"] == "ncnn/b.fp32.param"


def test_iter_manifest_artifacts_falls_back_to_grouped_models():
    manifest = {
        "models": [
            {
                "asset_stem": "mobilenet_v3_small.trislot_decoder.v2_0",
                "artifacts": {
                    "pytorch": {
                        "fp32": {
                            "format": "checkpoint",
                            "files": [{"path": "pytorch/a.pt"}],
                        }
                    }
                },
            }
        ]
    }

    artifacts = iter_manifest_artifacts(manifest)

    assert len(artifacts) == 1
    assert artifacts[0]["engine"] == "pytorch"
    assert artifacts[0]["precision"] == "fp32"
    assert artifacts[0]["asset_stem"] == "mobilenet_v3_small.trislot_decoder.v2_0"
