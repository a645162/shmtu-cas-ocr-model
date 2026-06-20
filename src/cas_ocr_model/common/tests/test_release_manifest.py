from __future__ import annotations

from cas_ocr_model.common.release_manifest import (
    build_release_manifest,
    friendly_model_name,
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


def test_friendly_model_name_known_backbones():
    """Test display name translation for known backbones."""
    cases = [
        ("mobilenet_v3_small.trislot_decoder.v2_0", "MobileNetV3-Small + TriSlot Decoder + v2.0"),
        ("mobilenetv4_conv_small.trislot_decoder.v2_0", "MobileNetV4-Conv-Small + TriSlot Decoder + v2.0"),
        ("mobilenetv4_conv_medium.trislot_decoder.v2_0", "MobileNetV4-Conv-Medium + TriSlot Decoder + v2.0"),
        ("mobilenetv4_conv_large.trislot_decoder.v2_0", "MobileNetV4-Conv-Large + TriSlot Decoder + v2.0"),
        ("efficientnet_b0.trislot_decoder.v2_0", "EfficientNet-B0 + TriSlot Decoder + v2.0"),
        ("efficientnet_b3.trislot_decoder.v2_0", "EfficientNet-B3 + TriSlot Decoder + v2.0"),
        ("efficientnetv2_s.trislot_decoder.v2_0", "EfficientNetV2-S + TriSlot Decoder + v2.0"),
        ("resnet18.trislot_decoder.v2_0", "ResNet-18 + TriSlot Decoder + v2.0"),
        ("resnet50.single_head.v1_0", "ResNet-50 + Single-Head + v1.0"),
        ("convnext_tiny.trislot_decoder.v2_0", "ConvNeXt-Tiny + TriSlot Decoder + v2.0"),
        ("vit_small_patch16_224.trislot_decoder.v2_0", "ViT-S/16 + TriSlot Decoder + v2.0"),
    ]
    for asset_stem, expected in cases:
        got = friendly_model_name(asset_stem)
        assert got == expected, f"{asset_stem!r} → {got!r}, expected {expected!r}"


def test_friendly_model_name_unknown_backbone_pascalcase_fallback():
    """Unknown backbones should fallback to PascalCase conversion."""
    got = friendly_model_name("mystery_model_xyz.trislot_decoder.v2_0")
    assert got == "MysteryModelXyz + TriSlot Decoder + v2.0"

    # Without v prefix in version
    got = friendly_model_name("mobilenet_v3_small.trislot_decoder")
    assert got == "MobileNetV3-Small + TriSlot Decoder"


def test_friendly_model_name_fallback_to_backbone_field():
    """When asset_stem is empty, fallback to backbone field."""
    got = friendly_model_name("", backbone="mobilenet_v3_small", family="trislot_decoder")
    assert got == "MobileNetV3-Small + TriSlot Decoder"


def test_build_release_manifest_auto_injects_display_name():
    """build_release_manifest should auto-inject display_name from asset_stem."""
    manifest = build_release_manifest(
        model_entries=[
            {
                "asset_stem": "mobilenet_v3_small.trislot_decoder.v2_0",
                "backbone": "mobilenet_v3_small",
                "family": "trislot_decoder",
                "version": "v2_0",
            }
        ],
        artifacts=[],
        digests=[],
    )
    assert manifest["models"][0]["display_name"] == "MobileNetV3-Small + TriSlot Decoder + v2.0"


def test_build_release_manifest_preserves_existing_display_name():
    """If display_name is already provided, do not overwrite."""
    manifest = build_release_manifest(
        model_entries=[
            {
                "asset_stem": "mobilenet_v3_small.trislot_decoder.v2_0",
                "backbone": "mobilenet_v3_small",
                "display_name": "My Custom Display Name",
            }
        ],
        artifacts=[],
        digests=[],
    )
    assert manifest["models"][0]["display_name"] == "My Custom Display Name"


def test_build_release_manifest_deduplicates_by_asset_stem():
    """Duplicate asset_stems are deduplicated, only first kept."""
    manifest = build_release_manifest(
        model_entries=[
            {"asset_stem": "mobilenet_v3_small.trislot_decoder.v2_0"},
            {"asset_stem": "mobilenet_v3_small.trislot_decoder.v2_0"},  # duplicate
        ],
        artifacts=[],
        digests=[],
    )
    assert manifest["model_count"] == 1
