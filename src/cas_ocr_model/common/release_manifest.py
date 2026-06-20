"""Release manifest 结构化与兼容读取工具."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any


# ========== 模型显示名称翻译字典 (与 Android shmtu_ocr 端对齐) ==========
# 未来新增 backbone 时同步更新此表 + Android `canonicalBackboneLabels`
# 同步方式: 此文件 + shmtu-terminal-android/shmtu_ocr/src/main/java/cn/edu/shmtu/cas/ocr/ModelDownloader.kt
# 用途: 在 model-assets.json 的 `display_name` 字段中输出人类可读名称,
#       客户端可优先用此字段, fallback 到客户端字典.

_BACKBONE_DISPLAY_NAMES: dict[str, str] = {
    # MobileNet 系列
    "mobilenet_v3_small": "MobileNetV3-Small",
    "mobilenet_v3_large": "MobileNetV3-Large",
    "mobilenetv4_conv_small": "MobileNetV4-Conv-Small",
    "mobilenetv4_conv_medium": "MobileNetV4-Conv-Medium",
    "mobilenetv4_conv_large": "MobileNetV4-Conv-Large",
    "mobilenetv4_hybrid_medium": "MobileNetV4-Hybrid-Medium",
    # EfficientNet 系列
    "efficientnet_b0": "EfficientNet-B0",
    "efficientnet_b1": "EfficientNet-B1",
    "efficientnet_b2": "EfficientNet-B2",
    "efficientnet_b3": "EfficientNet-B3",
    "efficientnetv2_s": "EfficientNetV2-S",
    "efficientnetv2_m": "EfficientNetV2-M",
    # ResNet 系列 (历史 v1 模型)
    "resnet18": "ResNet-18",
    "resnet34": "ResNet-34",
    "resnet50": "ResNet-50",
    "resnet101": "ResNet-101",
    # ConvNeXt 系列
    "convnext_tiny": "ConvNeXt-Tiny",
    "convnext_small": "ConvNeXt-Small",
    "convnext_base": "ConvNeXt-Base",
    # ViT 系列
    "vit_small_patch16_224": "ViT-S/16",
    "vit_base_patch16_224": "ViT-B/16",
}

_FAMILY_DISPLAY_NAMES: dict[str, str] = {
    "trislot_decoder": "TriSlot Decoder",
    "single_head": "Single-Head",
    "multi_head": "Multi-Head",
    "ctc": "CTC",
}


def _to_pascal_case(snake: str) -> str:
    """snake_case / kebab-case → PascalCase (回退转换)."""
    if not snake:
        return ""
    parts = [p for p in snake.replace("-", "_").split("_") if p]
    return "".join(p[:1].upper() + p[1:] for p in parts)


def _is_version_token(s: str) -> bool:
    """判断字符串是否是 version 段: 'v2_0' / '2_0' / 'v1.0' 等.

    版本特征:
    - 纯数字 (e.g. "2", "210", "0")
    - v/V 开头后跟数字 (e.g. "v2_0", "V1.0", "v3.0.0")
    - 数字开头, 后跟 _ 或 . + 数字, 不含字母 (e.g. "2_0", "2.0.1")
    - backbone 特征: 含字母数字下划线且其中包含字母单词 (mobilenet_v3_small 含 "mobilenet")
    区分: 有字母下划线不算 version, 纯数字下划线算 version
    """
    if not s:
        return False
    # 纯数字
    if s.isdigit():
        return True
    # v/V 开头 + 数字
    if s[0] in "vV" and len(s) >= 2 and s[1].isdigit():
        return True
    # 数字开头, 后续只含数字/下划线/点 (e.g. "2_0", "2.0.1", "2_0_1")
    if s[0].isdigit():
        ok = True
        for ch in s:
            if not (ch.isdigit() or ch in "_."):
                ok = False
                break
        if ok:
            return True
    return False


def friendly_model_name(
    asset_stem: str,
    backbone: str = "",
    family: str = "",
    *,
    separator: str = " + ",
) -> str:
    """把 asset_stem 翻译成人类可读名称.

    asset_stem 约定: `<backbone>.<family>.<version>`
    例: `mobilenet_v3_small.trislot_decoder.v2_0`
    →  `MobileNetV3-Small + TriSlot Decoder + v2.0`

    解析策略: 三段或两段都支持, 末段识别为 version (数字开头或 v 开头).
    当有四段或更多 (如 backbone/family/v1.0), 末段是 v1.0, 倒数第二段是 family.

    Parameters
    ----------
    asset_stem : str
        模型文件名 (不含扩展名)
    backbone : str, optional
        兼容历史 manifest: 没有 asset_stem 时用 backbone 字段
    family : str, optional
        同 backbone
    separator : str, optional
        分隔符, 默认 " + "
    """
    parts = [p for p in asset_stem.split(".") if p]
    if not parts:
        # 退回到 backbone / family 字段 (历史 manifest 兼容)
        bb = _BACKBONE_DISPLAY_NAMES.get(backbone.lower(), _to_pascal_case(backbone))
        fa = _FAMILY_DISPLAY_NAMES.get(family.lower(), _to_pascal_case(family))
        return separator.join(p for p in (bb, fa) if p)

    # 从后往前找 version token, 合并连续的 ver/数字 token 为完整版本号
    ver_end = len(parts)  # 第一个非 version token 的位置
    for i in range(len(parts) - 1, -1, -1):
        if _is_version_token(parts[i]):
            ver_end = i
        else:
            break

    if ver_end < len(parts):
        bb_raw = parts[0] if ver_end > 1 else backbone
        fa_raw = ".".join(parts[1:ver_end]) if ver_end > 1 else (parts[1] if ver_end == 1 else family)
        ver_raw = ".".join(parts[ver_end:])
    elif len(parts) >= 2:
        bb_raw = parts[0]
        fa_raw = ".".join(parts[1:])
        ver_raw = ""
    else:
        bb_raw = parts[0]
        fa_raw = family
        ver_raw = ""

    bb = _BACKBONE_DISPLAY_NAMES.get(bb_raw.lower(), _to_pascal_case(bb_raw))
    fa = _FAMILY_DISPLAY_NAMES.get(fa_raw.lower(), _to_pascal_case(fa_raw))
    # version 兼容 "v2_0" / "2_0" / "v2.0" / "2.0" / "v1.0" → "v2.0"
    ver = ""
    if ver_raw:
        cleaned = ver_raw.lstrip("vV").replace("_", ".")
        if cleaned:
            ver = f"v{cleaned}"

    return separator.join(p for p in (bb, fa, ver) if p)


def _normalize_artifact_files(raw_files: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_files, list):
        return []
    files: list[dict[str, Any]] = []
    for item in raw_files:
        if not isinstance(item, Mapping):
            continue
        files.append(dict(item))
    return files


def _normalize_artifact(raw_artifact: Any) -> dict[str, Any] | None:
    if not isinstance(raw_artifact, Mapping):
        return None
    artifact = dict(raw_artifact)
    artifact["files"] = _normalize_artifact_files(raw_artifact.get("files"))
    return artifact


def iter_manifest_artifacts(raw_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
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
        if not isinstance(raw_model, Mapping):
            continue
        grouped = raw_model.get("artifacts")
        if not isinstance(grouped, Mapping):
            continue
        for engine, precision_map in grouped.items():
            if not isinstance(engine, str) or not isinstance(precision_map, Mapping):
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


def group_artifacts_by_model(artifacts: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
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
        # 自动注入 display_name (e.g. "MobileNetV3-Small + TriSlot Decoder + v2.0")
        # 如果调用方已显式提供, 则保留调用方值
        if "display_name" not in entry or not entry["display_name"]:
            entry = dict(entry)  # 不修改原 entry
            entry["display_name"] = friendly_model_name(
                asset_stem=asset_stem,
                backbone=str(entry.get("backbone", "")),
                family=str(entry.get("family", "")),
            )
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
