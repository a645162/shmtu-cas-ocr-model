"""模型版本注册表与发布资产命名."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import torch.nn as nn

from cas_ocr_model.common.release_manifest import iter_manifest_artifacts
from .backbones import is_supported_backbone, list_available_backbones
from .versions import MODEL_FAMILY as DEFAULT_MODEL_FAMILY
from .versions import MODEL_VERSION as DEFAULT_MODEL_VERSION
from .versions import build_v2_0_model

ModelBuilder = Callable[..., nn.Module]


@dataclass(frozen=True)
class ModelSpec:
    version: str
    family: str
    display_name: str
    builder: ModelBuilder
    supported_backbones: tuple[str, ...]


_MODEL_SPECS: dict[str, ModelSpec] = {
    DEFAULT_MODEL_VERSION: ModelSpec(
        version=DEFAULT_MODEL_VERSION,
        family=DEFAULT_MODEL_FAMILY,
        display_name="CAS OCR TriSlot Decoder",
        builder=build_v2_0_model,
        supported_backbones=tuple(list_available_backbones()),
    ),
}


def normalize_model_version(version: str | None) -> str:
    if version is None:
        return DEFAULT_MODEL_VERSION
    raw = str(version).strip()
    if not raw:
        return DEFAULT_MODEL_VERSION
    if raw == "2":
        return "2.0"
    return raw


def list_model_versions() -> list[str]:
    return sorted(_MODEL_SPECS.keys())


def get_model_spec(version: str | None = None) -> ModelSpec:
    normalized = normalize_model_version(version)
    if normalized not in _MODEL_SPECS:
        raise ValueError(
            f"unknown model version: {normalized}; available={list_model_versions()}"
        )
    return _MODEL_SPECS[normalized]


def sanitize_version_for_filename(version: str | None) -> str:
    normalized = normalize_model_version(version)
    return normalized.replace(".", "_").replace("-", "_")


def build_release_asset_stem(
    *,
    backbone: str,
    version: str | None = None,
    family: str | None = None,
) -> str:
    spec = get_model_spec(version)
    resolved_family = (family or spec.family).strip().lower()
    return f"{backbone}.{resolved_family}.v{sanitize_version_for_filename(spec.version)}"


def extract_model_config(raw_cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(raw_cfg, Mapping):
        return {}
    model_cfg = raw_cfg.get("model", raw_cfg)
    return dict(model_cfg) if isinstance(model_cfg, Mapping) else {}


def build_model_metadata(model_cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    cfg = extract_model_config(model_cfg)
    spec = get_model_spec(cfg.get("version"))
    backbone = str(cfg.get("backbone", "resnet18"))
    if not is_supported_backbone(backbone):
        raise ValueError(
            f"model version {spec.version} does not support backbone={backbone}; "
            f"available={list(spec.supported_backbones)}"
        )
    asset_stem = build_release_asset_stem(
        backbone=backbone,
        version=spec.version,
        family=spec.family,
    )
    return {
        "version": spec.version,
        "family": spec.family,
        "display_name": spec.display_name,
        "backbone": backbone,
        "asset_stem": asset_stem,
        "supported_backbones": list(spec.supported_backbones),
    }


def build_model_from_config(
    model_cfg: Mapping[str, Any] | None,
    *,
    pretrained_override: bool | None = None,
    num_digit_classes: int = 10,
    num_operator_classes: int = 3,
) -> nn.Module:
    cfg = extract_model_config(model_cfg)
    metadata = build_model_metadata(cfg)
    spec = get_model_spec(metadata["version"])
    pretrained = cfg.get("pretrained", True)
    if pretrained_override is not None:
        pretrained = pretrained_override
    return spec.builder(
        backbone=metadata["backbone"],
        pretrained=bool(pretrained),
        dropout=float(cfg.get("dropout", 0.2)),
        slot_hidden_dim=int(cfg.get("slot_hidden_dim", 256)),
        slot_attention_heads=int(cfg.get("slot_attention_heads", 4)),
        num_digit_classes=num_digit_classes,
        num_operator_classes=num_operator_classes,
    )


def extract_checkpoint_metadata(raw_checkpoint: Any) -> dict[str, Any]:
    if not isinstance(raw_checkpoint, Mapping):
        return build_model_metadata({})
    explicit = raw_checkpoint.get("model_metadata")
    if isinstance(explicit, Mapping) and explicit.get("asset_stem"):
        data = dict(explicit)
        data["version"] = normalize_model_version(data.get("version"))
        return data
    cfg = raw_checkpoint.get("config", {})
    return build_model_metadata(cfg)


def inspect_checkpoint(checkpoint: str | Path) -> dict[str, Any]:
    raw = torch.load(checkpoint, map_location="cpu")
    metadata = extract_checkpoint_metadata(raw)
    metadata["checkpoint"] = str(Path(checkpoint).expanduser().resolve())
    return metadata


def find_release_checkpoint(release_root: str | Path) -> Path:
    root = Path(release_root).expanduser().resolve()
    manifest_path = root / "model-assets.json"
    if manifest_path.is_file():
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        for artifact in iter_manifest_artifacts(raw):
            if artifact.get("engine") != "pytorch":
                continue
            if artifact.get("precision") != "fp32":
                continue
            files = artifact.get("files", [])
            if not files:
                continue
            candidate = root / files[0]["path"]
            if candidate.is_file():
                return candidate.resolve()

    pytorch_dir = root / "pytorch"
    if pytorch_dir.is_dir():
        candidates = [
            path for path in sorted(pytorch_dir.glob("*.pt"))
            if path.name not in {"best.pt", "last.pt"}
        ]
        if candidates:
            return candidates[0].resolve()

    raise FileNotFoundError(f"unable to find release pytorch checkpoint under {root}")
