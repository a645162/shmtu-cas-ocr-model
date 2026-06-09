"""Release manifest 结构化与兼容读取工具."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping


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
