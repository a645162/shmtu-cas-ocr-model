from __future__ import annotations

import json

import torch
from cas_ocr_model.common.checkpoint_pip import (
    extract_checkpoint_pip_list,
    load_checkpoint_pip_snapshot,
    write_pip_list_json,
)


def test_extract_checkpoint_pip_list_from_top_level():
    raw = {
        "pip_list": [
            {"name": "torch", "version": "2.7.0"},
            {"name": "numpy", "version": "2.3.0"},
        ]
    }

    packages = extract_checkpoint_pip_list(raw)

    assert packages == raw["pip_list"]


def test_load_checkpoint_pip_snapshot_roundtrip(tmp_path):
    checkpoint_path = tmp_path / "model.pt"
    torch.save(
        {
            "pip_list": [{"name": "torch", "version": "2.7.0"}],
            "pip_list_metadata": {"schema_version": 1},
        },
        checkpoint_path,
    )

    packages, metadata = load_checkpoint_pip_snapshot(checkpoint_path)

    assert packages == [{"name": "torch", "version": "2.7.0"}]
    assert metadata == {"schema_version": 1}


def test_write_pip_list_json_writes_json_array(tmp_path):
    output_path = tmp_path / "model.pip-list.json"

    write_pip_list_json(output_path, [{"name": "torch", "version": "2.7.0"}])

    assert json.loads(output_path.read_text(encoding="utf-8")) == [
        {"name": "torch", "version": "2.7.0"}
    ]
    assert output_path.read_text(encoding="utf-8").endswith("\n")
