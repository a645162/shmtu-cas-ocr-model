"""数据集格式规范 (manifest).

目录结构:
    dataset_root/
        NNNNNNNN.jpg          # 8 位编号图片
        NNNNNNNN.json         # 同名 json, 含 expression/answer
        manifest.json         # 本模块负责生成/读取的元数据

manifest.json 字段:
    version:        int       规范版本, 后续可演进
    created_at:     int       unix 秒
    label_set:      dict      {"digit": ["0".."9"], "operator": ["+","-","*",""/"]}
    source:         dict      采集来源 (backend, ocr_url, ...)
    splits:         dict      {"train": [...], "val": [...], "test": [...]}
    stats:          dict      {"n_total": int, "n_train": int, "n_val": int, "n_test": int}

DDP DataLoader 友好: 训练时读 manifest, 按本进程 rank/world_size 分片
(分布式采样器), 不必物理复制图片.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

# 8 位编号 jpg
FILENAME_RE = re.compile(r"^(\d{8})\.jpg$")
# 8 位编号 json
JSON_RE = re.compile(r"^(\d{8})\.json$")

MANIFEST_FILENAME = "manifest.json"
MANIFEST_VERSION = 1


@dataclass
class DatasetManifest:
    version: int = MANIFEST_VERSION
    created_at: int = 0
    label_set: dict = field(default_factory=lambda: {"digit": [str(i) for i in range(10)], "operator": ["+", "-", "*", "/"]})
    source: dict = field(default_factory=dict)
    splits: dict = field(default_factory=lambda: {"train": [], "val": [], "test": []})
    stats: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "DatasetManifest":
        raw = json.loads(s)
        return cls(**raw)

    @classmethod
    def load(cls, dataset_root: str | Path) -> "DatasetManifest":
        path = Path(dataset_root) / MANIFEST_FILENAME
        if not path.is_file():
            raise FileNotFoundError(
                f"manifest not found at {path}; 请先用 datasets/split.py 生成"
            )
        return cls.from_json(path.read_text(encoding="utf-8"))

    def save(self, dataset_root: str | Path) -> Path:
        path = Path(dataset_root) / MANIFEST_FILENAME
        path.write_text(self.to_json(), encoding="utf-8")
        return path


# ----------------------------------------------------------------------------
# 扫描工具
# ----------------------------------------------------------------------------


@dataclass
class ScanResult:
    """扫描 dataset_root 得到的结果."""

    jpg_names: list[str]      # ["00000000.jpg", ...]
    json_names: list[str]     # ["00000000.json", ...]
    paired_names: list[str]   # 同时有 jpg + json 的 NNNNNNNN.jpg
    missing_json: list[str]   # 有 jpg 缺 json
    missing_jpg: list[str]    # 有 json 缺 jpg

    @property
    def n_paired(self) -> int:
        return len(self.paired_names)


def scan_dataset(dataset_root: str | Path) -> ScanResult:
    """扫描 dataset_root 下的 jpg/json 配对."""
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset_root not found: {root}")

    jpgs = {p.name for p in root.glob("[0-9]" * 8 + ".jpg")}
    jsons = {p.name for p in root.glob("[0-9]" * 8 + ".json")}

    paired: list[str] = []
    missing_json: list[str] = []
    for jpg in sorted(jpgs):
        json_name = jpg.replace(".jpg", ".json")
        if json_name in jsons:
            paired.append(jpg)
        else:
            missing_json.append(jpg)
    missing_jpg = sorted(jsons - {j.replace(".jpg", ".json") for j in jpgs})

    return ScanResult(
        jpg_names=sorted(jpgs),
        json_names=sorted(jsons),
        paired_names=paired,
        missing_json=missing_json,
        missing_jpg=missing_jpg,
    )
