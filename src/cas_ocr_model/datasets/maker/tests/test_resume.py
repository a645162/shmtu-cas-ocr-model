"""断点续采 (resume) 相关单测.

覆盖:
    * config.scan_existing_max_index 各种边界 (空目录 / 混合编号 / 忽略非 jpg / 忽略 .tmp / 忽略非数字)
    * config.build_arg_parser 的 --resume 开关语义 (store_true, 默认 False)
    * maker 包顶层导出的 scan_existing_max_index 与 config 同一函数 (兼容性)

运行: python -m pytest src/cas_ocr_model/datasets/maker/tests/ -v
路径注入由 conftest.py 负责.
"""
from __future__ import annotations

import pytest

from cas_ocr_model.datasets.maker.config import (
    build_arg_parser,
    scan_existing_max_index,
)
from cas_ocr_model.datasets.maker import scan_existing_max_index as maker_export


def test_scan_existing_max_index_empty_dir(tmp_path):
    assert scan_existing_max_index(tmp_path) == -1


def test_scan_existing_max_index_nonexistent_dir(tmp_path):
    assert scan_existing_max_index(tmp_path / "does-not-exist") == -1


def test_scan_existing_max_index_finds_max(tmp_path):
    (tmp_path / "00000070.jpg").write_bytes(b"")
    (tmp_path / "00000000.jpg").write_bytes(b"")
    (tmp_path / "00000500.jpg").write_bytes(b"")
    # 下一个可用编号 = max(0, 70, 500) + 1 = 501
    assert scan_existing_max_index(tmp_path) + 1 == 501


def test_scan_existing_max_index_ignores_non_jpg(tmp_path):
    (tmp_path / "00000010.jpg").write_bytes(b"")
    (tmp_path / "00000020.json").write_bytes(b"")  # json 不算
    (tmp_path / "00000030.png").write_bytes(b"")  # png 不算
    (tmp_path / "00000040.txt").write_bytes(b"")  # txt 不算
    assert scan_existing_max_index(tmp_path) == 10


def test_scan_existing_max_index_ignores_tmp_and_garbage(tmp_path):
    (tmp_path / "00000005.jpg").write_bytes(b"")
    # 原子写残留
    (tmp_path / ".00000006.jpg.tmp").write_bytes(b"")
    (tmp_path / ".00000007.json.tmp").write_bytes(b"")
    # 非数字 stem (7 位 / 9 位)
    (tmp_path / "1234567.jpg").write_bytes(b"")
    (tmp_path / "123456789.jpg").write_bytes(b"")
    # 字母 stem
    (tmp_path / "abcdefgh.jpg").write_bytes(b"")
    assert scan_existing_max_index(tmp_path) == 5


def test_scan_existing_max_index_does_not_raise_on_garbage(tmp_path):
    (tmp_path / "00000003.jpg").write_bytes(b"")
    (tmp_path / "..jpg").write_bytes(b"")  # 空 stem
    (tmp_path / "00.00.00.00.jpg").write_bytes(b"")  # 字母+数字
    assert scan_existing_max_index(tmp_path) == 3


def test_arg_parser_resume_default_false():
    args = build_arg_parser().parse_args(["--output", "/tmp/x", "--count", "10"])
    assert isinstance(args.resume, bool)
    assert args.resume is False


def test_arg_parser_resume_explicit_true():
    args = build_arg_parser().parse_args(
        ["--output", "/tmp/x", "--count", "10", "--resume"]
    )
    assert args.resume is True


def test_arg_parser_resume_is_store_true_flag():
    """--resume 是 store_true, 传等号值必须报错 (避免歧义)."""
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(
            ["--output", "/tmp/x", "--count", "10", "--resume=1"]
        )


def test_maker_package_exports_scan_existing_max_index():
    import cas_ocr_model.datasets.maker as maker

    assert "scan_existing_max_index" in maker.__all__
    # 顶层导出必须是 config 里的同一个函数 (兼容)
    assert maker_export is scan_existing_max_index


def test_resume_preserves_existing_data(tmp_path):
    """回归: scan_existing_max_index 不应删除/修改任何文件."""
    files_before = set(p.name for p in tmp_path.iterdir())
    for i in range(5):
        (tmp_path / f"{i:08d}.jpg").write_bytes(f"img{i}".encode())

    files_during = set(p.name for p in tmp_path.iterdir())
    max_idx = scan_existing_max_index(tmp_path)
    files_after = set(p.name for p in tmp_path.iterdir())

    assert max_idx == 4
    # scan 调用前后文件集合完全一致
    assert files_during == files_after
    # 跟调用前比, 只多了 5 个 jpg (写入发生在 scan 之前, 不属于 scan 的副作用)
    assert files_during - files_before == {
        "00000000.jpg",
        "00000001.jpg",
        "00000002.jpg",
        "00000003.jpg",
        "00000004.jpg",
    }


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
