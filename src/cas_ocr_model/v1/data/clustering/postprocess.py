"""聚类后处理：把原图按聚类结果拷贝到新目录。"""

from __future__ import annotations

import os
import shutil

from ...helpers.filesystem import create_dirs, divide_files_into_processes

_current_source_dir: str = ""
_current_output_dir: str = ""


def _process_file(file_path: str) -> None:
    base_name = os.path.basename(file_path)
    source_path = os.path.join(_current_source_dir, base_name)
    if not os.path.exists(source_path):
        return
    output_path = os.path.join(_current_output_dir, base_name)
    shutil.copyfile(source_path, output_path)


def copy_by_cluster_result(
    source_dir: str,
    classify_dir: str,
    output_dir: str,
    num_processes: int = 10,
) -> None:
    """把 classify_dir 下所有文件名对应在 source_dir 的文件复制到 output_dir。"""
    create_dirs([output_dir])

    global _current_source_dir, _current_output_dir
    _current_source_dir = source_dir
    _current_output_dir = output_dir

    divide_files_into_processes(
        classify_dir, num_processes, _process_file, include_subdir=False
    )


if __name__ == "__main__":
    copy_by_cluster_result(
        "../../workdir/OriData/ori_gray",
        "../workdir/resnet18/equal_symbol/a",
        "../workdir/Classify/OriImg/a",
    )
    copy_by_cluster_result(
        "../../workdir/OriData/ori_gray",
        "../workdir/resnet18/equal_symbol/b",
        "../workdir/Classify/OriImg/b",
    )
