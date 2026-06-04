"""按关键点把验证码主体切成多张子图（数字 / 运算符 / 数字）。"""

from __future__ import annotations

import os
from typing import List

import cv2

from ...helpers.filesystem import create_dirs, divide_files_into_processes
from ...helpers.image import split_image_by_ratio

# 当前关键点列表（[0.25, 0.58, 0.75] 之类）
_current_key_points: List[float] = []
# 当前输出基础目录
_output_base_dir: str = ""


def _split_single(file_path: str) -> None:
    base_name = os.path.basename(file_path)
    base_name_no_ext, ext_name = os.path.splitext(base_name)

    image = cv2.imread(file_path)
    if image is None:
        print(f"[split_image] cannot read: {file_path}")
        return

    key_points = _current_key_points.copy()
    if not key_points:
        return
    key_points.sort()
    key_points.insert(0, 0.0)
    key_points.append(1.0)

    for i in range(len(key_points) - 1):
        segment = split_image_by_ratio(image, key_points[i], key_points[i + 1])
        output_dir = os.path.join(_output_base_dir, str(i))
        output_path = os.path.join(output_dir, f"{base_name_no_ext}_{i}{ext_name}")
        try:
            cv2.imwrite(output_path, segment)
        except Exception as exc:  # noqa: BLE001
            print(f"[split_image] error saving {output_path}: {exc}")


def _process_file(file_path: str) -> None:
    try:
        _split_single(file_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[split_image] error on {file_path}: {exc}")


def start_splitting(
    source_dir: str,
    output_dir: str,
    key_points: List[float],
    num_processes: int = 10,
) -> None:
    """把 source_dir 下的图像按 key_points 切成多段，写到 output_dir/{0,1,...}。"""
    global _current_key_points, _output_base_dir
    _current_key_points = key_points
    _output_base_dir = output_dir

    for i in range(len(key_points) + 1):
        create_dirs([os.path.join(output_dir, str(i))])

    divide_files_into_processes(source_dir, num_processes, _process_file)


if __name__ == "__main__":
    from ...configs.defaults import key_point_chs, key_point_symbol

    if len(key_point_symbol) == 3:
        start_splitting(
            "../../workdir/Classify/OriImg/symbol",
            "../workdir/Spilt/MainBody_symbol",
            key_point_symbol,
        )
        print("Split key point (symbol) finished!")
    else:
        print("Key point error (symbol)!")

    if len(key_point_chs) == 3:
        start_splitting(
            "../../workdir/Classify/OriImg/chs",
            "../workdir/Spilt/MainBody_chs",
            key_point_chs,
        )
        print("Split key point (chs) finished!")
    else:
        print("Key point error (chs)!")
