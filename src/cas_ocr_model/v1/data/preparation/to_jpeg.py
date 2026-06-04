"""将任意格式图像重写为 JPEG（用于统一格式）。"""

from __future__ import annotations

import os

import cv2

from ...helpers.filesystem import divide_files_into_processes


def _to_jpeg(file_path: str) -> None:
    """单文件处理：读取后用同名 .jpg 写出。"""
    image = cv2.imread(file_path)
    if image is None:
        return
    directory = os.path.dirname(file_path)
    base_name_no_ext = os.path.splitext(os.path.basename(file_path))[0]
    cv2.imwrite(os.path.join(directory, base_name_no_ext + ".jpg"), image)


def convert_to_jpeg(input_dir: str, num_processes: int = 12) -> None:
    """将目录下所有图像以 JPEG 格式重新写出。"""
    divide_files_into_processes(
        input_dir, num_processes, _to_jpeg, include_subdir=True
    )


if __name__ == "__main__":
    convert_to_jpeg("../workdir/OriData/ori_gray")
