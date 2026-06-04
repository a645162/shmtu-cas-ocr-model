"""删除过白的空白图片（基本无信息量）。"""

from __future__ import annotations

import os

import cv2

from ...helpers.filesystem import divide_files_into_processes, get_all_files


def is_blank(image) -> bool:
    """判断图像是否几乎全白。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    threshold = cv2.bitwise_not(threshold)
    return cv2.countNonZero(threshold) == 0


def _process_file(file_path: str) -> None:
    image = cv2.imread(file_path)
    if image is None:
        print(f"[remove_blank] cannot read: {file_path}")
        return
    if is_blank(image):
        os.remove(file_path)


def remove_blank_images(input_dir: str, num_processes: int = 14) -> int:
    """多进程删除空白图，返回处理后剩余文件数。"""
    divide_files_into_processes(input_dir, num_processes, _process_file)
    remaining = get_all_files(input_dir, include_subdir=False)
    print(f"[remove_blank] remaining files: {len(remaining)}")
    return len(remaining)


if __name__ == "__main__":
    remove_blank_images("../../workdir/Spilt/ori_gray_div_last")
