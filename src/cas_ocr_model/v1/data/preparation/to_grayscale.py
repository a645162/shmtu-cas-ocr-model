"""将彩色图转灰度图 + 二值化。"""

from __future__ import annotations

import os

import cv2

from ...helpers.filesystem import divide_files_into_processes


def _convert_one(input_path: str, output_dir: str) -> None:
    if not os.path.isfile(input_path):
        return
    if not input_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        return
    try:
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"[to_grayscale] skip unreadable: {input_path}")
            return
        image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)[1]
        cv2.imwrite(os.path.join(output_dir, os.path.basename(input_path)), image)
    except Exception as exc:  # noqa: BLE001
        print(f"[to_grayscale] error on {input_path}: {exc}")


def convert_to_grayscale(
    input_dir: str, output_dir: str, num_processes: int = 12
) -> None:
    """批量执行灰度化 + 二值化。"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def process_file(file_path: str) -> None:
        _convert_one(file_path, output_dir)

    divide_files_into_processes(
        input_dir, num_processes, process_file, include_subdir=True
    )


if __name__ == "__main__":
    convert_to_grayscale(
        input_dir="../../workdir/OriData/ori",
        output_dir="../../workdir/OriData/ori_gray",
    )
