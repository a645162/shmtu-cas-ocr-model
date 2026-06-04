"""按比例从验证码图中裁出"最后一段"，用于训练等号区域。"""

from __future__ import annotations

import os

import cv2

from ...configs.defaults import equal_symbol_key_end, equal_symbol_key_start
from ...helpers.filesystem import create_dirs, divide_files_into_processes
from ...helpers.image import split_image_by_ratio


def _save_last_part(image_path: str, output_dir: str) -> None:
    image = cv2.imread(image_path)
    if image is None:
        print(f"[extract_last_segment] cannot read: {image_path}")
        return
    horizontal_part = split_image_by_ratio(
        image, equal_symbol_key_start, equal_symbol_key_end
    )
    resized = cv2.resize(horizontal_part, (224, 224))
    cv2.imwrite(os.path.join(output_dir, os.path.basename(image_path)), resized)


def extract_last_segment(
    input_dir: str, output_dir: str, num_processes: int = 10
) -> None:
    """多进程批量裁剪最后一段。"""
    create_dirs([output_dir])

    def process_file(file_path: str) -> None:
        _save_last_part(file_path, output_dir)

    divide_files_into_processes(input_dir, num_processes, process_file)


if __name__ == "__main__":
    extract_last_segment(
        input_dir="../workdir/OriData/ori_gray",
        output_dir="../../workdir/Spilt/ori_gray_div_last",
    )
