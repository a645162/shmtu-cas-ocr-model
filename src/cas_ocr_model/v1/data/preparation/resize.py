"""将图像统一缩放到 72x72（与最后一段裁剪后的尺寸一致）。"""

from __future__ import annotations

import cv2

from ...helpers.filesystem import create_dirs, divide_files_into_processes


def _process_file(file_path: str) -> None:
    image = cv2.imread(file_path)
    if image is None:
        print(f"[resize] cannot read: {file_path}")
        return
    image = cv2.resize(image, (72, 72))
    cv2.imwrite(file_path, image)


def resize_images(input_dir: str, num_processes: int = 14) -> None:
    """原地将所有图像 resize 到 72x72。"""
    create_dirs([input_dir])

    def process_file(file_path: str) -> None:
        _process_file(file_path)

    divide_files_into_processes(input_dir, num_processes, process_file)


if __name__ == "__main__":
    resize_images("../workdir/ori_gray_div_last")
