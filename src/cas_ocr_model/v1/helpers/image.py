"""图像工具：matplotlib 显示 + 按比例裁剪。"""

from __future__ import annotations

import cv2
import numpy as np


def show_opencv_image_by_matplotlib(image) -> None:
    """用 matplotlib 显示 BGR 图（自动转 RGB）。"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    import matplotlib.pyplot as plt

    plt.imshow(image_rgb)
    plt.axis("off")
    plt.show()


def split_image_by_ratio(
    image: np.ndarray,
    start_ratio: float = 0.8,
    end_ratio: float = 1.0,
) -> np.ndarray:
    """按宽度比例水平裁剪图像。start_ratio > end_ratio 时会自动交换。"""
    if start_ratio > end_ratio:
        start_ratio, end_ratio = end_ratio, start_ratio
    height, width, _ = image.shape
    horizontal_start = int(width * start_ratio)
    horizontal_end = width if end_ratio >= 1 else int(width * end_ratio)
    return image.copy()[:, horizontal_start:horizontal_end]
