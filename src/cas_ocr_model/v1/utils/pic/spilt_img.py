import numpy as np


def spilt_img_by_ratio(
        image: np.ndarray,
        start_ratio: float = 0.8, end_ratio: float = 1
) -> np.ndarray:
    # 获取图像的宽度和高度
    height, width, _ = image.shape

    # 计算水平方向上的裁剪范围
    if start_ratio > end_ratio:
        start_ratio, end_ratio = end_ratio, start_ratio

    horizontal_start = int(width * start_ratio)
    horizontal_end = int(width * end_ratio)
    if end_ratio >= 1:
        horizontal_end = width

    horizontal_part = image.copy()[:, horizontal_start:horizontal_end]

    return horizontal_part
