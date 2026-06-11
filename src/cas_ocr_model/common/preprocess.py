"""训练与推理共用的验证码预处理."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

BINARIZE_MODES = (
    "min_channel_otsu",
    "gray_otsu",
    "adaptive",
    "fixed",
)


def decode_color_image(image_input: str | Path | bytes | bytearray | np.ndarray) -> np.ndarray | None:
    if isinstance(image_input, (str, Path)):
        arr = np.fromfile(str(image_input), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if isinstance(image_input, (bytes, bytearray)):
        arr = np.frombuffer(image_input, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if isinstance(image_input, np.ndarray):
        return image_input
    raise TypeError(f"unsupported input type: {type(image_input)}")


def _normalize_block_size(block_size: int) -> int:
    block_size = max(3, int(block_size))
    if block_size % 2 == 0:
        block_size += 1
    return block_size


def _foreground_score_from_color(img: np.ndarray) -> np.ndarray:
    # 白底接近 (255,255,255), 随机颜色字符至少会有一个通道明显变暗.
    return 255 - np.min(img, axis=2)


def binarize_captcha(
    img: np.ndarray,
    threshold: int = 200,
    binarize_mode: str = "min_channel_otsu",
    adaptive_block_size: int = 25,
    adaptive_c: int = 15,
) -> np.ndarray:
    if binarize_mode not in BINARIZE_MODES:
        raise ValueError(f"unknown binarize_mode={binarize_mode}, available={BINARIZE_MODES}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if binarize_mode == "fixed":
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        return binary

    if binarize_mode == "gray_otsu":
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        return binary

    if binarize_mode == "adaptive":
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        return cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            _normalize_block_size(adaptive_block_size),
            adaptive_c,
        )

    score = _foreground_score_from_color(img)
    blur = cv2.GaussianBlur(score, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary


def preprocess_captcha_to_tensor(
    img: np.ndarray | None,
    image_size: tuple[int, int],
    threshold: int = 200,
    binarize_mode: str = "min_channel_otsu",
    adaptive_block_size: int = 25,
    adaptive_c: int = 15,
) -> torch.Tensor:
    if img is None:
        return torch.zeros(1, image_size[0], image_size[1], dtype=torch.float32)

    binary = binarize_captcha(
        img,
        threshold=threshold,
        binarize_mode=binarize_mode,
        adaptive_block_size=adaptive_block_size,
        adaptive_c=adaptive_c,
    )
    resized = cv2.resize(binary, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(resized).float().unsqueeze(0) / 255.0
