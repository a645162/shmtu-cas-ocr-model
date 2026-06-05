"""推理预处理: 与 trainer/data.py 保持一致."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from cas_ocr_model.preprocess_ops import decode_color_image, preprocess_captcha_to_tensor


@dataclass(frozen=True)
class CaptchaPreprocess:
    image_size_h: int = 64
    image_size_w: int = 192
    threshold: int = 200
    binarize_mode: str = "min_channel_otsu"
    adaptive_block_size: int = 25
    adaptive_c: int = 15

    def __call__(self, image_bytes_or_path: str | Path | bytes) -> torch.Tensor:
        """支持三种输入:
            * 文件路径 (str / Path)
            * 原始图片字节 (bytes, e.g. HTTP response.content)
            * 已解码的 numpy 数组 (BGR)
        """
        img = decode_color_image(image_bytes_or_path)
        return preprocess_captcha_to_tensor(
            img,
            image_size=(self.image_size_h, self.image_size_w),
            threshold=self.threshold,
            binarize_mode=self.binarize_mode,
            adaptive_block_size=self.adaptive_block_size,
            adaptive_c=self.adaptive_c,
        )


def build_preprocess(
    image_size_h: int = 64,
    image_size_w: int = 192,
    threshold: int = 200,
    binarize_mode: str = "min_channel_otsu",
    adaptive_block_size: int = 25,
    adaptive_c: int = 15,
) -> CaptchaPreprocess:
    return CaptchaPreprocess(
        image_size_h=image_size_h,
        image_size_w=image_size_w,
        threshold=threshold,
        binarize_mode=binarize_mode,
        adaptive_block_size=adaptive_block_size,
        adaptive_c=adaptive_c,
    )
