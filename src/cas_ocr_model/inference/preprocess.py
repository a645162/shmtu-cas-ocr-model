"""推理预处理: 灰度 + 二值化 + resize + 归一化, 与 trainer/data.py 保持一致."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class CaptchaPreprocess:
    image_size_h: int = 64
    image_size_w: int = 192
    threshold: int = 200

    def __call__(self, image_bytes_or_path: str | Path | bytes) -> torch.Tensor:
        """支持三种输入:
            * 文件路径 (str / Path)
            * 原始图片字节 (bytes, e.g. HTTP response.content)
            * 已解码的 numpy 数组 (BGR)
        """
        if isinstance(image_bytes_or_path, (str, Path)):
            arr = np.fromfile(str(image_bytes_or_path), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        elif isinstance(image_bytes_or_path, (bytes, bytearray)):
            arr = np.frombuffer(image_bytes_or_path, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        elif isinstance(image_bytes_or_path, np.ndarray):
            img = image_bytes_or_path
        else:
            raise TypeError(f"unsupported input type: {type(image_bytes_or_path)}")

        if img is None:
            # 兜底: 全黑图
            return torch.zeros(1, self.image_size_h, self.image_size_w, dtype=torch.float32)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        resized = cv2.resize(binary, (self.image_size_w, self.image_size_h), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).float().unsqueeze(0) / 255.0
        return tensor


def build_preprocess(image_size_h: int = 64, image_size_w: int = 192, threshold: int = 200) -> CaptchaPreprocess:
    return CaptchaPreprocess(image_size_h=image_size_h, image_size_w=image_size_w, threshold=threshold)
