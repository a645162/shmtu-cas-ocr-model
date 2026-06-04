"""设备选择器：自动选 mps / cuda / cpu。"""

from __future__ import annotations

import platform

import torch


def get_recommended_device() -> torch.device:
    """优先 Metal (macOS) > CUDA > CPU。"""
    print("[device] checking...")
    if "Darwin" in platform.system() and torch.backends.mps.is_available():
        print("[device] Apple Metal backend available.")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("[device] CUDA available.")
        return torch.device("cuda")
    print("[device] falling back to CPU.")
    return torch.device("cpu")


if __name__ == "__main__":
    print("Device:", get_recommended_device())
