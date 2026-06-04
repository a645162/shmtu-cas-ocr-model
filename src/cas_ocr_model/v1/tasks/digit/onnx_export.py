"""数字分类：导出 ONNX。"""

from __future__ import annotations

import os

from ...configs.defaults import model_digit_type, pth_save_dir_path
from ...configs.model import get_pth_name
from ...training.onnx_exporter import export_to_onnx


def export_digit_to_onnx(save_dir: str = pth_save_dir_path) -> str:
    pth = os.path.join(
        save_dir, get_pth_name(model_digit_type, "digit", "latest")
    )
    return export_to_onnx(
        model_type=model_digit_type,
        output_features_count=10,
        pth_path=pth,
    )


if __name__ == "__main__":
    export_digit_to_onnx()
