"""ONNX Runtime 推理后端.

加载 trainer/export.py 导出的 model.onnx, 单图/批量推理. 适合生产部署 (C++/Rust 之外的环境).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import onnxruntime as ort
except ImportError as e:  # pragma: no cover
    raise RuntimeError(
        "OnnxBackend 需要 onnxruntime, 请先 pip install onnxruntime"
    ) from e

from cas_ocr_model.trainer.config import (
    DIGIT_LABELS,
    NUM_DIGIT_CLASSES,
    NUM_OPERATOR_CLASSES,
    OPERATOR_LABELS,
)


class OnnxBackend:
    """ONNX Runtime 推理后端.

    用法:
        backend = OnnxBackend("model.onnx", device="cpu")
        logits = backend.infer(image_tensor)
    """

    def __init__(
        self,
        onnx_path: str | Path,
        device: str = "cpu",
        intra_op_num_threads: Optional[int] = None,
    ) -> None:
        sess_options = ort.SessionOptions()
        if intra_op_num_threads is not None:
            sess_options.intra_op_num_threads = intra_op_num_threads
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(
            str(onnx_path), sess_options=sess_options, providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name
        # 输出名固定 (与 trainer/export.py 对齐):
        #   digit_left_logits, operator_logits, digit_right_logits
        self.output_names = [o.name for o in self.session.get_outputs()]

    def infer(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """image: (B, 1, H, W) float32 in [0, 1], numpy 数组.

        返回 dict, 3 个 logits 数组.
        """
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        results = self.session.run(self.output_names, {self.input_name: image})
        return {name: arr for name, arr in zip(self.output_names, results)}

    @property
    def label_dict(self) -> dict[str, list[str]]:
        return {"digit_labels": DIGIT_LABELS, "operator_labels": OPERATOR_LABELS}
