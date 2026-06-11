"""ONNX Runtime 推理后端."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from cas_ocr_model.common.console import tag_print
from cas_ocr_model.trainer.config import DIGIT_LABELS, OPERATOR_LABELS

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover - 依赖取决于用户环境
    ort = None
    _ORT_IMPORT_ERROR = exc
else:
    _ORT_IMPORT_ERROR = None


HEAD_NAMES = ("digit_left_logits", "operator_logits", "digit_right_logits")


class OnnxBackend:
    """本地 ONNX Runtime 推理后端.

    当前默认只走 CPUExecutionProvider，便于和 PyTorch CPU / ncnn CPU 对齐测试。
    """

    def __init__(
        self,
        onnx_path: str | Path,
        providers: list[str] | None = None,
        device: str = "cpu",
    ) -> None:
        if ort is None:
            raise ModuleNotFoundError(
                "OnnxBackend 不可用: 缺少依赖 `onnxruntime`"
            ) from _ORT_IMPORT_ERROR
        if device != "cpu":
            raise ValueError("OnnxBackend 当前仅支持 device=cpu")

        self.device = "cpu"
        self.onnx_path = str(Path(onnx_path).expanduser().resolve())
        self.providers = providers or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(self.onnx_path, providers=self.providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_type = self.session.get_inputs()[0].type
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.output_name_map = self._build_output_name_map(self.output_names)

        tag_print("onnx-backend", f"model      = {self.onnx_path}")
        tag_print("onnx-backend", f"provider   = {self.providers}")
        tag_print("onnx-backend", f"input      = {self.input_name} ({self.input_type})")
        tag_print("onnx-backend", f"outputs    = {self.output_names}")

    @staticmethod
    def _build_output_name_map(output_names: list[str]) -> dict[str, str]:
        if all(head_name in output_names for head_name in HEAD_NAMES):
            return {head_name: head_name for head_name in HEAD_NAMES}
        if len(output_names) < len(HEAD_NAMES):
            raise ValueError(
                f"ONNX 输出数量不足: 期望至少 {len(HEAD_NAMES)} 个, 实际 {len(output_names)}"
            )
        return {
            head_name: output_name
            for head_name, output_name in zip(HEAD_NAMES, output_names)
        }

    def infer(self, image: torch.Tensor) -> dict[str, np.ndarray]:
        array = image.detach().cpu().numpy()
        input_array = array.astype(
            np.float16 if "float16" in self.input_type else np.float32,
            copy=False,
        )
        outputs = self.session.run(
            [self.output_name_map[head_name] for head_name in HEAD_NAMES],
            {self.input_name: input_array},
        )
        return {
            head_name: np.asarray(output, dtype=np.float32)
            for head_name, output in zip(HEAD_NAMES, outputs)
        }

    @property
    def label_dict(self) -> dict[str, list[str]]:
        return {"digit_labels": DIGIT_LABELS, "operator_labels": OPERATOR_LABELS}
