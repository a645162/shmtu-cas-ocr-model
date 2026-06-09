"""ncnn Python 推理后端."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from cas_ocr_model.common.console import tag_print
from cas_ocr_model.trainer.config import DIGIT_LABELS, OPERATOR_LABELS

try:
    import ncnn
except ImportError as exc:  # pragma: no cover - 依赖取决于用户环境
    ncnn = None
    _NCNN_IMPORT_ERROR = exc
else:
    _NCNN_IMPORT_ERROR = None


HEAD_NAMES = ("digit_left_logits", "operator_logits", "digit_right_logits")


class NcnnBackend:
    """本地 ncnn 推理后端.

    当前使用 Python binding，默认按单样本循环推理，再在 batch 维上拼回输出。
    """

    def __init__(
        self,
        param_path: str | Path,
        bin_path: str | Path,
        input_name: str | None = None,
        output_names: tuple[str, str, str] | list[str] | None = None,
        device: str = "cpu",
    ) -> None:
        if ncnn is None:
            raise ModuleNotFoundError(
                "NcnnBackend 不可用: 缺少依赖 `ncnn`"
            ) from _NCNN_IMPORT_ERROR
        if device != "cpu":
            raise ValueError("NcnnBackend 当前仅支持 device=cpu")

        self.device = "cpu"
        self.param_path = str(Path(param_path).expanduser().resolve())
        self.bin_path = str(Path(bin_path).expanduser().resolve())
        self.net = ncnn.Net()
        ret = self.net.load_param(self.param_path)
        if ret != 0:
            raise RuntimeError(f"加载 ncnn param 失败: {self.param_path}, ret={ret}")
        ret = self.net.load_model(self.bin_path)
        if ret != 0:
            raise RuntimeError(f"加载 ncnn bin 失败: {self.bin_path}, ret={ret}")

        available_input_names = list(self.net.input_names())
        available_output_names = list(self.net.output_names())
        self.input_name = input_name or available_input_names[0]
        if output_names is None:
            if len(available_output_names) < len(HEAD_NAMES):
                raise ValueError(
                    f"ncnn 输出数量不足: 期望至少 {len(HEAD_NAMES)} 个, 实际 {len(available_output_names)}"
                )
            self.output_names = available_output_names[: len(HEAD_NAMES)]
        else:
            self.output_names = list(output_names)
            if len(self.output_names) != len(HEAD_NAMES):
                raise ValueError(
                    f"output_names 数量错误: 期望 {len(HEAD_NAMES)} 个, 实际 {len(self.output_names)}"
                )

        tag_print("ncnn-backend", f"param      = {self.param_path}")
        tag_print("ncnn-backend", f"bin        = {self.bin_path}")
        tag_print("ncnn-backend", f"input      = {self.input_name}")
        tag_print("ncnn-backend", f"outputs    = {self.output_names}")

    def infer(self, image: torch.Tensor) -> dict[str, np.ndarray]:
        batch = np.ascontiguousarray(image.detach().cpu().numpy(), dtype=np.float32)
        outputs: dict[str, list[np.ndarray]] = {head_name: [] for head_name in HEAD_NAMES}

        for sample in batch:
            extractor = self.net.create_extractor()
            mat = ncnn.Mat(np.ascontiguousarray(sample, dtype=np.float32))
            ret = extractor.input(self.input_name, mat)
            if ret != 0:
                raise RuntimeError(
                    f"ncnn extractor.input 失败: input_name={self.input_name}, ret={ret}"
                )

            for head_name, blob_name in zip(HEAD_NAMES, self.output_names):
                ret, out = extractor.extract(blob_name)
                if ret != 0:
                    raise RuntimeError(
                        f"ncnn extractor.extract 失败: blob_name={blob_name}, ret={ret}"
                    )
                outputs[head_name].append(np.asarray(out.numpy(), dtype=np.float32))

        return {
            head_name: np.stack(head_outputs, axis=0)
            for head_name, head_outputs in outputs.items()
        }

    @property
    def label_dict(self) -> dict[str, list[str]]:
        return {"digit_labels": DIGIT_LABELS, "operator_labels": OPERATOR_LABELS}
