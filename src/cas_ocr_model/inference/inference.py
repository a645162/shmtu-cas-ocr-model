"""统一推理引擎.

CaptchaInferencer 接受任意 backend (PyTorch / ONNX / 未来 TensorRT),
对外暴露:
    * predict_one(image_input)  - 单图, 返回 InferenceResult
    * predict_batch(image_list) - 批量, 返回 List[InferenceResult]
    * predict_dir(dir, limit)   - 目录批量, 返回 List[(image_name, InferenceResult)]

InferenceResult 字段:
    digit_left:  str   "1"
    operator:    str   "+"
    digit_right: str   "2"
    expression:  str   "1+2"      (digit_left + operator + digit_right)
    result:      int   3          (表达式的算式结果, 若不可计算则为 None)
    confidence:  float 0.998      (3 个 head 最低置信度)
    softmax:     dict  {"digit_left": [...], "operator": [...], "digit_right": [...]}
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from .preprocess import CaptchaPreprocess, build_preprocess

# ----------------------------------------------------------------------------
# Backend 协议
# ----------------------------------------------------------------------------


class InferBackend(Protocol):
    """后端接口: 接受 (B, 1, H, W) float32 tensor, 返回 dict[head] -> (B, num_classes) ndarray."""

    def infer(self, image: torch.Tensor) -> dict[str, np.ndarray]: ...

    @property
    def label_dict(self) -> dict[str, list[str]]: ...


# ----------------------------------------------------------------------------
# 配置 / 结果
# ----------------------------------------------------------------------------


@dataclass
class InferencerConfig:
    image_size_h: int = 64
    image_size_w: int = 192
    threshold: int = 200
    binarize_mode: str = "min_channel_otsu"
    adaptive_block_size: int = 25
    adaptive_c: int = 15
    batch_size: int = 32


@dataclass
class InferenceResult:
    digit_left: str
    operator: str
    digit_right: str
    expression: str
    result: int | None
    confidence: float
    softmax: dict[str, list[float]] = field(default_factory=dict)


# ----------------------------------------------------------------------------
# 算式求值
# ----------------------------------------------------------------------------


def _safe_eval(d1: int, op: str, d2: int) -> int | None:
    try:
        if op == "+":
            return d1 + d2
        if op == "-":
            return d1 - d2
        if op == "*":
            return d1 * d2
    except Exception:
        return None
    return None


# ----------------------------------------------------------------------------
# 主类
# ----------------------------------------------------------------------------


class CaptchaInferencer:
    """统一推理入口."""

    def __init__(
        self,
        backend: InferBackend,
        config: InferencerConfig | None = None,
        preprocess: CaptchaPreprocess | None = None,
    ) -> None:
        self.backend = backend
        self.config = config or InferencerConfig()
        self.preprocess = preprocess or build_preprocess(
            self.config.image_size_h,
            self.config.image_size_w,
            self.config.threshold,
            self.config.binarize_mode,
            self.config.adaptive_block_size,
            self.config.adaptive_c,
        )
        self.digit_labels = backend.label_dict["digit_labels"]
        self.operator_labels = backend.label_dict["operator_labels"]

    # ---- 单图 / 批量 ----

    def predict_one(self, image_input: str | Path | bytes | np.ndarray) -> InferenceResult:
        tensor = self.preprocess(image_input)  # (1, H, W)
        logits = self.backend.infer(tensor.unsqueeze(0))  # 加 batch 维
        results = self._logits_to_results(logits)
        return results[0]

    def predict_batch(self, image_inputs: Sequence[str | Path | bytes | np.ndarray]) -> list[InferenceResult]:
        out: list[InferenceResult] = []
        for start in range(0, len(image_inputs), self.config.batch_size):
            chunk = list(image_inputs[start : start + self.config.batch_size])
            tensors = [self.preprocess(x) for x in chunk]
            batch = torch.stack(tensors, dim=0)
            logits = self.backend.infer(batch)
            out.extend(self._logits_to_results(logits))
        return out

    def predict_dir(
        self,
        directory: str | Path,
        pattern: str = "*.jpg",
        limit: int | None = None,
    ) -> list[tuple[str, InferenceResult]]:
        paths = sorted(Path(directory).glob(pattern))
        if limit is not None:
            paths = paths[:limit]
        if not paths:
            return []
        results = self.predict_batch(list(paths))
        return [(p.name, r) for p, r in zip(paths, results)]

    # ---- 内部 ----

    def _logits_to_results(self, logits: dict[str, np.ndarray]) -> list[InferenceResult]:
        dl = logits["digit_left_logits"]    # (B, 10)
        op = logits["operator_logits"]      # (B, 3)
        dr = logits["digit_right_logits"]   # (B, 10)

        # softmax
        def sm(x: np.ndarray) -> np.ndarray:
            x = x - x.max(axis=-1, keepdims=True)
            e = np.exp(x)
            return e / e.sum(axis=-1, keepdims=True)

        p_dl, p_op, p_dr = sm(dl), sm(op), sm(dr)

        idx_dl = p_dl.argmax(axis=-1)
        idx_op = p_op.argmax(axis=-1)
        idx_dr = p_dr.argmax(axis=-1)

        out: list[InferenceResult] = []
        for i in range(dl.shape[0]):
            d_l = self.digit_labels[idx_dl[i]]
            o = self.operator_labels[idx_op[i]]
            d_r = self.digit_labels[idx_dr[i]]
            conf = float(min(p_dl[i, idx_dl[i]], p_op[i, idx_op[i]], p_dr[i, idx_dr[i]]))
            try:
                result = _safe_eval(int(d_l), o, int(d_r))
            except Exception:
                result = None
            out.append(
                InferenceResult(
                    digit_left=d_l,
                    operator=o,
                    digit_right=d_r,
                    expression=f"{d_l}{o}{d_r}",
                    result=result,
                    confidence=conf,
                    softmax={
                        "digit_left": p_dl[i].tolist(),
                        "operator": p_op[i].tolist(),
                        "digit_right": p_dr[i].tolist(),
                    },
                )
            )
        return out
