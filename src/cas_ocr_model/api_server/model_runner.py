from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from cas_ocr_model.inference import CaptchaInferencer, InferencerConfig, PyTorchBackend
from cas_ocr_model.v1.inference.predictor import load_models, predict_validate_code
from cas_ocr_model.v1.models.operator_enum import (
    OperatorEnum,
    get_operator_type_by_int,
    get_operator_type_str,
)

from .config import ApiServerConfig


@dataclass(slots=True)
class OcrPrediction:
    success: bool
    expression: str
    result: int
    equal_symbol: int
    operator: int
    digit1: int
    digit2: int
    error: str | None = None


class Predictor(Protocol):
    def predict(self, image_bytes: bytes) -> OcrPrediction: ...


def resolve_device(device_name: str) -> str:
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device_name


def _canonical_operator_id(symbol: str) -> int:
    return {"+": 0, "-": 2, "*": 4}[symbol]


def _default_v2_checkpoint() -> Path:
    candidates = [
        Path("runs"),
        Path("workdir"),
        Path("."),
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        matches = sorted(candidate.rglob("best.pt"))
        if matches:
            return matches[0].resolve()
    raise FileNotFoundError("Unable to find best.pt, please pass --checkpoint")


def _default_v1_model_dir() -> Path:
    candidate = Path("workdir") / "Models"
    if candidate.is_dir() and any(candidate.glob("*.pth")):
        return candidate.resolve()
    raise FileNotFoundError(
        "Unable to find v1 model directory, please pass --v1-model-dir"
    )


class V2Predictor:
    def __init__(self, config: ApiServerConfig) -> None:
        self.device = resolve_device(config.device)
        checkpoint = config.checkpoint.resolve() if config.checkpoint else _default_v2_checkpoint()
        backend = PyTorchBackend(checkpoint=checkpoint, device=self.device)
        infer_config = InferencerConfig(
            image_size_h=config.image_size_h,
            image_size_w=config.image_size_w,
            threshold=config.threshold,
            binarize_mode=config.binarize_mode,
            adaptive_block_size=config.adaptive_block_size,
            adaptive_c=config.adaptive_c,
            batch_size=config.batch_size,
        )
        self.inferencer = CaptchaInferencer(backend=backend, config=infer_config)
        self.checkpoint = checkpoint

    def predict(self, image_bytes: bytes) -> OcrPrediction:
        result = self.inferencer.predict_one(image_bytes)
        op_id = _canonical_operator_id(result.operator)
        return OcrPrediction(
            success=True,
            expression=result.expression,
            result=int(result.result or 0),
            equal_symbol=-1,
            operator=op_id,
            digit1=int(result.digit_left),
            digit2=int(result.digit_right),
            error=None,
        )


class V1Predictor:
    def __init__(self, config: ApiServerConfig) -> None:
        import cv2

        self.cv2 = cv2
        self.device = torch.device(resolve_device(config.device))
        self.model_dir = (
            config.v1_model_dir.resolve() if config.v1_model_dir else _default_v1_model_dir()
        )
        self.models = load_models(self.device, pth_dir=str(self.model_dir))

    def predict(self, image_bytes: bytes) -> OcrPrediction:
        image = self.cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8),
            self.cv2.IMREAD_COLOR,
        )
        if image is None:
            raise ValueError("Invalid image bytes")

        calc_result, _calc_str, equal_id, operator_id, digit_left, digit_right = (
            predict_validate_code(
                image,
                self.device,
                *self.models,
                print_result=False,
            )
        )
        canonical = get_operator_type_by_int(OperatorEnum(operator_id))
        operator_symbol = get_operator_type_str(canonical)
        expression = f"{digit_left}{operator_symbol}{digit_right}"
        return OcrPrediction(
            success=True,
            expression=expression,
            result=int(calc_result),
            equal_symbol=int(equal_id),
            operator=int(operator_id),
            digit1=int(digit_left),
            digit2=int(digit_right),
            error=None,
        )


def build_predictor(config: ApiServerConfig) -> Predictor:
    if config.model_kind == "v1":
        return V1Predictor(config)
    return V2Predictor(config)
