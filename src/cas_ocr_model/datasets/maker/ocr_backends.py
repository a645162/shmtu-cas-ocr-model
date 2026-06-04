"""推理后端抽象: Restful / Tcp / Pytorch (采集阶段使用)."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from shmtu_cas.captcha.http_ocr import CaptchaOcrHttp
from shmtu_cas.captcha.tcp_ocr import CaptchaOcr
from shmtu_cas.captcha.utils import get_expr_result

from .config import ensure_pytorch_weights


@dataclass
class OcrHit:
    expression: str  # "12+34=46"
    answer: str      # "46"


class OcrBackend(Protocol):
    name: str

    def warmup(self) -> None: ...

    async def recognize(self, image_bytes: bytes) -> OcrHit: ...


class RestfulBackend:
    name = "restful"

    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        self._client = CaptchaOcrHttp(base_url=base_url, timeout=timeout)

    def warmup(self) -> None:
        try:
            asyncio.run(self._client.health_check_async())
        except Exception as e:  # noqa: BLE001
            print(f"[ocr:restful] health check failed (ignored): {e}", flush=True)

    async def recognize(self, image_bytes: bytes) -> OcrHit:
        expr = await self._client.ocr_auto_retry_async(image_bytes, max_retries=2)
        return OcrHit(expression=expr, answer=get_expr_result(expr))


class TcpBackend:
    name = "tcp"

    def __init__(self, host: str, port: int) -> None:
        self._client = CaptchaOcr(host=host, port=port)

    def warmup(self) -> None:
        try:
            self._client.ocr_auto_retry(b"", max_retries=1)
        except Exception:
            pass

    async def recognize(self, image_bytes: bytes) -> OcrHit:
        expr = await asyncio.to_thread(self._client.ocr_auto_retry, image_bytes, 2)
        return OcrHit(expression=expr, answer=get_expr_result(expr))


class PytorchBackend:
    name = "pytorch"

    def __init__(self, weights_dir: Path) -> None:
        self._weights = ensure_pytorch_weights(weights_dir)
        from cas_ocr_model.v1.configs.paths import pth_save_dir_path  # type: ignore
        target = Path(pth_save_dir_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.mkdir(parents=True, exist_ok=True)
        for label, src in self._weights.items():
            dst = target / src.name
            if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                dst.write_bytes(src.read_bytes())

        import cv2  # type: ignore
        import numpy as np  # type: ignore
        from cas_ocr_model.v1.data_modules.device import get_recommended_device  # type: ignore
        from cas_ocr_model.v1.inference.predictor import (  # type: ignore
            load_models,
            predict_validate_code,
        )

        self._device = get_recommended_device()
        self._models = load_models(self._device, pth_dir=str(target))
        self._predict = predict_validate_code
        self._cv2 = cv2
        self._np = np

    def warmup(self) -> None:
        blank = self._np.full((30, 100, 3), 255, dtype=self._np.uint8)
        try:
            self._predict(blank, self._device, *self._models, print_result=False)
            print(f"[ocr:pytorch] warmup ok on {self._device}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[ocr:pytorch] warmup failed (ignored): {e}", flush=True)

    async def recognize(self, image_bytes: bytes) -> OcrHit:
        return await asyncio.to_thread(self._recognize_sync, image_bytes)

    def _recognize_sync(self, image_bytes: bytes) -> OcrHit:
        arr = self._np.frombuffer(image_bytes, dtype=self._np.uint8)
        img = self._cv2.imdecode(arr, self._cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("captcha image decode failed")
        calc_result, calc_str, *_ = self._predict(
            img, self._device, *self._models, print_result=False
        )
        normalized = calc_str.replace(" ", "")
        return OcrHit(expression=normalized, answer=str(calc_result))


def build_backend(args) -> OcrBackend:
    if args.backend == "restful":
        return RestfulBackend(args.ocr_url, timeout=args.ocr_timeout)
    if args.backend == "tcp":
        return TcpBackend(args.ocr_host, args.ocr_port)
    if args.backend == "pytorch":
        return PytorchBackend(Path(args.weights_dir))
    raise ValueError(f"unknown backend: {args.backend}")
