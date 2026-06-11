"""推理后端与本地模型抽象: Restful / Tcp / PyTorch."""
from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from shmtu_cas.captcha.http_ocr import CaptchaOcrHttp
from shmtu_cas.captcha.tcp_ocr import CaptchaOcr
from shmtu_cas.captcha.utils import get_expr_result

from .config import ensure_pytorch_weights


@dataclass(slots=True)
class OcrHit:
    expression: str
    answer: str


class OcrBackend(Protocol):
    name: str

    def warmup(self) -> None: ...

    async def recognize(self, image_bytes: bytes) -> OcrHit: ...


class OcrModel(Protocol):
    name: str

    def warmup(self) -> None: ...

    def predict(self, image_bytes: bytes) -> OcrHit: ...


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
        with contextlib.suppress(Exception):
            self._client.ocr_auto_retry(b"", max_retries=1)

    async def recognize(self, image_bytes: bytes) -> OcrHit:
        expr = await asyncio.to_thread(self._client.ocr_auto_retry, image_bytes, 2)
        return OcrHit(expression=expr, answer=get_expr_result(expr))


class BasePytorchModel:
    name = "pytorch"

    def warmup(self) -> None:
        raise NotImplementedError

    def predict(self, image_bytes: bytes) -> OcrHit:
        raise NotImplementedError


class PytorchV1Model(BasePytorchModel):
    name = "pytorch_v1"

    def __init__(self, weights_dir: Path) -> None:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        from cas_ocr_model.v1.configs.paths import pth_save_dir_path  # type: ignore
        from cas_ocr_model.v1.data_modules.device import get_recommended_device  # type: ignore
        from cas_ocr_model.v1.inference.predictor import (  # type: ignore
            load_models,
            predict_validate_code,
        )

        self._weights = ensure_pytorch_weights(weights_dir)
        target = Path(pth_save_dir_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.mkdir(parents=True, exist_ok=True)
        for _label, src in self._weights.items():
            dst = target / src.name
            if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                dst.write_bytes(src.read_bytes())

        self._device = get_recommended_device()
        self._models = load_models(self._device, pth_dir=str(target))
        self._predict_impl = predict_validate_code
        self._cv2 = cv2
        self._np = np

    def warmup(self) -> None:
        blank = self._np.full((30, 100, 3), 255, dtype=self._np.uint8)
        try:
            self._predict_impl(blank, self._device, *self._models, print_result=False)
            print(f"[ocr:{self.name}] warmup ok on {self._device}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[ocr:{self.name}] warmup failed (ignored): {e}", flush=True)

    def predict(self, image_bytes: bytes) -> OcrHit:
        arr = self._np.frombuffer(image_bytes, dtype=self._np.uint8)
        img = self._cv2.imdecode(arr, self._cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("captcha image decode failed")
        calc_result, calc_str, *_ = self._predict_impl(
            img, self._device, *self._models, print_result=False
        )
        return OcrHit(expression=calc_str.replace(" ", ""), answer=str(calc_result))


class PytorchV2Model(BasePytorchModel):
    name = "pytorch_v2"

    def __init__(self, checkpoint: Path | None = None) -> None:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        import torch  # type: ignore
        from cas_ocr_model.inference import CaptchaInferencer, InferencerConfig, PyTorchBackend

        self._cv2 = cv2
        self._np = np
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._checkpoint = checkpoint.resolve() if checkpoint else self._find_default_checkpoint()
        backend = PyTorchBackend(checkpoint=self._checkpoint, device=self._device)
        self._inferencer = CaptchaInferencer(backend=backend, config=InferencerConfig())

    def _find_default_checkpoint(self) -> Path:
        from cas_ocr_model.model import inspect_checkpoint

        for candidate in (Path("runs"), Path("workdir"), Path(".")):
            if not candidate.exists():
                continue
            for pattern in ("best.pt", "*.trislot_decoder.v*.pt", "*.pt"):
                matches = sorted(candidate.rglob(pattern))
                for match in matches:
                    if match.name == "last.pt":
                        continue
                    try:
                        inspect_checkpoint(match)
                    except Exception:  # noqa: BLE001
                        continue
                    return match.resolve()
        raise FileNotFoundError("unable to find v2 checkpoint, please set --checkpoint")

    def warmup(self) -> None:
        blank = self._np.full((30, 100, 3), 255, dtype=self._np.uint8)
        ok, encoded = self._cv2.imencode(".png", blank)
        if not ok:
            return
        try:
            self.predict(encoded.tobytes())
            print(f"[ocr:{self.name}] warmup ok on {self._device}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[ocr:{self.name}] warmup failed (ignored): {e}", flush=True)

    def predict(self, image_bytes: bytes) -> OcrHit:
        result = self._inferencer.predict_one(image_bytes)
        return OcrHit(
            expression=result.expression,
            answer=str(result.result) if result.result is not None else "",
        )


class ModelBackend:
    def __init__(self, model: OcrModel) -> None:
        self._model = model
        self.name = model.name

    def warmup(self) -> None:
        self._model.warmup()

    async def recognize(self, image_bytes: bytes) -> OcrHit:
        return await asyncio.to_thread(self._model.predict, image_bytes)


LOCAL_MODEL_BUILDERS = {
    "pytorch_v1": PytorchV1Model,
    "pytorch_v2": PytorchV2Model,
}


def build_model(name: str, *, weights_dir: Path, checkpoint: Path | None = None) -> OcrModel:
    if name == "pytorch_v1":
        return PytorchV1Model(weights_dir)
    if name == "pytorch_v2":
        return PytorchV2Model(checkpoint=checkpoint)
    raise ValueError(f"unknown model: {name}")


def build_backend(args) -> OcrBackend:
    if args.backend == "restful":
        return RestfulBackend(args.ocr_url, timeout=args.ocr_timeout)
    if args.backend == "tcp":
        return TcpBackend(args.ocr_host, args.ocr_port)
    if args.backend == "pytorch":
        model_name = getattr(args, "model_name", "pytorch_v1")
        checkpoint_arg = getattr(args, "checkpoint", None)
        checkpoint = Path(checkpoint_arg).resolve() if checkpoint_arg else None
        return ModelBackend(
            build_model(
                model_name,
                weights_dir=Path(args.weights_dir),
                checkpoint=checkpoint,
            )
        )
    raise ValueError(f"unknown backend: {args.backend}")
