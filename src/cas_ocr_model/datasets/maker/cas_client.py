"""CAS 三阶段流程: probe -> challenge -> submit, 落盘或丢弃."""
from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from shmtu_cas import EpayAuth

from .config import random_probe_account
from .ocr_backends import OcrBackend, OcrHit


@dataclass(slots=True)
class VerificationResult:
    status: Literal["correct", "incorrect", "failure", "error"]
    hit: OcrHit | None
    image_bytes: bytes | None = None
    variant: str = ""
    message: str = ""
    probe_username: str = ""


async def verify_one(
    backend: OcrBackend,
    auth: EpayAuth,
    log_prefix: str = "",
    log_q: Any = None,
) -> VerificationResult:
    def _log(level: str, msg: str) -> None:
        if log_q is None:
            return
        try:
            log_q.put_nowait({"level": level, "msg": f"{log_prefix} {msg}", "ts": time.time()})
        except Exception:  # noqa: BLE001
            pass

    try:
        probe = await auth.probe_login()
    except Exception as e:  # noqa: BLE001
        _log("error", f"probe failed: {e}")
        return VerificationResult(status="error", hit=None, message=str(e))
    if not probe.is_need_login:
        await auth.aclose()
        auth = EpayAuth()

    try:
        challenge = await auth.prepare_challenge()
    except Exception as e:  # noqa: BLE001
        _log("error", f"prepare_challenge failed: {e}")
        return VerificationResult(status="error", hit=None, message=str(e))

    try:
        hit = await backend.recognize(challenge.captcha_image)
    except Exception as e:  # noqa: BLE001
        _log("error", f"ocr failed: {e}")
        return VerificationResult(status="error", hit=None, image_bytes=challenge.captcha_image, message=str(e))

    if not hit.answer:
        return VerificationResult(
            status="failure",
            hit=hit,
            image_bytes=challenge.captcha_image,
            message="empty answer",
        )

    student_no, password = random_probe_account()
    try:
        result = await auth.submit_login(student_no, password, hit.answer, challenge.execution)
    except Exception as e:  # noqa: BLE001
        _log("error", f"submit failed: {e}")
        return VerificationResult(
            status="error",
            hit=hit,
            image_bytes=challenge.captcha_image,
            message=str(e),
            probe_username=student_no,
        )

    if result.is_password_error or result.is_success:
        return VerificationResult(
            status="correct",
            hit=hit,
            image_bytes=challenge.captcha_image,
            variant=result.variant,
            probe_username=student_no,
        )
    if result.is_validate_code_error:
        return VerificationResult(
            status="incorrect",
            hit=hit,
            image_bytes=challenge.captcha_image,
            variant=result.variant,
            probe_username=student_no,
        )
    return VerificationResult(
        status="failure",
        hit=hit,
        image_bytes=challenge.captcha_image,
        variant=result.variant,
        message=result.message,
        probe_username=student_no,
    )


async def collect_one(
    backend: OcrBackend,
    auth: EpayAuth,
    counter: Any,
    output_dir: Path,
    log_prefix: str,
    log_q: Any = None,
) -> Literal["saved", "rejected", "error"]:
    """单次采集闭环."""
    def _log(level: str, msg: str) -> None:
        if log_q is None:
            return
        try:
            log_q.put_nowait({"level": level, "msg": f"{log_prefix} {msg}", "ts": time.time()})
        except Exception:  # noqa: BLE001
            pass

    verification = await verify_one(backend, auth, log_prefix=log_prefix, log_q=log_q)
    if verification.status == "error":
        return "error"
    if verification.status != "correct" or verification.hit is None or verification.image_bytes is None:
        return "rejected"

    idx = counter.value
    counter.value = idx + 1
    stem = f"{idx:08d}"
    jpg_path = output_dir / f"{stem}.jpg"
    json_path = output_dir / f"{stem}.json"

    with tempfile.NamedTemporaryFile(
        delete=False, dir=output_dir, prefix=f".{stem}.", suffix=".jpg.tmp"
    ) as tmp_img:
        tmp_img.write(verification.image_bytes)
        tmp_img_path = Path(tmp_img.name)
    tmp_img_path.replace(jpg_path)

    payload = {
        "id": idx,
        "filename": jpg_path.name,
        "expression": verification.hit.expression,
        "answer": verification.hit.answer,
        "verification": "password_error_or_success",
        "probe_username": verification.probe_username,
        "backend": backend.name,
        "created_at": int(time.time()),
    }
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, dir=output_dir, prefix=f".{stem}.",
        suffix=".json.tmp", encoding="utf-8",
    ) as tmp_json:
        json.dump(payload, tmp_json, ensure_ascii=False, indent=2)
        tmp_json_path = Path(tmp_json.name)
    tmp_json_path.replace(json_path)

    _log(
        "info",
        f"saved #{stem}: expr='{verification.hit.expression}' answer={verification.hit.answer} "
        f"verify={verification.variant}",
    )
    return "saved"
