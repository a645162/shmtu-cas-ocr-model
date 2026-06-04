"""CAS 三阶段流程: probe -> challenge -> submit, 落盘或丢弃.

与 tauri/captcha.rs 等价:
    * PasswordError / Success -> 验证码正确, 落盘 jpg + json
    * ValidateCodeError / Failure / 异常 -> 丢弃
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

from shmtu_cas import EpayAuth

from .config import random_probe_account
from .ocr_backends import OcrBackend


async def collect_one(
    backend: OcrBackend,
    auth: EpayAuth,
    counter: Any,
    output_dir: Path,
    log_prefix: str,
) -> Literal["saved", "rejected", "error"]:
    """单次采集闭环."""
    try:
        probe = await auth.probe_login()
    except Exception as e:  # noqa: BLE001
        print(f"{log_prefix} probe failed: {e}", flush=True)
        return "error"
    if not probe.is_need_login:
        await auth.aclose()
        auth = EpayAuth()

    try:
        challenge = await auth.prepare_challenge()
    except Exception as e:  # noqa: BLE001
        print(f"{log_prefix} prepare_challenge failed: {e}", flush=True)
        return "error"

    try:
        hit = await backend.recognize(challenge.captcha_image)
    except Exception as e:  # noqa: BLE001
        print(f"{log_prefix} ocr failed: {e}", flush=True)
        return "error"

    if not hit.answer:
        return "rejected"

    student_no, password = random_probe_account()
    try:
        result = await auth.submit_login(
            student_no, password, hit.answer, challenge.execution
        )
    except Exception as e:  # noqa: BLE001
        print(f"{log_prefix} submit failed: {e}", flush=True)
        return "error"

    if not (result.is_password_error or result.is_success):
        return "rejected"

    idx = counter.value
    counter.value = idx + 1
    stem = f"{idx:08d}"
    jpg_path = output_dir / f"{stem}.jpg"
    json_path = output_dir / f"{stem}.json"

    with tempfile.NamedTemporaryFile(
        delete=False, dir=output_dir, prefix=f".{stem}.", suffix=".jpg.tmp"
    ) as tmp_img:
        tmp_img.write(challenge.captcha_image)
        tmp_img_path = Path(tmp_img.name)
    tmp_img_path.replace(jpg_path)

    payload = {
        "id": idx,
        "filename": jpg_path.name,
        "expression": hit.expression,
        "answer": hit.answer,
        "verification": "password_error_or_success",
        "probe_username": student_no,
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

    print(
        f"{log_prefix} saved #{stem}: expr='{hit.expression}' answer={hit.answer} "
        f"verify={result.variant}",
        flush=True,
    )
    return "saved"
