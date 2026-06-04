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
    log_q: Any = None,
) -> Literal["saved", "rejected", "error"]:
    """单次采集闭环.

    日志不再直接 print (会与主进程 rich 进度条抢行), 改放 log_q 队列,
    由主进程统一渲染. log_q 为 None 时静默 (兼容旧调用).
    """
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
        return "error"
    if not probe.is_need_login:
        await auth.aclose()
        auth = EpayAuth()

    try:
        challenge = await auth.prepare_challenge()
    except Exception as e:  # noqa: BLE001
        _log("error", f"prepare_challenge failed: {e}")
        return "error"

    try:
        hit = await backend.recognize(challenge.captcha_image)
    except Exception as e:  # noqa: BLE001
        _log("error", f"ocr failed: {e}")
        return "error"

    if not hit.answer:
        return "rejected"

    student_no, password = random_probe_account()
    try:
        result = await auth.submit_login(
            student_no, password, hit.answer, challenge.execution
        )
    except Exception as e:  # noqa: BLE001
        _log("error", f"submit failed: {e}")
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

    _log(
        "info",
        f"saved #{stem}: expr='{hit.expression}' answer={hit.answer} "
        f"verify={result.variant}",
    )
    return "saved"
