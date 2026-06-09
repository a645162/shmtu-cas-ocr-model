#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../env.sh"

CONFIG="${CONFIG:-$SHMTU_SRC/cas_ocr_model/trainer/configs/8gpu_ddp.yaml}"
if [ -z "${SHMTU_PROFILE_NAME:-}" ]; then
    SHMTU_PROFILE_NAME="$(basename "${CONFIG%.*}")"
    export SHMTU_PROFILE_NAME
    export SHMTU_PROFILE_DIR="$SHMTU_RUNS_ROOT/$SHMTU_PROFILE_NAME"
fi

RUN_DIR="${RUN_DIR:-$(bash "$SCRIPT_DIR/../run_path.sh" resolve)}"
EXPORT_ROOT="${EXPORT_ROOT:-$RUN_DIR/export}"
DIGEST_FILE="${DIGEST_FILE:-$EXPORT_ROOT/SHA256SUMS.txt}"

if command -v sha256sum >/dev/null 2>&1; then
    HASH_CMD="sha256sum"
elif command -v shasum >/dev/null 2>&1; then
    HASH_CMD="shasum -a 256"
else
    echo "[digest] 缺少 sha256sum 或 shasum"
    exit 1
fi

mkdir -p "$EXPORT_ROOT"

export EXPORT_ROOT DIGEST_FILE HASH_CMD
"$SHMTU_PYTHON" - <<'PY'
from __future__ import annotations

import os
import subprocess
from pathlib import Path

export_root = Path(os.environ["EXPORT_ROOT"]).resolve()
digest_file = Path(os.environ["DIGEST_FILE"]).resolve()
hash_cmd = os.environ["HASH_CMD"].split()

files = sorted(
    path
    for subdir in ("onnx", "ncnn", "torchscript")
    for path in (export_root / subdir).rglob("*")
    if path.is_file()
    and path.resolve() != digest_file
    and "__pycache__" not in path.parts
    and not path.name.endswith(".ncnn.param")
    and not path.name.endswith(".ncnn.bin")
    and not path.name.endswith(".ncnn.opt.param")
    and not path.name.endswith(".ncnn.opt.bin")
)

lines: list[str] = []
for path in files:
    rel = path.relative_to(export_root).as_posix()
    result = subprocess.run(
        [*hash_cmd, str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    digest = result.stdout.strip().split()[0]
    lines.append(f"{digest}  {rel}")

digest_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
print(f"[digest] wrote -> {digest_file}")
print(f"[digest] files = {len(files)}")
PY
