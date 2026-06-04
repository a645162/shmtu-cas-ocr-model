#!/usr/bin/env bash
# 仅下载 PyTorch 权重到 $SHMTU_WEIGHTS_DIR
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"
echo "[weights] 目标目录: $SHMTU_WEIGHTS_DIR"
mkdir -p "$SHMTU_WEIGHTS_DIR"
cd "$SHMTU_MODEL_ROOT"
$SHMTU_PYTHON -m cas_ocr_model.datasets.maker.cli --help >/dev/null  # sanity check
$SHMTU_PYTHON - <<PY
from cas_ocr_model.datasets.maker.config import ensure_pytorch_weights
from pathlib import Path
paths = ensure_pytorch_weights(Path("$SHMTU_WEIGHTS_DIR"))
for k, p in paths.items():
    print(f"[weights] {k}: {p} ({p.stat().st_size} bytes)")
PY
echo "[weights] done."
