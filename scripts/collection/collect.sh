#!/usr/bin/env bash
# 启动 maker 采集 CAS 验证码图片 + json
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../env.sh"
COUNT="${COUNT:-500000}"
PROCESSES="${PROCESSES:-4}"
PER_PROCESS="${PER_PROCESS:-8}"
echo "[collect] backend=$SHMTU_BACKEND output=$SHMTU_DATASET_ROOT count=$COUNT procs=$PROCESSES x$PER_PROCESS"
cd "$SHMTU_MODEL_ROOT"
$SHMTU_PYTHON -m cas_ocr_model.datasets.maker.cli \
    --backend "$SHMTU_BACKEND" \
    --ocr-url "$SHMTU_OCR_HTTP_URL" \
    --ocr-host "$SHMTU_OCR_HOST" \
    --ocr-port "$SHMTU_OCR_PORT" \
    --weights-dir "$SHMTU_WEIGHTS_DIR" \
    --output "$SHMTU_DATASET_ROOT" \
    --count "$COUNT" \
    --processes "$PROCESSES" \
    --per-process "$PER_PROCESS" \
    "$@"
