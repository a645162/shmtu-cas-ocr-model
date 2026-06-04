#!/usr/bin/env bash
# 物理分割 train/val/test, 写 manifest.json
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
VAL_RATIO="${VAL_RATIO:-0.1}"
TEST_RATIO="${TEST_RATIO:-0.1}"
SEED="${SEED:-42}"
cd "$SHMTU_MODEL_ROOT"
$SHMTU_PYTHON -m cas_ocr_model.datasets.split \
    --dataset-root "$SHMTU_DATASET_ROOT" \
    --train-ratio "$TRAIN_RATIO" \
    --val-ratio "$VAL_RATIO" \
    --test-ratio "$TEST_RATIO" \
    --seed "$SEED" \
    --source-backend "$SHMTU_BACKEND" \
    --source-url "$SHMTU_OCR_HTTP_URL"
