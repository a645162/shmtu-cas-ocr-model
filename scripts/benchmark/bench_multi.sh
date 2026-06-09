#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../inference/common.sh"

if shmtu_inference_is_help_request "$@"; then
    cd "$SHMTU_MODEL_ROOT"
    exec "$SHMTU_PYTHON" -m cas_ocr_model.inference.multi_gpu_benchmark --help
fi

shmtu_inference_init
BACKEND=pytorch
CHECKPOINT="${CHECKPOINT:-$RUN_DIR/best.pt}"
REPORT="${REPORT:-$RUN_DIR/multi_gpu_report.json}"
LIMIT="${LIMIT:-}"

if [ ! -f "$CHECKPOINT" ]; then
    echo "[bench-multi] checkpoint 不存在: $CHECKPOINT"
    exit 1
fi

cd "$SHMTU_MODEL_ROOT"
accelerate launch \
    --num_processes "$SHMTU_NUM_GPUS" \
    --num_machines 1 \
    --dynamo_backend "$SHMTU_DYNAMO_BACKEND" \
    --mixed_precision fp16 \
    -m cas_ocr_model.inference.multi_gpu_benchmark \
    --checkpoint "$CHECKPOINT" \
    --data-root "$SHMTU_DATASET_ROOT" \
    --output "$REPORT" \
    ${LIMIT:+--limit "$LIMIT"} \
    "$@"
