#!/usr/bin/env bash
# 多卡 DDP 精度 benchmark (accelerate launch)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"
CHECKPOINT="${CHECKPOINT:-$SHMTU_RUN_DIR/best.pt}"
BACKEND="${BACKEND:-pytorch}"
REPORT="${REPORT:-$SHMTU_RUN_DIR/multi_gpu_report.json}"
LIMIT="${LIMIT:-}"
if [ ! -f "$CHECKPOINT" ]; then
    echo "[bench-multi] checkpoint 不存在: $CHECKPOINT"
    exit 1
fi
cd "$SHMTU_MODEL_ROOT"
accelerate launch \
    --num_processes "$SHMTU_NUM_GPUS" \
    --num_machines 1 \
    --dynamo_backend no \
    --mixed_precision fp16 \
    -m cas_ocr_model.inference.multi_gpu_benchmark \
    --backend "$BACKEND" \
    --checkpoint "$CHECKPOINT" \
    --data-root "$SHMTU_DATASET_ROOT" \
    --output "$REPORT" \
    ${LIMIT:+--limit "$LIMIT"} \
    "$@"
