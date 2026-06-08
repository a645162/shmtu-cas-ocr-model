#!/usr/bin/env bash
# 单卡速度 benchmark (QPS / p50/p99)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"
CHECKPOINT="${CHECKPOINT:-$SHMTU_RUN_DIR/best.pt}"
BACKEND="${BACKEND:-pytorch}"
DEVICE="${DEVICE:-cuda}"
REPORT="${REPORT:-$SHMTU_RUN_DIR/single_gpu_bench.json}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
BATCH_SIZES="${BATCH_SIZES:-1,8,32,128}"
if [ ! -f "$CHECKPOINT" ]; then
    echo "[bench-single] checkpoint 不存在: $CHECKPOINT"
    exit 1
fi
cd "$SHMTU_MODEL_ROOT"
$SHMTU_PYTHON -m cas_ocr_model.inference.single_gpu_benchmark \
    --backend "$BACKEND" \
    --checkpoint "$CHECKPOINT" \
    --device "$DEVICE" \
    --num-samples "$NUM_SAMPLES" \
    --batch-sizes "$BATCH_SIZES" \
    --output "$REPORT" \
    "$@"
