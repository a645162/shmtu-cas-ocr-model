#!/usr/bin/env bash
# 单卡速度 benchmark (QPS / p50/p99)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"
CONFIG="${CONFIG:-$SHMTU_SRC/cas_ocr_model/trainer/configs/8gpu_ddp.yaml}"
if [ -z "${SHMTU_PROFILE_NAME:-}" ]; then
    SHMTU_PROFILE_NAME="$(basename "${CONFIG%.*}")"
    export SHMTU_PROFILE_NAME
    export SHMTU_PROFILE_DIR="$SHMTU_RUNS_ROOT/$SHMTU_PROFILE_NAME"
fi
RUN_DIR="${RUN_DIR:-$(bash "$SCRIPT_DIR/run_path.sh" resolve)}"
CHECKPOINT="${CHECKPOINT:-$RUN_DIR/best.pt}"
BACKEND="${BACKEND:-pytorch}"
DEVICE="${DEVICE:-cuda}"
REPORT="${REPORT:-$RUN_DIR/single_gpu_bench.json}"
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
