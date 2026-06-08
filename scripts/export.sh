#!/usr/bin/env bash
# 导出 best.pt 为 ONNX
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
OUTPUT="${OUTPUT:-$RUN_DIR/model.onnx}"
BACKBONE="${BACKBONE:-resnet18}"
if [ ! -f "$CHECKPOINT" ]; then
    echo "[export] checkpoint 不存在: $CHECKPOINT"
    echo "[export] 请先跑 scripts/train.sh"
    exit 1
fi
cd "$SHMTU_MODEL_ROOT"
$SHMTU_PYTHON -m cas_ocr_model.trainer.export \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT" \
    --backbone "$BACKBONE" \
    --dynamic-batch
