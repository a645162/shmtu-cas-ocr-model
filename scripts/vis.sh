#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

CONFIG="${CONFIG:-$SHMTU_SRC/cas_ocr_model/trainer/configs/8gpu_ddp.yaml}"
CHECKPOINT="${CHECKPOINT:-$SHMTU_RUN_DIR/best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$SHMTU_MODEL_ROOT/output}"
DEVICE="${DEVICE:-cuda}"
N="${N:-20}"
SEED="${SEED:-42}"
SUBDIR="${SUBDIR:-}"

if [ ! -f "$CHECKPOINT" ]; then
    echo "[vis] checkpoint 不存在: $CHECKPOINT"
    exit 1
fi

echo "[vis] config:     $CONFIG"
echo "[vis] checkpoint: $CHECKPOINT"
echo "[vis] output:     $OUTPUT_DIR"
echo "[vis] device:     $DEVICE"
echo "[vis] n:          $N"
echo "[vis] seed:       $SEED"

cd "$SHMTU_MODEL_ROOT"
cmd=(
    "$SHMTU_PYTHON" scripts/visualize_test_predictions.py
    --config "$CONFIG"
    --checkpoint "$CHECKPOINT"
    --data-root "$SHMTU_DATASET_ROOT"
    --output-dir "$OUTPUT_DIR"
    --device "$DEVICE"
    --n "$N"
    --seed "$SEED"
)
if [ -n "$SUBDIR" ]; then
    cmd+=(--subdir "$SUBDIR")
fi
cmd+=("$@")
"${cmd[@]}"
