#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../inference/common.sh"

if shmtu_inference_is_help_request "$@"; then
    cd "$SHMTU_MODEL_ROOT"
    exec "$SHMTU_PYTHON" scripts/visualization/visualize_test_predictions.py --help
fi

shmtu_inference_init
shmtu_inference_build_backend_args

OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/outputs/$BACKEND}"
N="${N:-20}"
SEED="${SEED:-42}"
SUBDIR="${SUBDIR:-}"

echo "[vis] config:     $CONFIG"
echo "[vis] backend:    $BACKEND"
echo "[vis] run dir:    $RUN_DIR"
echo "[vis] output:     $OUTPUT_DIR"
echo "[vis] device:     $DEVICE"
echo "[vis] n:          $N"
echo "[vis] seed:       $SEED"
case "$BACKEND" in
    pytorch)
        echo "[vis] checkpoint: $CHECKPOINT"
        ;;
    onnx)
        echo "[vis] onnx:       $ONNX_PATH"
        ;;
    ncnn)
        echo "[vis] ncnn param: $NCNN_PARAM"
        echo "[vis] ncnn bin:   $NCNN_BIN"
        ;;
esac

cd "$SHMTU_MODEL_ROOT"
cmd=(
    "$SHMTU_PYTHON" scripts/visualization/visualize_test_predictions.py
    --config "$CONFIG"
    "${BACKEND_ARGS[@]}"
    --data-root "$SHMTU_DATASET_ROOT"
    --output-dir "$OUTPUT_DIR"
    --n "$N"
    --seed "$SEED"
)
if [ -n "$SUBDIR" ]; then
    cmd+=(--subdir "$SUBDIR")
fi
cmd+=("$@")
"${cmd[@]}"
