#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../env.sh"

CONFIG="${CONFIG:-$SHMTU_SRC/cas_ocr_model/trainer/configs/8gpu_ddp.yaml}"
if [ -z "${SHMTU_PROFILE_NAME:-}" ]; then
    SHMTU_PROFILE_NAME="$(basename "${CONFIG%.*}")"
    export SHMTU_PROFILE_NAME
    export SHMTU_PROFILE_DIR="$SHMTU_RUNS_ROOT/$SHMTU_PROFILE_NAME"
fi
RUN_DIR="${RUN_DIR:-$(bash "$SCRIPT_DIR/../run_path.sh" resolve)}"
CHECKPOINT="${CHECKPOINT:-$RUN_DIR/best.pt}"
EXPORT_DIR="${EXPORT_DIR:-$RUN_DIR/export}"
MODEL_NAME="${MODEL_NAME:-$(basename "${CHECKPOINT%.*}")}"
OUTPUT="${OUTPUT:-$EXPORT_DIR/$MODEL_NAME.onnx}"
IMAGE_SIZE_H="${IMAGE_SIZE_H:-64}"
IMAGE_SIZE_W="${IMAGE_SIZE_W:-192}"
OPSET="${OPSET:-17}"
LEGACY_EXPORTER="${LEGACY_EXPORTER:-1}"
DYNAMIC_BATCH="${DYNAMIC_BATCH:-1}"

mkdir -p "$EXPORT_DIR"

if [ ! -f "$CHECKPOINT" ]; then
    echo "[export-onnx] checkpoint 不存在: $CHECKPOINT"
    exit 1
fi

ARGS=(
    -m cas_ocr_model.trainer.export
    --checkpoint "$CHECKPOINT"
    --output "$OUTPUT"
    --image-size-h "$IMAGE_SIZE_H"
    --image-size-w "$IMAGE_SIZE_W"
    --opset "$OPSET"
)

if [ "$DYNAMIC_BATCH" = "1" ]; then
    ARGS+=(--dynamic-batch)
fi

if [ "$LEGACY_EXPORTER" = "1" ]; then
    ARGS+=(--legacy-exporter)
fi

cd "$SHMTU_MODEL_ROOT"
echo "[export-onnx] checkpoint = $CHECKPOINT"
echo "[export-onnx] output     = $OUTPUT"
"$SHMTU_PYTHON" "${ARGS[@]}"
