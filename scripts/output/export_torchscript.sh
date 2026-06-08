#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../env.sh"

CHECKPOINT="${CHECKPOINT:-$SHMTU_RUN_DIR/best.pt}"
EXPORT_DIR="${EXPORT_DIR:-$SHMTU_RUN_DIR/export}"
MODEL_NAME="${MODEL_NAME:-$(basename "${CHECKPOINT%.*}")}"
OUTPUT="${OUTPUT:-$EXPORT_DIR/$MODEL_NAME.ts.pt}"
IMAGE_SIZE_H="${IMAGE_SIZE_H:-64}"
IMAGE_SIZE_W="${IMAGE_SIZE_W:-192}"

mkdir -p "$EXPORT_DIR"

if [ ! -f "$CHECKPOINT" ]; then
    echo "[export-ts] checkpoint 不存在: $CHECKPOINT"
    exit 1
fi

cd "$SHMTU_MODEL_ROOT"
echo "[export-ts] checkpoint = $CHECKPOINT"
echo "[export-ts] output     = $OUTPUT"
"$SHMTU_PYTHON" -m cas_ocr_model.trainer.export_torchscript \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT" \
    --image-size-h "$IMAGE_SIZE_H" \
    --image-size-w "$IMAGE_SIZE_W"
