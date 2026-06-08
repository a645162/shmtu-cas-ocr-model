#!/usr/bin/env bash
# 导出 best.pt 为 ONNX
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"
CHECKPOINT="${CHECKPOINT:-$SHMTU_RUN_DIR/best.pt}"
OUTPUT="${OUTPUT:-$SHMTU_RUN_DIR/model.onnx}"
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
