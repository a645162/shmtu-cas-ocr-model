#!/usr/bin/env bash
# 单卡 evaluate (算指标)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"
CHECKPOINT="${CHECKPOINT:-$SHMTU_RUN_DIR/best.pt}"
BACKEND="${BACKEND:-pytorch}"
REPORT="${REPORT:-$SHMTU_RUN_DIR/eval_report.json}"
LIMIT="${LIMIT:-}"
if [ ! -f "$CHECKPOINT" ]; then
    echo "[evaluate] checkpoint 不存在: $CHECKPOINT"
    exit 1
fi
cd "$SHMTU_MODEL_ROOT"
$SHMTU_PYTHON -m cas_ocr_model.inference.cli \
    --mode evaluate \
    --backend "$BACKEND" \
    --checkpoint "$CHECKPOINT" \
    --gt-dir "$SHMTU_DATASET_ROOT" \
    --output "$REPORT" \
    ${LIMIT:+--limit "$LIMIT"} \
    "$@"
