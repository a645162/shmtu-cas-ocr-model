#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

DATASET_ROOT="${DATASET_ROOT:-$SHMTU_MODEL_ROOT/dataset}"
ARCHIVE_DIR="${ARCHIVE_DIR:-$SHMTU_MODEL_ROOT/archives}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
VAL_RATIO="${VAL_RATIO:-0.1}"
TEST_RATIO="${TEST_RATIO:-0.1}"
SEED="${SEED:-42}"
SOURCE_BACKEND="${SOURCE_BACKEND:-}"
SOURCE_URL="${SOURCE_URL:-}"
ZIP_ROOT_NAME="${ZIP_ROOT_NAME:-datasets}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ZIP="${OUTPUT_ZIP:-$ARCHIVE_DIR/${ZIP_ROOT_NAME}_${TIMESTAMP}.zip}"

echo "[split-zip] dataset_root=$DATASET_ROOT"
echo "[split-zip] output_zip=$OUTPUT_ZIP"
echo "[split-zip] ratios=$TRAIN_RATIO/$VAL_RATIO/$TEST_RATIO seed=$SEED"

cmd_split=(
    "$SHMTU_PYTHON" "$SCRIPT_DIR/generate_dataset_split.py"
    --dataset-root "$DATASET_ROOT"
    --train-ratio "$TRAIN_RATIO"
    --val-ratio "$VAL_RATIO"
    --test-ratio "$TEST_RATIO"
    --seed "$SEED"
)
if [ -n "$SOURCE_BACKEND" ]; then
    cmd_split+=(--source-backend "$SOURCE_BACKEND")
fi
if [ -n "$SOURCE_URL" ]; then
    cmd_split+=(--source-url "$SOURCE_URL")
fi
"${cmd_split[@]}"

cmd_zip=(
    "$SHMTU_PYTHON" "$SCRIPT_DIR/package_dataset_zip.py"
    --dataset-root "$DATASET_ROOT"
    --output-zip "$OUTPUT_ZIP"
    --zip-root-name "$ZIP_ROOT_NAME"
)
"${cmd_zip[@]}"

echo "[split-zip] done -> $OUTPUT_ZIP"
