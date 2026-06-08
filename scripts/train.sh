#!/usr/bin/env bash

# 启动 8 卡 DDP 训练 (accelerate launch + fp16 + 8gpu_ddp.yaml)
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

if ! command -v accelerate >/dev/null 2>&1; then
    echo "[train] accelerate 未安装, 请先 pip install accelerate>=0.27"
    exit 1
fi

RUN_DIR="$(bash "$SCRIPT_DIR/run_path.sh" create)"

echo "[train] accelerate launch --num_processes $SHMTU_NUM_GPUS --num_machines 1 --dynamo_backend no"
echo "[train] config:  $CONFIG"
echo "[train] dataset: $SHMTU_DATASET_ROOT"
echo "[train] profile: $SHMTU_PROFILE_NAME"
echo "[train] output:  $RUN_DIR"

mkdir -p "$RUN_DIR"
cd "$SHMTU_MODEL_ROOT"

accelerate launch \
    --num_processes "$SHMTU_NUM_GPUS" \
    --num_machines 1 \
    --dynamo_backend no \
    --mixed_precision fp16 \
    -m cas_ocr_model.trainer.train \
    --config "$CONFIG" \
    --data-root "$SHMTU_DATASET_ROOT" \
    --output-dir "$RUN_DIR" \
    "$@"
