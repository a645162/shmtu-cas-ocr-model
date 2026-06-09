#!/usr/bin/env bash

# 启动 8 卡 DDP 训练 (accelerate launch + fp16 + 8gpu_ddp.yaml)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../env.sh"
CONFIG="${CONFIG:-$SHMTU_SRC/cas_ocr_model/trainer/configs/8gpu_ddp.yaml}"
RESUME_FROM="${RESUME_FROM:-${SHMTU_RESUME_FROM:-}}"
RESUME="${RESUME:-${SHMTU_RESUME:-0}}"
AUTO_VIS="${AUTO_VIS:-${SHMTU_AUTO_VIS:-1}}"
VIS_CHECKPOINT="${VIS_CHECKPOINT:-last.pt}"
VIS_N="${VIS_N:-20}"
VIS_DEVICE="${VIS_DEVICE:-cuda}"

if [ -z "${SHMTU_PROFILE_NAME:-}" ]; then
    SHMTU_PROFILE_NAME="$(basename "${CONFIG%.*}")"
    export SHMTU_PROFILE_NAME
    export SHMTU_PROFILE_DIR="$SHMTU_RUNS_ROOT/$SHMTU_PROFILE_NAME"
fi

if ! command -v accelerate >/dev/null 2>&1; then
    echo "[train] accelerate 未安装, 请先 pip install accelerate>=0.27"
    exit 1
fi

if [ -z "$RESUME_FROM" ] && [ "$RESUME" = "1" ]; then
    RESOLVED_RUN_DIR="$(bash "$SCRIPT_DIR/../common/run_path.sh" resolve)"
    if [ ! -d "$RESOLVED_RUN_DIR" ]; then
        echo "[train] resume run 不存在: $RESOLVED_RUN_DIR"
        exit 1
    fi
    RESUME_FROM="$RESOLVED_RUN_DIR/last.pt"
fi

if [ -n "$RESUME_FROM" ]; then
    if [ ! -f "$RESUME_FROM" ]; then
        echo "[train] resume checkpoint 不存在: $RESUME_FROM"
        exit 1
    fi
    RUN_DIR="$(dirname "$RESUME_FROM")"
else
    RUN_DIR="$(bash "$SCRIPT_DIR/../common/run_path.sh" create)"
fi

echo "[train] accelerate launch --num_processes $SHMTU_NUM_GPUS --num_machines 1 --dynamo_backend $SHMTU_DYNAMO_BACKEND"
echo "[train] config:  $CONFIG"
echo "[train] dataset: $SHMTU_DATASET_ROOT"
echo "[train] profile: $SHMTU_PROFILE_NAME"
echo "[train] output:  $RUN_DIR"
if [ -n "$RESUME_FROM" ]; then
    echo "[train] resume:  $RESUME_FROM"
fi

mkdir -p "$RUN_DIR"
cd "$SHMTU_MODEL_ROOT"

TRAIN_ARGS=(
    --config "$CONFIG"
    --data-root "$SHMTU_DATASET_ROOT"
    --output-dir "$RUN_DIR"
)
if [ -n "$RESUME_FROM" ]; then
    TRAIN_ARGS+=(--resume-from "$RESUME_FROM")
fi

accelerate launch \
    --num_processes "$SHMTU_NUM_GPUS" \
    --num_machines 1 \
    --dynamo_backend "$SHMTU_DYNAMO_BACKEND" \
    --mixed_precision fp16 \
    -m cas_ocr_model.trainer.train \
    "${TRAIN_ARGS[@]}" \
    "$@"

if [ "$AUTO_VIS" = "1" ]; then
    echo "[train] auto vis enabled"
    CHECKPOINT="$RUN_DIR/$VIS_CHECKPOINT" \
    OUTPUT_DIR="$RUN_DIR/outputs" \
    N="$VIS_N" \
    DEVICE="$VIS_DEVICE" \
    SUBDIR="auto_test_vis" \
        bash "$SCRIPT_DIR/../visualization/vis.sh"
fi
