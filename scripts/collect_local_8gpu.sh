#!/usr/bin/env bash
# 8 卡本地模型采集: 每卡 1 进程, 本地推理后提交 CAS 验证
set -euo pipefail

export NVI_NOTIFY_IGNORE_TASK=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

COUNT="${COUNT:-10_0000}"
CHECKPOINT="${CHECKPOINT:-$SHMTU_RUN_DIR/best.pt}"
OUTPUT="${OUTPUT:-$SHMTU_DATASET_ROOT}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$SHMTU_WEIGHTS_DIR}"
THROTTLE="${THROTTLE:-0.0}"
REPORT_INTERVAL="${REPORT_INTERVAL:-5.0}"
GPU_IDS="${GPU_IDS:-}"

if [ ! -f "$CHECKPOINT" ]; then
    echo "[collect-local-8gpu] checkpoint 不存在: $CHECKPOINT"
    exit 1
fi

if [ -z "$GPU_IDS" ]; then
    gpu_list=()
    for ((i = 0; i < SHMTU_NUM_GPUS; i++)); do
        gpu_list+=("$i")
    done
    GPU_IDS="$(IFS=,; echo "${gpu_list[*]}")"
fi

echo "[collect-local-8gpu] checkpoint=$CHECKPOINT"
echo "[collect-local-8gpu] output=$OUTPUT"
echo "[collect-local-8gpu] count=$COUNT"
echo "[collect-local-8gpu] gpu_ids=$GPU_IDS"
echo "[collect-local-8gpu] throttle=$THROTTLE"

cd "$SHMTU_MODEL_ROOT"
cmd=(
    "$SHMTU_PYTHON" -m cas_ocr_model.datasets.maker.local_model_collect
    --checkpoint "$CHECKPOINT"
    --output "$OUTPUT"
    --count "$COUNT"
    --gpu-ids "$GPU_IDS"
    --weights-dir "$WEIGHTS_DIR"
    --throttle "$THROTTLE"
    --report-interval "$REPORT_INTERVAL"
    --resume
)
cmd+=("$@")
"${cmd[@]}"
