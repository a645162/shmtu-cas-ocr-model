#!/usr/bin/env bash

# 自动选择:
# 1) 没有 latest run -> 新建训练
# 2) 有 latest run 且 last.pt 未完成 -> 续训
# 3) 有 latest run 且已完成 -> 直接退出
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

if ! command -v accelerate >/dev/null 2>&1; then
    echo "[train-auto] accelerate 未安装, 请先 pip install accelerate>=0.27"
    exit 1
fi

resolve_latest_run() {
    if RUN_DIR="$(bash "$SCRIPT_DIR/../common/run_path.sh" resolve 2>/dev/null)"; then
        printf '%s\n' "$RUN_DIR"
        return 0
    fi
    return 1
}

read_checkpoint_status() {
    local ckpt_path="$1"
    "$SHMTU_PYTHON" - "$ckpt_path" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import torch

ckpt_path = Path(sys.argv[1])
raw = torch.load(ckpt_path, map_location="cpu")

epoch = raw.get("epoch")
config = raw.get("config") or {}
train_cfg = config.get("train") or {}
total_epochs = train_cfg.get("epochs")
stop_reason = raw.get("stop_reason")
early_stop_triggered = bool(raw.get("early_stop_triggered"))

if epoch is None or total_epochs is None:
    raise SystemExit("invalid-checkpoint")

epoch = int(epoch)
total_epochs = int(total_epochs)
done = 1 if (epoch + 1) >= total_epochs or early_stop_triggered or stop_reason == "early_stop" else 0
print(f"{done} {epoch + 1} {total_epochs}")
PY
}

if ! RUN_DIR="$(resolve_latest_run)"; then
    echo "[train-auto] latest run 不存在, 启动新训练"
    exec env CONFIG="$CONFIG" bash "$SCRIPT_DIR/train.sh" "$@"
fi

LAST_CKPT="$RUN_DIR/last.pt"
if [ ! -f "$LAST_CKPT" ]; then
    echo "[train-auto] latest run 缺少 last.pt, 启动新训练"
    exec env CONFIG="$CONFIG" bash "$SCRIPT_DIR/train.sh" "$@"
fi

STATUS="$(read_checkpoint_status "$LAST_CKPT")"
if [ "$STATUS" = "invalid-checkpoint" ]; then
    echo "[train-auto] checkpoint 缺少 epoch 或 total_epochs: $LAST_CKPT"
    exit 1
fi

read -r IS_DONE CURRENT_EPOCH TOTAL_EPOCHS <<<"$STATUS"

echo "[train-auto] profile: $SHMTU_PROFILE_NAME"
echo "[train-auto] output:  $RUN_DIR"
echo "[train-auto] last:    $LAST_CKPT"
echo "[train-auto] progress: ${CURRENT_EPOCH}/${TOTAL_EPOCHS}"

if [ "$IS_DONE" = "1" ]; then
    echo "[train-auto] latest run 已完成, 直接退出"
    exit 0
fi

echo "[train-auto] latest run 未完成, 继续续训"
exec env \
    CONFIG="$CONFIG" \
    SHMTU_RESUME_FROM="$LAST_CKPT" \
    bash "$SCRIPT_DIR/train.sh" "$@"
