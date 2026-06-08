#!/usr/bin/env bash

# 自动续训当前 profile/latest 对应的最后一个 run。
# 若 last.pt 已经达到配置中的总 epoch，则直接退出。
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
    echo "[train-resume] accelerate 未安装, 请先 pip install accelerate>=0.27"
    exit 1
fi

RUN_DIR="$(bash "$SCRIPT_DIR/run_path.sh" resolve)"
LAST_CKPT="$RUN_DIR/last.pt"
if [ ! -f "$LAST_CKPT" ]; then
    echo "[train-resume] last checkpoint 不存在: $LAST_CKPT"
    exit 1
fi

STATUS="$("$SHMTU_PYTHON" - "$LAST_CKPT" <<'PY'
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

if epoch is None or total_epochs is None:
    raise SystemExit("invalid-checkpoint")

epoch = int(epoch)
total_epochs = int(total_epochs)
done = 1 if (epoch + 1) >= total_epochs else 0
print(f"{done} {epoch + 1} {total_epochs}")
PY
)"

if [ "$STATUS" = "invalid-checkpoint" ]; then
    echo "[train-resume] checkpoint 缺少 epoch 或 total_epochs: $LAST_CKPT"
    exit 1
fi

read -r IS_DONE CURRENT_EPOCH TOTAL_EPOCHS <<<"$STATUS"

echo "[train-resume] profile: $SHMTU_PROFILE_NAME"
echo "[train-resume] output:  $RUN_DIR"
echo "[train-resume] last:    $LAST_CKPT"
echo "[train-resume] progress: ${CURRENT_EPOCH}/${TOTAL_EPOCHS}"

if [ "$IS_DONE" = "1" ]; then
    echo "[train-resume] run 已完成, 直接退出"
    exit 0
fi

exec env \
    CONFIG="$CONFIG" \
    SHMTU_RESUME_FROM="$LAST_CKPT" \
    bash "$SCRIPT_DIR/train.sh" "$@"
