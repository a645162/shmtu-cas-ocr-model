#!/usr/bin/env bash

# 尝试模仿 scripts/training/train.sh 的风格，提供一致的 env/run 管理
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../env.sh"

OUT_DIR_OVERRIDE="${1:-}"

if [ -z "${SHMTU_PROFILE_NAME:-}" ]; then
	SHMTU_PROFILE_NAME="one_epoch_debug"
	export SHMTU_PROFILE_NAME
	export SHMTU_PROFILE_DIR="$SHMTU_RUNS_ROOT/$SHMTU_PROFILE_NAME"
fi

if [ -n "$OUT_DIR_OVERRIDE" ]; then
	RUN_DIR="$OUT_DIR_OVERRIDE"
else
	RUN_DIR="$(bash "$SCRIPT_DIR/../common/run_path.sh" create)"
fi

MAIN_PROCESS_PORT="$(bash "$SCRIPT_DIR/../common/ddp_port.sh")"
export MASTER_PORT="$MAIN_PROCESS_PORT"

echo "[run_one_epoch] python: ${SHMTU_PYTHON}"
echo "[run_one_epoch] profile: $SHMTU_PROFILE_NAME"
echo "[run_one_epoch] output:  $RUN_DIR"
echo "[run_one_epoch] port:    $MAIN_PROCESS_PORT"

mkdir -p "$RUN_DIR"
cd "$SHMTU_MODEL_ROOT"

# Reuse the existing train.sh pipeline but force --epochs 1 at the end so the
# CLI parser will pick the last value and we run a single epoch.

"$SCRIPT_DIR/train.sh" "$@" --output-dir "$RUN_DIR" --epochs 1

# When train.sh finishes, simulate the GitHub Action extraction step that
# would scan release/pytorch and the run dir for .pt files and produce a
# model-assets-simulated.json for review.
echo "[run_one_epoch] running simulated release extraction"
"$SHMTU_PYTHON" "$SHMTU_MODEL_ROOT/scripts/export/simulate_release.py" "$RUN_DIR" || \
	echo "[run_one_epoch] simulate_release.py exited with non-zero status"
