#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../inference/common.sh"

if shmtu_inference_is_help_request "$@"; then
    cd "$SHMTU_MODEL_ROOT"
    exec "$SHMTU_PYTHON" -m cas_ocr_model.inference.cli --help
fi

shmtu_inference_init
shmtu_inference_build_backend_args

GT_DIR="${GT_DIR:-$SHMTU_DATASET_ROOT}"
LIMIT="${LIMIT:-}"
case "$BACKEND" in
    pytorch)
        REPORT="${REPORT:-$RUN_DIR/eval_report.json}"
        ;;
    *)
        REPORT="${REPORT:-$RUN_DIR/eval_report.${BACKEND}.json}"
        ;;
esac

echo "[evaluate] backend   = $BACKEND"
echo "[evaluate] run dir   = $RUN_DIR"
echo "[evaluate] report    = $REPORT"

cd "$SHMTU_MODEL_ROOT"
cmd=(
    "$SHMTU_PYTHON" -m cas_ocr_model.inference.cli
    --mode evaluate
    "${BACKEND_ARGS[@]}"
    --gt-dir "$GT_DIR"
    --output "$REPORT"
)
if [ -n "$LIMIT" ]; then
    cmd+=(--limit "$LIMIT")
fi
cmd+=("$@")
"${cmd[@]}"
