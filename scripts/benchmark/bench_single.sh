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

NUM_SAMPLES="${NUM_SAMPLES:-500}"
WARMUP="${WARMUP:-20}"
BATCH_SIZES="${BATCH_SIZES:-1,8,32,128}"
case "$BACKEND" in
    pytorch)
        REPORT="${REPORT:-$RUN_DIR/single_gpu_bench.json}"
        ;;
    *)
        REPORT="${REPORT:-$RUN_DIR/single_gpu_bench.${BACKEND}.json}"
        ;;
esac

echo "[bench-single] backend = $BACKEND"
echo "[bench-single] device  = $DEVICE"
echo "[bench-single] report  = $REPORT"

cd "$SHMTU_MODEL_ROOT"
cmd=(
    "$SHMTU_PYTHON" -m cas_ocr_model.inference.cli
    --mode benchmark
    "${BACKEND_ARGS[@]}"
    --num-samples "$NUM_SAMPLES"
    --warmup "$WARMUP"
    --batch-sizes "$BATCH_SIZES"
    --output "$REPORT"
)
cmd+=("$@")
"${cmd[@]}"
