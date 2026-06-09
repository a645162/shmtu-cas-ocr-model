#!/usr/bin/env bash
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

RUN_DIR="${RUN_DIR:-$(bash "$SCRIPT_DIR/../common/run_path.sh" resolve)}"
RELEASE_ROOT="${RELEASE_ROOT:-$RUN_DIR/release}"
EXPORT_ROOT="${EXPORT_ROOT:-$RELEASE_ROOT}"
ONNX_EXPORT_DIR="${ONNX_EXPORT_DIR:-$EXPORT_ROOT/onnx}"
NCNN_EXPORT_DIR="${NCNN_EXPORT_DIR:-$EXPORT_ROOT/ncnn}"

EXPORT_ROOT="$EXPORT_ROOT" EXPORT_DIR="$ONNX_EXPORT_DIR" RUN_DIR="$RUN_DIR" RELEASE_ROOT="$RELEASE_ROOT" CONFIG="$CONFIG" \
    bash "$SCRIPT_DIR/export_onnx.sh"
EXPORT_ROOT="$EXPORT_ROOT" EXPORT_DIR="$NCNN_EXPORT_DIR" RUN_DIR="$RUN_DIR" RELEASE_ROOT="$RELEASE_ROOT" CONFIG="$CONFIG" \
    bash "$SCRIPT_DIR/export_ncnn.sh"

rm -f "$EXPORT_ROOT/SHA256SUMS.txt"
