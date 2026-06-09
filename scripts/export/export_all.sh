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

RUN_DIR="${RUN_DIR:-$(bash "$SCRIPT_DIR/../run_path.sh" resolve)}"
EXPORT_ROOT="${EXPORT_ROOT:-$RUN_DIR/export}"
ONNX_EXPORT_DIR="${ONNX_EXPORT_DIR:-$EXPORT_ROOT/onnx}"
NCNN_EXPORT_DIR="${NCNN_EXPORT_DIR:-$EXPORT_ROOT/ncnn}"

EXPORT_ROOT="$EXPORT_ROOT" EXPORT_DIR="$ONNX_EXPORT_DIR" RUN_DIR="$RUN_DIR" CONFIG="$CONFIG" \
    bash "$SCRIPT_DIR/export_onnx.sh"
EXPORT_ROOT="$EXPORT_ROOT" EXPORT_DIR="$NCNN_EXPORT_DIR" RUN_DIR="$RUN_DIR" CONFIG="$CONFIG" \
    bash "$SCRIPT_DIR/export_ncnn.sh"
EXPORT_ROOT="$EXPORT_ROOT" RUN_DIR="$RUN_DIR" CONFIG="$CONFIG" \
    bash "$SCRIPT_DIR/generate_release_digest.sh"
