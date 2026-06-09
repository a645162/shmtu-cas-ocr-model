#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../env.sh"

shmtu_inference_init() {
    CONFIG="${CONFIG:-$SHMTU_SRC/cas_ocr_model/trainer/configs/8gpu_ddp.yaml}"
    if [ -z "${SHMTU_PROFILE_NAME:-}" ]; then
        SHMTU_PROFILE_NAME="$(basename "${CONFIG%.*}")"
        export SHMTU_PROFILE_NAME
        export SHMTU_PROFILE_DIR="$SHMTU_RUNS_ROOT/$SHMTU_PROFILE_NAME"
    fi

    RUN_DIR="${RUN_DIR:-$(bash "$SCRIPT_DIR/../run_path.sh" resolve)}"
    EXPORT_ROOT="${EXPORT_ROOT:-$RUN_DIR/export}"
    CHECKPOINT="${CHECKPOINT:-$RUN_DIR/best.pt}"
    ONNX_PATH="${ONNX_PATH:-$EXPORT_ROOT/onnx/best.fp32.onnx}"
    NCNN_PARAM="${NCNN_PARAM:-$EXPORT_ROOT/ncnn/best.fp32.param}"
    NCNN_BIN="${NCNN_BIN:-$EXPORT_ROOT/ncnn/best.fp32.bin}"
    BACKEND="${BACKEND:-pytorch}"

    case "$BACKEND" in
        pytorch)
            DEVICE="${DEVICE:-cuda}"
            ;;
        onnx|ncnn)
            DEVICE="cpu"
            ;;
        *)
            echo "[inference] 不支持的 BACKEND: $BACKEND"
            exit 1
            ;;
    esac
}

shmtu_inference_build_backend_args() {
    BACKEND_ARGS=(--backend "$BACKEND" --device "$DEVICE")

    case "$BACKEND" in
        pytorch)
            if [ ! -f "$CHECKPOINT" ]; then
                echo "[inference] checkpoint 不存在: $CHECKPOINT"
                exit 1
            fi
            BACKEND_ARGS+=(--checkpoint "$CHECKPOINT")
            ;;
        onnx)
            if [ ! -f "$ONNX_PATH" ]; then
                echo "[inference] onnx 不存在: $ONNX_PATH"
                exit 1
            fi
            BACKEND_ARGS+=(--onnx "$ONNX_PATH")
            ;;
        ncnn)
            if [ ! -f "$NCNN_PARAM" ]; then
                echo "[inference] ncnn param 不存在: $NCNN_PARAM"
                exit 1
            fi
            if [ ! -f "$NCNN_BIN" ]; then
                echo "[inference] ncnn bin 不存在: $NCNN_BIN"
                exit 1
            fi
            BACKEND_ARGS+=(--ncnn-param "$NCNN_PARAM" --ncnn-bin "$NCNN_BIN")
            ;;
    esac
}

shmtu_inference_is_help_request() {
    local arg=""
    for arg in "$@"; do
        if [ "$arg" = "-h" ] || [ "$arg" = "--help" ]; then
            return 0
        fi
    done
    return 1
}
