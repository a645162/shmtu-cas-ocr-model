#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../env.sh"

resolve_tool() {
    local explicit_var="$1"
    shift
    local explicit_value="${!explicit_var:-}"
    if [ -n "$explicit_value" ]; then
        echo "$explicit_value"
        return 0
    fi

    local candidate=""
    for candidate in "$@"; do
        if [ -n "$candidate" ] && [ -x "$candidate" ]; then
            echo "$candidate"
            return 0
        fi
    done

    return 1
}

CONFIG="${CONFIG:-$SHMTU_SRC/cas_ocr_model/trainer/configs/8gpu_ddp.yaml}"
if [ -z "${SHMTU_PROFILE_NAME:-}" ]; then
    SHMTU_PROFILE_NAME="$(basename "${CONFIG%.*}")"
    export SHMTU_PROFILE_NAME
    export SHMTU_PROFILE_DIR="$SHMTU_RUNS_ROOT/$SHMTU_PROFILE_NAME"
fi
RUN_DIR="${RUN_DIR:-$(bash "$SCRIPT_DIR/../run_path.sh" resolve)}"
CHECKPOINT="${CHECKPOINT:-$RUN_DIR/best.pt}"
EXPORT_DIR="${EXPORT_DIR:-$RUN_DIR/export}"
MODEL_NAME="${MODEL_NAME:-$(basename "${CHECKPOINT%.*}")}"
TORCHSCRIPT_PATH="${TORCHSCRIPT_PATH:-$EXPORT_DIR/$MODEL_NAME.ts.pt}"
NCNN_PARAM="${NCNN_PARAM:-$EXPORT_DIR/$MODEL_NAME.ncnn.param}"
NCNN_BIN="${NCNN_BIN:-$EXPORT_DIR/$MODEL_NAME.ncnn.bin}"
PNNX_PARAM="${PNNX_PARAM:-$EXPORT_DIR/$MODEL_NAME.pnnx.param}"
PNNX_BIN="${PNNX_BIN:-$EXPORT_DIR/$MODEL_NAME.pnnx.bin}"
PNNX_PY="${PNNX_PY:-$EXPORT_DIR/${MODEL_NAME}_pnnx.py}"
PNNX_ONNX="${PNNX_ONNX:-$EXPORT_DIR/$MODEL_NAME.pnnx.onnx}"
IMAGE_SIZE_H="${IMAGE_SIZE_H:-64}"
IMAGE_SIZE_W="${IMAGE_SIZE_W:-192}"
REBUILD_TORCHSCRIPT="${REBUILD_TORCHSCRIPT:-1}"
PNNX_FP16="${PNNX_FP16:-0}"
PNNX_OPTLEVEL="${PNNX_OPTLEVEL:-2}"
RUN_NCNNOPTIMIZE="${RUN_NCNNOPTIMIZE:-1}"
NCNN_OPT_PARAM="${NCNN_OPT_PARAM:-$EXPORT_DIR/$MODEL_NAME.ncnn.opt.param}"
NCNN_OPT_BIN="${NCNN_OPT_BIN:-$EXPORT_DIR/$MODEL_NAME.ncnn.opt.bin}"
NCNNOPTIMIZE_FLAG="${NCNNOPTIMIZE_FLAG:-1}"

mkdir -p "$EXPORT_DIR"

if [ ! -f "$CHECKPOINT" ]; then
    echo "[export-ncnn] checkpoint 不存在: $CHECKPOINT"
    exit 1
fi

if [ "$REBUILD_TORCHSCRIPT" = "1" ] || [ ! -f "$TORCHSCRIPT_PATH" ]; then
    CHECKPOINT="$CHECKPOINT" \
    EXPORT_DIR="$EXPORT_DIR" \
    MODEL_NAME="$MODEL_NAME" \
    OUTPUT="$TORCHSCRIPT_PATH" \
    IMAGE_SIZE_H="$IMAGE_SIZE_H" \
    IMAGE_SIZE_W="$IMAGE_SIZE_W" \
        bash "$SCRIPT_DIR/export_torchscript.sh"
fi

if ! PNNX_BIN_PATH="$(
    resolve_tool \
        PNNX \
        "$(command -v pnnx 2>/dev/null || true)" \
        "$SHMTU_MODEL_ROOT/3rdparty/ncnn/bin/pnnx" \
        "${PNNX_HOME:-}/bin/pnnx" \
        "${PNNX_HOME:-}/build/tools/pnnx/pnnx" \
        "${NCNN_HOME:-}/bin/pnnx" \
        "${NCNN_HOME:-}/build/tools/pnnx/pnnx"
)"; then
    PNNX_BIN_PATH=""
fi

if [ -z "$PNNX_BIN_PATH" ]; then
    echo "[export-ncnn] 未找到 pnnx."
    echo "[export-ncnn] 可通过环境变量 PNNX=/abs/path/to/pnnx 指定."
    echo "[export-ncnn] 或先执行: bash scripts/output/install_ncnn_tools.sh"
    exit 1
fi

echo "[export-ncnn] torchscript = $TORCHSCRIPT_PATH"
echo "[export-ncnn] pnnx        = $PNNX_BIN_PATH"
echo "[export-ncnn] ncnn param  = $NCNN_PARAM"
echo "[export-ncnn] ncnn bin    = $NCNN_BIN"

"$PNNX_BIN_PATH" "$TORCHSCRIPT_PATH" \
    "inputshape=[1,1,${IMAGE_SIZE_H},${IMAGE_SIZE_W}]" \
    "fp16=${PNNX_FP16}" \
    "optlevel=${PNNX_OPTLEVEL}" \
    "pnnxparam=$PNNX_PARAM" \
    "pnnxbin=$PNNX_BIN" \
    "pnnxpy=$PNNX_PY" \
    "pnnxonnx=$PNNX_ONNX" \
    "ncnnparam=$NCNN_PARAM" \
    "ncnnbin=$NCNN_BIN"

if [ "$RUN_NCNNOPTIMIZE" = "1" ]; then
    if ! NCNNOPTIMIZE_BIN_PATH="$(
        resolve_tool \
            NCNNOPTIMIZE \
            "$(command -v ncnnoptimize 2>/dev/null || true)" \
            "$SHMTU_MODEL_ROOT/3rdparty/ncnn/bin/ncnnoptimize" \
            "${NCNN_HOME:-}/bin/ncnnoptimize" \
            "${NCNN_HOME:-}/build/tools/ncnnoptimize/ncnnoptimize"
    )"; then
        NCNNOPTIMIZE_BIN_PATH=""
    fi

    if [ -n "$NCNNOPTIMIZE_BIN_PATH" ]; then
        echo "[export-ncnn] ncnnoptimize = $NCNNOPTIMIZE_BIN_PATH"
        "$NCNNOPTIMIZE_BIN_PATH" \
            "$NCNN_PARAM" \
            "$NCNN_BIN" \
            "$NCNN_OPT_PARAM" \
            "$NCNN_OPT_BIN" \
            "$NCNNOPTIMIZE_FLAG"
    else
        echo "[export-ncnn] 未找到 ncnnoptimize，跳过优化."
    fi
fi
