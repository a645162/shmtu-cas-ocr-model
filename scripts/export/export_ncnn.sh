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

prepend_library_path() {
    local lib_dir="$1"
    if [ -z "$lib_dir" ] || [ ! -d "$lib_dir" ]; then
        return 0
    fi

    case ":${LD_LIBRARY_PATH:-}:" in
        *":$lib_dir:"*) ;;
        *)
            export LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            ;;
    esac
}

CONFIG="${CONFIG:-$SHMTU_SRC/cas_ocr_model/trainer/configs/8gpu_ddp.yaml}"
if [ -z "${SHMTU_PROFILE_NAME:-}" ]; then
    SHMTU_PROFILE_NAME="$(basename "${CONFIG%.*}")"
    export SHMTU_PROFILE_NAME
    export SHMTU_PROFILE_DIR="$SHMTU_RUNS_ROOT/$SHMTU_PROFILE_NAME"
fi
RUN_DIR="${RUN_DIR:-$(bash "$SCRIPT_DIR/../common/run_path.sh" resolve)}"
CHECKPOINT="${CHECKPOINT:-$RUN_DIR/best.pt}"
EXPORT_ROOT="${EXPORT_ROOT:-$RUN_DIR/export}"
EXPORT_DIR="${EXPORT_DIR:-$EXPORT_ROOT/ncnn}"
EXPORT_NCNN_MODE="${EXPORT_NCNN_MODE:-python}"
IMAGE_SIZE_H="${IMAGE_SIZE_H:-64}"
IMAGE_SIZE_W="${IMAGE_SIZE_W:-192}"
REBUILD_TORCHSCRIPT="${REBUILD_TORCHSCRIPT:-1}"
PNNX_OPTLEVEL="${PNNX_OPTLEVEL:-2}"
RUN_NCNNOPTIMIZE="${RUN_NCNNOPTIMIZE:-0}"
NCNNOPTIMIZE_FLAG="${NCNNOPTIMIZE_FLAG:-1}"

mkdir -p "$EXPORT_DIR"

if [ ! -f "$CHECKPOINT" ]; then
    echo "[export-ncnn] checkpoint 不存在: $CHECKPOINT"
    exit 1
fi

MODEL_NAME="${MODEL_NAME:-$("$SHMTU_PYTHON" -m cas_ocr_model.model.cli checkpoint-metadata --checkpoint "$CHECKPOINT" --field asset_stem 2>/dev/null || basename "${CHECKPOINT%.*}")}"

normalize_precision() {
    case "$1" in
        fp16|float16|half)
            echo "fp16"
            ;;
        fp32|float32)
            echo "fp32"
            ;;
        *)
            echo "[export-ncnn] 不支持的精度: $1" >&2
            return 1
            ;;
    esac
}

collect_precisions() {
    local raw=""
    if [ -n "${EXPORT_PRECISIONS:-}" ]; then
        raw="$EXPORT_PRECISIONS"
    elif [ -n "${EXPORT_PRECISION_TAG:-}" ]; then
        raw="$EXPORT_PRECISION_TAG"
    else
        raw="fp16 fp32"
    fi

    raw="${raw//,/ }"
    read -r -a PRECISIONS <<< "$raw"
    if [ "${#PRECISIONS[@]}" -eq 0 ]; then
        echo "[export-ncnn] 未解析到任何导出精度." >&2
        exit 1
    fi
}

ensure_single_precision_overrides() {
    local explicit_vars=(
        OUTPUT
        TORCHSCRIPT_PATH
        PYTHON_EXPORT_PT
        NCNN_PARAM
        NCNN_BIN
        PNNX_PARAM
        PNNX_BIN
        PNNX_PY
        PNNX_ONNX
        NCNN_OPT_PARAM
        NCNN_OPT_BIN
    )
    local var_name=""
    if [ "${#PRECISIONS[@]}" -eq 1 ]; then
        return 0
    fi

    for var_name in "${explicit_vars[@]}"; do
        if [ -n "${!var_name:-}" ]; then
            echo "[export-ncnn] 显式设置 $var_name 时, EXPORT_PRECISIONS 只能包含一个精度."
            exit 1
        fi
    done
}

collect_precisions
ensure_single_precision_overrides

if [ "$EXPORT_NCNN_MODE" = "cli" ]; then
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
        echo "[export-ncnn] 或先执行: bash scripts/export/install_ncnn_tools.sh"
        exit 1
    fi
elif [ "$EXPORT_NCNN_MODE" != "python" ]; then
    echo "[export-ncnn] EXPORT_NCNN_MODE 仅支持 cli 或 python, 当前值: $EXPORT_NCNN_MODE"
    exit 1
fi

cd "$SHMTU_MODEL_ROOT"
echo "[export-ncnn] checkpoint  = $CHECKPOINT"
for raw_precision in "${PRECISIONS[@]}"; do
    precision="$(normalize_precision "$raw_precision")"
    if [ "$precision" = "fp16" ]; then
        pnnx_fp16=1
    else
        pnnx_fp16=0
    fi

    torchscript_path="${TORCHSCRIPT_PATH:-$EXPORT_DIR/$MODEL_NAME.$precision.ts.pt}"
    python_export_pt="${PYTHON_EXPORT_PT:-$EXPORT_DIR/$MODEL_NAME.$precision.pt}"
    ncnn_param="${NCNN_PARAM:-$EXPORT_DIR/$MODEL_NAME.$precision.param}"
    ncnn_bin="${NCNN_BIN:-$EXPORT_DIR/$MODEL_NAME.$precision.bin}"
    pnnx_param="${PNNX_PARAM:-$EXPORT_DIR/$MODEL_NAME.$precision.pnnx.param}"
    pnnx_bin="${PNNX_BIN:-$EXPORT_DIR/$MODEL_NAME.$precision.pnnx.bin}"
    pnnx_py="${PNNX_PY:-$EXPORT_DIR/${MODEL_NAME}.${precision}_pnnx.py}"
    pnnx_onnx="${PNNX_ONNX:-$EXPORT_DIR/$MODEL_NAME.$precision.pnnx.onnx}"
    ncnn_opt_param="${NCNN_OPT_PARAM:-$EXPORT_DIR/$MODEL_NAME.$precision.opt.param}"
    ncnn_opt_bin="${NCNN_OPT_BIN:-$EXPORT_DIR/$MODEL_NAME.$precision.opt.bin}"

    case "$EXPORT_NCNN_MODE" in
        cli)
            if [ "$REBUILD_TORCHSCRIPT" = "1" ] || [ ! -f "$torchscript_path" ]; then
                CHECKPOINT="$CHECKPOINT" \
                EXPORT_DIR="$EXPORT_DIR" \
                MODEL_NAME="$MODEL_NAME" \
                OUTPUT="$torchscript_path" \
                IMAGE_SIZE_H="$IMAGE_SIZE_H" \
                IMAGE_SIZE_W="$IMAGE_SIZE_W" \
                    bash "$SCRIPT_DIR/export_torchscript.sh"
            fi

            echo "[export-ncnn] mode        = cli"
            echo "[export-ncnn] precision   = $precision"
            echo "[export-ncnn] torchscript = $torchscript_path"
            echo "[export-ncnn] pnnx        = $PNNX_BIN_PATH"
            echo "[export-ncnn] ncnn param  = $ncnn_param"
            echo "[export-ncnn] ncnn bin    = $ncnn_bin"

            "$PNNX_BIN_PATH" "$torchscript_path" \
                "inputshape=[1,1,${IMAGE_SIZE_H},${IMAGE_SIZE_W}]" \
                "fp16=${pnnx_fp16}" \
                "optlevel=${PNNX_OPTLEVEL}" \
                "pnnxparam=$pnnx_param" \
                "pnnxbin=$pnnx_bin" \
                "pnnxpy=$pnnx_py" \
                "pnnxonnx=$pnnx_onnx" \
                "ncnnparam=$ncnn_param" \
                "ncnnbin=$ncnn_bin"
            ;;
        python)
            echo "[export-ncnn] mode        = python"
            echo "[export-ncnn] precision   = $precision"
            echo "[export-ncnn] output pt   = $python_export_pt"
            echo "[export-ncnn] ncnn param  = $ncnn_param"
            echo "[export-ncnn] ncnn bin    = $ncnn_bin"
            "$SHMTU_PYTHON" -m cas_ocr_model.export.export_ncnn \
                --checkpoint "$CHECKPOINT" \
                --output "$python_export_pt" \
                --image-size-h "$IMAGE_SIZE_H" \
                --image-size-w "$IMAGE_SIZE_W" \
                --precision "$precision" \
                --optlevel "$PNNX_OPTLEVEL" \
                --ncnn-param "$ncnn_param" \
                --ncnn-bin "$ncnn_bin" \
                --pnnx-param "$pnnx_param" \
                --pnnx-bin "$pnnx_bin" \
                --pnnx-py "$pnnx_py" \
                --pnnx-onnx "$pnnx_onnx"
            ;;
    esac

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
            prepend_library_path "$(cd "$(dirname "$NCNNOPTIMIZE_BIN_PATH")/../lib" 2>/dev/null && pwd || true)"
            echo "[export-ncnn] ncnnoptimize = $NCNNOPTIMIZE_BIN_PATH"
            "$NCNNOPTIMIZE_BIN_PATH" \
                "$ncnn_param" \
                "$ncnn_bin" \
                "$ncnn_opt_param" \
                "$ncnn_opt_bin" \
                "$NCNNOPTIMIZE_FLAG"
        else
            echo "[export-ncnn] 未找到 ncnnoptimize，跳过优化."
        fi
    fi
done

EXPORT_ROOT="$EXPORT_DIR" \
RUN_DIR="$RUN_DIR" \
CONFIG="$CONFIG" \
    bash "$SCRIPT_DIR/generate_release_digest.sh"

if [ "$EXPORT_DIR" != "$EXPORT_ROOT" ]; then
    rm -f "$EXPORT_ROOT/SHA256SUMS.txt"
fi
