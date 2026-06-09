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
CHECKPOINT="${CHECKPOINT:-$RUN_DIR/best.pt}"
EXPORT_ROOT="${EXPORT_ROOT:-$RUN_DIR/export}"
EXPORT_DIR="${EXPORT_DIR:-$EXPORT_ROOT/onnx}"
EXPORT_DEVICE="${EXPORT_DEVICE:-auto}"
IMAGE_SIZE_H="${IMAGE_SIZE_H:-64}"
IMAGE_SIZE_W="${IMAGE_SIZE_W:-192}"
OPSET="${OPSET:-17}"
LEGACY_EXPORTER="${LEGACY_EXPORTER:-1}"
DYNAMIC_BATCH="${DYNAMIC_BATCH:-1}"

mkdir -p "$EXPORT_DIR"

if [ ! -f "$CHECKPOINT" ]; then
    echo "[export-onnx] checkpoint 不存在: $CHECKPOINT"
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
            echo "[export-onnx] 不支持的精度: $1" >&2
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
        echo "[export-onnx] 未解析到任何导出精度." >&2
        exit 1
    fi
}

collect_precisions

if [ -n "${OUTPUT:-}" ] && [ "${#PRECISIONS[@]}" -ne 1 ]; then
    echo "[export-onnx] 显式设置 OUTPUT 时, EXPORT_PRECISIONS 只能包含一个精度."
    exit 1
fi

cd "$SHMTU_MODEL_ROOT"
echo "[export-onnx] checkpoint = $CHECKPOINT"
for raw_precision in "${PRECISIONS[@]}"; do
    precision="$(normalize_precision "$raw_precision")"
    output="${OUTPUT:-$EXPORT_DIR/$MODEL_NAME.$precision.onnx}"

    ARGS=(
        -m cas_ocr_model.trainer.export
        --checkpoint "$CHECKPOINT"
        --output "$output"
        --image-size-h "$IMAGE_SIZE_H"
        --image-size-w "$IMAGE_SIZE_W"
        --opset "$OPSET"
        --precision "$precision"
        --device "$EXPORT_DEVICE"
    )

    if [ "$DYNAMIC_BATCH" = "1" ]; then
        ARGS+=(--dynamic-batch)
    fi

    if [ "$LEGACY_EXPORTER" = "1" ]; then
        ARGS+=(--legacy-exporter)
    fi

    echo "[export-onnx] precision  = $precision"
    echo "[export-onnx] output     = $output"
    "$SHMTU_PYTHON" "${ARGS[@]}"
done

EXPORT_ROOT="$EXPORT_DIR" \
RUN_DIR="$RUN_DIR" \
CONFIG="$CONFIG" \
    bash "$SCRIPT_DIR/generate_release_digest.sh"

if [ "$EXPORT_DIR" != "$EXPORT_ROOT" ]; then
    rm -f "$EXPORT_ROOT/SHA256SUMS.txt"
fi
