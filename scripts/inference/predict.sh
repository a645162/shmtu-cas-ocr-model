#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

if shmtu_inference_is_help_request "$@"; then
    cd "$SHMTU_MODEL_ROOT"
    exec "$SHMTU_PYTHON" -m cas_ocr_model.inference.cli --help
fi

shmtu_inference_init
shmtu_inference_build_backend_args

IMAGE="${IMAGE:-}"
DIR="${DIR:-}"
PATTERN="${PATTERN:-*.jpg}"
LIMIT="${LIMIT:-}"
OUTPUT="${OUTPUT:-}"

echo "[predict] backend    = $BACKEND"
echo "[predict] run dir    = $RUN_DIR"
echo "[predict] device     = $DEVICE"

cd "$SHMTU_MODEL_ROOT"
cmd=(
    "$SHMTU_PYTHON" -m cas_ocr_model.inference.cli
    --mode predict
    "${BACKEND_ARGS[@]}"
)
if [ -n "$IMAGE" ]; then
    cmd+=(--image "$IMAGE")
fi
if [ -n "$DIR" ]; then
    cmd+=(--dir "$DIR" --pattern "$PATTERN")
fi
if [ -n "$LIMIT" ]; then
    cmd+=(--limit "$LIMIT")
fi
if [ -n "$OUTPUT" ]; then
    cmd+=(--output "$OUTPUT")
fi
cmd+=("$@")
"${cmd[@]}"
