#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec env BACKEND=ncnn DEVICE=cpu bash "$SCRIPT_DIR/bench_single.sh" "$@"
