#!/usr/bin/env bash
set -euo pipefail

if [ -n "${SHMTU_MAIN_PROCESS_PORT:-}" ]; then
    printf '%s\n' "$SHMTU_MAIN_PROCESS_PORT"
    exit 0
fi

if [ -n "${MASTER_PORT:-}" ]; then
    printf '%s\n' "$MASTER_PORT"
    exit 0
fi

if command -v torch_ddp_port >/dev/null 2>&1; then
    torch_ddp_port
    exit 0
fi

printf '29500\n'
