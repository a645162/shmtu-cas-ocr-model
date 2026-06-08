#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
SHMTU_ENV_SILENT=1 source "$SCRIPT_DIR/env.sh"

usage() {
    cat <<'EOF'
用法:
  bash scripts/run_path.sh create
  bash scripts/run_path.sh resolve

环境变量:
  SHMTU_RUNS_ROOT      runs 根目录, 默认 ./runs
  SHMTU_PROFILE_NAME   profile 名称, 如 8gpu_ddp
  SHMTU_RUN_DATE_TIME  日期时间目录名, 如 20260608_153000
  SHMTU_RUN_DIR        显式指定具体 run 目录, resolve 时优先级最高
  SHMTU_USE_LATEST     resolve 时是否默认读取 latest, 默认 1

输出:
  create  -> 打印创建后的 run 绝对路径, 并刷新 profile/latest (内容为相对路径)
  resolve -> 打印解析后的 run 绝对路径
EOF
}

die() {
    echo "[run-path] $*" >&2
    exit 1
}

require_profile() {
    if [ -z "${SHMTU_PROFILE_NAME:-}" ]; then
        die "缺少 SHMTU_PROFILE_NAME。"
    fi
}

default_date_time() {
    date '+%Y%m%d_%H%M%S'
}

resolve_explicit_run_dir() {
    if [ -z "${SHMTU_RUN_DIR:-}" ]; then
        return 1
    fi

    "$SHMTU_PYTHON" - "$SHMTU_RUN_DIR" "$SHMTU_MODEL_ROOT" <<'PY'
from pathlib import Path
import sys

run_dir = Path(sys.argv[1])
base = Path(sys.argv[2])
if not run_dir.is_absolute():
    run_dir = (base / run_dir).resolve()
else:
    run_dir = run_dir.resolve()
print(run_dir)
PY
}

create_run_dir() {
    require_profile

    local date_time="${SHMTU_RUN_DATE_TIME:-}"
    if [ -z "$date_time" ]; then
        date_time="$(default_date_time)"
    fi

    local profile_dir="$SHMTU_RUNS_ROOT/$SHMTU_PROFILE_NAME"
    local run_dir="$profile_dir/$date_time"
    mkdir -p "$run_dir"

    local latest_file="$profile_dir/latest"
    local rel_path="$date_time"
    printf '%s\n' "$rel_path" >"$latest_file"

    "$SHMTU_PYTHON" - "$run_dir" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
}

resolve_latest_run_dir() {
    require_profile

    local profile_dir="$SHMTU_RUNS_ROOT/$SHMTU_PROFILE_NAME"
    local latest_file="$profile_dir/latest"
    [ -f "$latest_file" ] || die "latest 不存在: $latest_file"

    local rel_path
    rel_path="$(tr -d '\r' <"$latest_file" | sed '/^[[:space:]]*$/d' | head -n 1)"
    [ -n "$rel_path" ] || die "latest 为空: $latest_file"

    "$SHMTU_PYTHON" - "$profile_dir" "$rel_path" <<'PY'
from pathlib import Path
import sys

profile_dir = Path(sys.argv[1]).resolve()
rel_path = sys.argv[2].strip()
target = (profile_dir / rel_path).resolve()
print(target)
PY
}

resolve_run_dir() {
    if explicit="$(resolve_explicit_run_dir 2>/dev/null)"; then
        printf '%s\n' "$explicit"
        return 0
    fi

    local use_latest="${SHMTU_USE_LATEST:-1}"
    if [ "$use_latest" = "1" ]; then
        resolve_latest_run_dir
        return 0
    fi

    die "未设置 SHMTU_RUN_DIR，且 SHMTU_USE_LATEST != 1，无法解析 run 目录。"
}

cmd="${1:-resolve}"
case "$cmd" in
    create)
        create_run_dir
        ;;
    resolve)
        resolve_run_dir
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage
        exit 1
        ;;
esac
