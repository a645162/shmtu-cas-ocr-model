#!/usr/bin/env bash
# 公共环境变量 — 命名严格对齐 shmtu-cas-rs 的 shmtu-cas-cli (clap env = "...")
# 详见 vendor/shmtu-cas-rs/Core/shmtu-cas-cli/src/main.rs
#
# 用户可 export 覆盖:
#   SHMTU_OCR_HTTP_URL      RESTful OCR base url       (对齐 clap env "SHMTU_OCR_HTTP_URL")
#   SHMTU_OCR_HOST          TCP OCR host               (对齐 clap env "SHMTU_OCR_HOST")
#   SHMTU_OCR_PORT          TCP OCR port               (对齐 clap env "SHMTU_OCR_PORT", default 21601)
#   SHMTU_HTTP_HOST         HTTP server bind host      (对齐 server_config.cpp)
#   SHMTU_TCP_HOST          TCP  server bind host      (对齐 server_config.cpp)
#   SHMTU_HTTP_PORT         HTTP server bind port      (对齐 server_config.cpp, default 21600)
#   SHMTU_TCP_PORT          TCP  server bind port      (对齐 server_config.cpp, default 21601)
#   SHMTU_MODEL_DIR         模型目录                   (对齐 server_config.cpp)
#   SHMTU_DATASET_ROOT      数据集根目录
#   SHMTU_RUNS_ROOT         runs 根目录
#   SHMTU_PROFILE_NAME      profile 名称 (如 8gpu_ddp)
#   SHMTU_RUN_DATE_TIME     训练输出目录名中的日期时间 (如 20260608_153000)
#   SHMTU_RUN_DIR           显式指定某个具体 run 目录 (可选, 优先级最高)
#   SHMTU_WEIGHTS_DIR       PyTorch 权重缓存
#   SHMTU_NUM_GPUS          训练用 GPU 数
#   SHMTU_PYTHON            Python 解释器
#   SHMTU_DISABLE_WANDB     设为 1/true/yes/on 时禁用训练自动接入 wandb
#   SHMTU_DYNAMO_BACKEND    accelerate dynamo backend, 默认 inductor; 可设为 no 回退

# ---- 模型根 / src ----
export SHMTU_MODEL_ROOT="${SHMTU_MODEL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export SHMTU_SRC="${SHMTU_SRC:-$SHMTU_MODEL_ROOT/src}"

# ---- 采集 / 推理: 后端 OCR 地址 (对齐 clap env) ----
# shmtu-ocr-server C++ server.h: http_port=21600, tcp_port=21601
# shmtu-cas-cli 默认 SHMTU_OCR_HTTP_URL="http://127.0.0.1:5000" (cli 端, 不是 server 端)
# 我们对齐 C++ server 实际监听端口:
export SHMTU_OCR_HOST="${SHMTU_OCR_HOST:-127.0.0.1}"
export SHMTU_OCR_PORT="${SHMTU_OCR_PORT:-21601}"
export SHMTU_HTTP_HOST="${SHMTU_HTTP_HOST:-0.0.0.0}"
export SHMTU_TCP_HOST="${SHMTU_TCP_HOST:-0.0.0.0}"
export SHMTU_HTTP_PORT="${SHMTU_HTTP_PORT:-21600}"
export SHMTU_TCP_PORT="${SHMTU_TCP_PORT:-21601}"

# 自动拼接 SHMTU_OCR_HTTP_URL: 用户没显式 export 时,
# 从 SHMTU_OCR_HOST:SHMTU_HTTP_PORT 拼出 http://host:port
# 例: export SHMTU_OCR_HOST=127.0.0.1
#     -> SHMTU_OCR_HTTP_URL=http://127.0.0.1:21600
if [ -z "${SHMTU_OCR_HTTP_URL:-}" ]; then
    export SHMTU_OCR_HTTP_URL="http://${SHMTU_OCR_HOST}:${SHMTU_HTTP_PORT}"
fi
# 同样, SHMTU_OCR_PORT 没设时, 用 SHMTU_TCP_PORT 兜底
if [ -z "${SHMTU_OCR_PORT:-}" ]; then
    export SHMTU_OCR_PORT="${SHMTU_TCP_PORT}"
fi
export SHMTU_MODEL_DIR="${SHMTU_MODEL_DIR:-$SHMTU_MODEL_ROOT/weights}"

# ---- 路径 ----
export SHMTU_DATASET_ROOT="${SHMTU_DATASET_ROOT:-$SHMTU_MODEL_ROOT/dataset}"
export SHMTU_RUNS_ROOT="${SHMTU_RUNS_ROOT:-$SHMTU_MODEL_ROOT/runs}"
export SHMTU_PROFILE_NAME="${SHMTU_PROFILE_NAME:-${PROFILE_NAME:-}}"
export SHMTU_RUN_DATE_TIME="${SHMTU_RUN_DATE_TIME:-${RUN_DATE_TIME:-}}"
export SHMTU_RUN_DIR="${SHMTU_RUN_DIR:-${RUN_DIR:-}}"
if [ -n "${SHMTU_PROFILE_NAME:-}" ]; then
    export SHMTU_PROFILE_DIR="${SHMTU_PROFILE_DIR:-$SHMTU_RUNS_ROOT/$SHMTU_PROFILE_NAME}"
else
    export SHMTU_PROFILE_DIR="${SHMTU_PROFILE_DIR:-}"
fi
export SHMTU_WEIGHTS_DIR="${SHMTU_WEIGHTS_DIR:-$SHMTU_MODEL_DIR}"
export SHMTU_NUM_GPUS="${SHMTU_NUM_GPUS:-8}"
export SHMTU_PYTHON="${SHMTU_PYTHON:-python3}"
# inductor or no
export SHMTU_DYNAMO_BACKEND="${SHMTU_DYNAMO_BACKEND:-no}"

# 仓库根 (lib shmtu-cas-python 路径)
export SHMTU_REPO_ROOT="${SHMTU_REPO_ROOT:-$(cd "$SHMTU_MODEL_ROOT/../.." && pwd)}"
export PYTHONPATH="${PYTHONPATH:-}:$SHMTU_SRC:$SHMTU_REPO_ROOT/Lib/shmtu-cas-python/src"

# 采集阶段 OCR 后端选择 (restful | tcp | pytorch)
export SHMTU_BACKEND="${SHMTU_BACKEND:-restful}"

if [ "${SHMTU_ENV_SILENT:-0}" != "1" ]; then
    echo "[env] SHMTU_MODEL_ROOT    = $SHMTU_MODEL_ROOT"
    echo "[env] SHMTU_DATASET_ROOT   = $SHMTU_DATASET_ROOT"
    echo "[env] SHMTU_RUNS_ROOT      = $SHMTU_RUNS_ROOT"
    echo "[env] SHMTU_PROFILE_NAME   = ${SHMTU_PROFILE_NAME:-<unset>}"
    echo "[env] SHMTU_PROFILE_DIR    = ${SHMTU_PROFILE_DIR:-<unset>}"
    echo "[env] SHMTU_RUN_DIR        = ${SHMTU_RUN_DIR:-<auto/latest>}"
    echo "[env] SHMTU_BACKEND        = $SHMTU_BACKEND"
    echo "[env] SHMTU_OCR_HTTP_URL   = $SHMTU_OCR_HTTP_URL"
    echo "[env] SHMTU_OCR_HOST:PORT  = $SHMTU_OCR_HOST:$SHMTU_OCR_PORT"
    echo "[env] SHMTU_NUM_GPUS       = $SHMTU_NUM_GPUS"
    echo "[env] SHMTU_DYNAMO_BACKEND = $SHMTU_DYNAMO_BACKEND"
fi
