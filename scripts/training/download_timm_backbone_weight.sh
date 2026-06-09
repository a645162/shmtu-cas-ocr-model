#!/usr/bin/env bash
# 通过 huggingface_hub 预拉取 timm backbone 预训练权重到本地 cache
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../env.sh"

HF_REPO_ID="${HF_REPO_ID:-timm/mobilenetv4_conv_small.e2400_r224_in1k}"
HF_FILENAME="${HF_FILENAME:-model.safetensors}"
HF_REVISION="${HF_REVISION:-main}"
HF_LOCAL_DIR="${HF_LOCAL_DIR:-}"
HF_FORCE_DOWNLOAD="${HF_FORCE_DOWNLOAD:-0}"
export HF_REPO_ID HF_FILENAME HF_REVISION HF_LOCAL_DIR HF_FORCE_DOWNLOAD

# huggingface_hub 1.17.0 默认 timeout=10s，国内链路偏短，这里统一放宽
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"

echo "[hf] repo_id         = $HF_REPO_ID"
echo "[hf] filename        = $HF_FILENAME"
echo "[hf] revision        = $HF_REVISION"
echo "[hf] local_dir       = ${HF_LOCAL_DIR:-<cache only>}"
echo "[hf] force_download  = $HF_FORCE_DOWNLOAD"
echo "[hf] etag_timeout    = $HF_HUB_ETAG_TIMEOUT"
echo "[hf] download_timeout= $HF_HUB_DOWNLOAD_TIMEOUT"

cd "$SHMTU_MODEL_ROOT"
"$SHMTU_PYTHON" - <<'PY'
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

repo_id = os.environ["HF_REPO_ID"]
filename = os.environ["HF_FILENAME"]
revision = os.environ["HF_REVISION"]
local_dir_raw = os.environ.get("HF_LOCAL_DIR", "").strip()
force_download = os.environ.get("HF_FORCE_DOWNLOAD", "0").strip().lower() in {"1", "true", "yes", "on"}

kwargs = {
    "repo_id": repo_id,
    "filename": filename,
    "revision": revision,
    "force_download": force_download,
}
if local_dir_raw:
    local_dir = Path(local_dir_raw).expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    kwargs["local_dir"] = str(local_dir)
    kwargs["local_dir_use_symlinks"] = False

path = Path(hf_hub_download(**kwargs)).resolve()
print(f"[hf] downloaded: {path}")
if path.is_file():
    print(f"[hf] size: {path.stat().st_size} bytes")
PY

echo "[hf] done."
