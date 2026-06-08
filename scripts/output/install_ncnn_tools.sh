#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../env.sh"

if ! command -v unzip >/dev/null 2>&1; then
    echo "[install-ncnn] 需要 unzip, 请先安装系统包."
    exit 1
fi

if command -v curl >/dev/null 2>&1; then
    DOWNLOADER="curl"
elif command -v wget >/dev/null 2>&1; then
    DOWNLOADER="wget"
else
    echo "[install-ncnn] 需要 curl 或 wget."
    exit 1
fi

INSTALL_DIR="${INSTALL_DIR:-$SHMTU_MODEL_ROOT/3rdparty/ncnn}"
RELEASE_API="${RELEASE_API:-https://api.github.com/repos/Tencent/ncnn/releases/latest}"
FORCE_CLEAN="${FORCE_CLEAN:-0}"

OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"

asset_selector() {
    case "$OS_NAME:$ARCH_NAME" in
        Linux:x86_64)
            echo "ubuntu.*\\.zip$"
            ;;
        Linux:aarch64|Linux:arm64)
            echo "(ubuntu|linux).*(aarch64|arm64).*\\.zip$"
            ;;
        Darwin:x86_64)
            echo "macos.*\\.zip$"
            ;;
        Darwin:arm64)
            echo "macos.*(arm64|apple).*\\.zip$"
            ;;
        *)
            return 1
            ;;
    esac
}

if ! ASSET_REGEX="$(asset_selector)"; then
    echo "[install-ncnn] 暂不支持自动匹配平台: $OS_NAME $ARCH_NAME"
    echo "[install-ncnn] 请手动下载后设置 PNNX/NCNNOPTIMIZE."
    exit 1
fi

if [ "$FORCE_CLEAN" = "1" ] && [ -d "$INSTALL_DIR" ]; then
    rm -rf "$INSTALL_DIR"
fi

mkdir -p "$INSTALL_DIR"
TMP_DIR="$(mktemp -d)"
cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

RELEASE_JSON="$TMP_DIR/release.json"
ARCHIVE_PATH="$TMP_DIR/ncnn.zip"

echo "[install-ncnn] release api = $RELEASE_API"
echo "[install-ncnn] install dir = $INSTALL_DIR"
echo "[install-ncnn] platform    = $OS_NAME $ARCH_NAME"

if [ "$DOWNLOADER" = "curl" ]; then
    curl -LfsS "$RELEASE_API" -o "$RELEASE_JSON"
else
    wget -qO "$RELEASE_JSON" "$RELEASE_API"
fi

ASSET_URL="$(
    RELEASE_JSON="$RELEASE_JSON" ASSET_REGEX="$ASSET_REGEX" python - <<'PY'
import json
import os
import re
import sys

path = os.environ["RELEASE_JSON"]
pattern = re.compile(os.environ["ASSET_REGEX"], re.IGNORECASE)

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

assets = data.get("assets", [])
for asset in assets:
    name = asset.get("name", "")
    if pattern.search(name):
        print(asset.get("browser_download_url", ""))
        sys.exit(0)

for asset in assets:
    name = asset.get("name", "")
    if name.lower().endswith(".zip"):
        print(asset.get("browser_download_url", ""))
        sys.exit(0)

sys.exit(1)
PY
)"

if [ -z "$ASSET_URL" ]; then
    echo "[install-ncnn] 未找到匹配当前平台的 ncnn 发行包."
    exit 1
fi

echo "[install-ncnn] download    = $ASSET_URL"
if [ "$DOWNLOADER" = "curl" ]; then
    curl -LfsS "$ASSET_URL" -o "$ARCHIVE_PATH"
else
    wget -qO "$ARCHIVE_PATH" "$ASSET_URL"
fi

UNZIP_DIR="$TMP_DIR/unzip"
mkdir -p "$UNZIP_DIR"
unzip -q "$ARCHIVE_PATH" -d "$UNZIP_DIR"

SRC_ROOT="$(find "$UNZIP_DIR" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
if [ -z "$SRC_ROOT" ]; then
    echo "[install-ncnn] 解压结果异常，未找到根目录."
    exit 1
fi

mkdir -p "$INSTALL_DIR"
find "$INSTALL_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
cp -a "$SRC_ROOT"/. "$INSTALL_DIR"/

for tool in "$INSTALL_DIR/bin/pnnx" "$INSTALL_DIR/bin/ncnnoptimize"; do
    if [ -f "$tool" ]; then
        chmod +x "$tool"
    fi
done

echo "[install-ncnn] installed:"
find "$INSTALL_DIR/bin" -maxdepth 1 \( -name 'pnnx' -o -name 'ncnnoptimize' \) -printf '  %M %p\n' 2>/dev/null || true
