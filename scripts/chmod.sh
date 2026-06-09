#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while IFS= read -r -d '' file; do
  chmod +x "$file"
  echo "chmod +x $file"
done < <(find "$script_dir" -type f -name '*.sh' -print0 | sort -z)
