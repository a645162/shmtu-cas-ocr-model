from __future__ import annotations

import argparse
import json
from pathlib import Path

ALLOWED_SUFFIXES = {".pt", ".pth", ".onnx", ".param", ".bin", ".txt", ".json"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="基于 model-assets.json 生成 release 上传清单")
    p.add_argument("--output-root", required=True)
    p.add_argument("--output", required=True, help="输出 TSV 路径")
    return p.parse_args()


def resolve_release_path(output_root: Path, rel_path: Path) -> Path:
    abs_path = (output_root / rel_path).resolve()
    try:
        abs_path.relative_to(output_root)
    except ValueError as exc:
        raise SystemExit(f"release asset escapes output root: {rel_path}") from exc
    return abs_path


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    manifest_path = output_root / "model-assets.json"
    output_path = Path(args.output).expanduser().resolve()

    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    uploads: list[tuple[Path, str]] = [(manifest_path, manifest_path.name)]

    for artifact in raw.get("artifacts", []):
        for file_info in artifact.get("files", []):
            rel_path = Path(file_info["path"])
            abs_path = resolve_release_path(output_root, rel_path)
            if abs_path.suffix not in ALLOWED_SUFFIXES:
                raise SystemExit(f"unexpected release asset suffix: {abs_path.name}")
            uploads.append((abs_path, file_info.get("release_asset_name", abs_path.name)))

    for digest in raw.get("digests", []):
        rel_path = Path(digest["path"])
        abs_path = resolve_release_path(output_root, rel_path)
        if abs_path.suffix not in ALLOWED_SUFFIXES:
            raise SystemExit(f"unexpected release digest suffix: {abs_path.name}")
        uploads.append((abs_path, digest.get("release_asset_name", abs_path.name)))

    seen: set[Path] = set()
    lines: list[str] = []
    for abs_path, release_name in uploads:
        if abs_path in seen:
            continue
        seen.add(abs_path)
        if not abs_path.is_file():
            raise SystemExit(f"release asset missing: {abs_path}")
        lines.append(f"{abs_path}\t{release_name}")

    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


if __name__ == "__main__":
    main()
