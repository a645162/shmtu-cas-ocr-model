#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cas_ocr_model.datasets.format import DatasetManifest
from cas_ocr_model.common.expression import parse_captcha_expression
from cas_ocr_model.inference import CaptchaInferencer, InferencerConfig
from cas_ocr_model.inference import backends
from cas_ocr_model.trainer.config import FullConfig, load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="从 test split 随机抽样并输出按预测表达式命名的图片"
    )
    p.add_argument("--config", default=None, help="训练配置文件路径 (yaml/toml)")
    p.add_argument("--data-root", default=None, help="含 manifest.json 的数据集目录")
    p.add_argument("--backend", default="pytorch", choices=["pytorch", "onnx", "ncnn"])
    p.add_argument("--checkpoint", default=None, help="PyTorch backend 使用的 checkpoint 路径")
    p.add_argument("--onnx", default=None, help="ONNX backend 使用的 .onnx 路径")
    p.add_argument("--ncnn-param", default=None, help="ncnn backend 使用的 .param 路径")
    p.add_argument("--ncnn-bin", default=None, help="ncnn backend 使用的 .bin 路径")
    p.add_argument("--output-dir", default="output", help="输出根目录")
    p.add_argument("--subdir", default=None, help="输出子目录名, 默认自动生成")
    p.add_argument("--n", type=int, default=20, help="随机抽样数量")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--backbone", default=None)
    p.add_argument("--image-size-h", type=int, default=None)
    p.add_argument("--image-size-w", type=int, default=None)
    p.add_argument("--threshold", type=int, default=None)
    p.add_argument("--binarize-mode", default=None)
    p.add_argument("--adaptive-block-size", type=int, default=None)
    p.add_argument("--adaptive-c", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    return p.parse_args()


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    cfg = load_config(args.config) if args.config else FullConfig()
    if args.data_root is None:
        args.data_root = cfg.data.data_root
    if args.backend == "pytorch" and args.checkpoint is None:
        args.checkpoint = str(Path(cfg.train.output_dir) / "best.pt")
    if args.backbone is None:
        args.backbone = cfg.model.backbone
    if args.image_size_h is None:
        args.image_size_h = cfg.data.image_size_h
    if args.image_size_w is None:
        args.image_size_w = cfg.data.image_size_w
    if args.threshold is None:
        args.threshold = cfg.data.threshold
    if args.binarize_mode is None:
        args.binarize_mode = cfg.data.binarize_mode
    if args.adaptive_block_size is None:
        args.adaptive_block_size = cfg.data.adaptive_block_size
    if args.adaptive_c is None:
        args.adaptive_c = cfg.data.adaptive_c
    if args.batch_size is None:
        args.batch_size = min(32, cfg.train.per_device_batch_size)
    return args


def build_backend(args: argparse.Namespace):
    availability = backends.get_backend_availability()
    class_name_map = {
        "pytorch": "PyTorchBackend",
        "onnx": "OnnxBackend",
        "ncnn": "NcnnBackend",
    }
    backend_class_name = class_name_map[args.backend]
    missing_dependency = availability[backend_class_name]
    if missing_dependency is not None:
        raise SystemExit(
            f"{args.backend} backend 不可用: 缺少依赖 `{missing_dependency}`"
        )

    if args.backend == "pytorch":
        if not args.checkpoint:
            raise SystemExit("pytorch backend 必须指定 --checkpoint")
        PyTorchBackend = backends.PyTorchBackend
        return PyTorchBackend(
            checkpoint=args.checkpoint,
            backbone=args.backbone,
            device=args.device,
        )

    if args.backend == "onnx":
        if args.device != "cpu":
            raise SystemExit("onnx backend 当前只支持 --device cpu")
        if not args.onnx:
            raise SystemExit("onnx backend 必须指定 --onnx")
        OnnxBackend = backends.OnnxBackend
        return OnnxBackend(
            onnx_path=args.onnx,
            device="cpu",
        )

    if args.device != "cpu":
        raise SystemExit("ncnn backend 当前只支持 --device cpu")
    if not args.ncnn_param or not args.ncnn_bin:
        raise SystemExit("ncnn backend 必须同时指定 --ncnn-param 和 --ncnn-bin")
    NcnnBackend = backends.NcnnBackend
    return NcnnBackend(
        param_path=args.ncnn_param,
        bin_path=args.ncnn_bin,
        device="cpu",
    )


def build_inferencer(args: argparse.Namespace) -> CaptchaInferencer:
    backend = build_backend(args)
    cfg = InferencerConfig(
        image_size_h=args.image_size_h,
        image_size_w=args.image_size_w,
        threshold=args.threshold,
        binarize_mode=args.binarize_mode,
        adaptive_block_size=args.adaptive_block_size,
        adaptive_c=args.adaptive_c,
        batch_size=args.batch_size,
    )
    return CaptchaInferencer(backend=backend, config=cfg)


def sanitize_expr_filename(expr: str, result: int | None) -> str:
    compact = "".join(expr.split())
    stem = f"{compact}={result}" if result is not None else compact
    return stem.replace("/", "_")


def choose_test_samples(data_root: Path, n: int, seed: int) -> list[Path]:
    manifest = DatasetManifest.load(data_root)
    test_files = list(manifest.splits.get("test", []))
    if not test_files:
        raise RuntimeError("manifest.json 中没有 test split")
    jpg_paths = [data_root / name for name in test_files]
    jpg_paths = [p for p in jpg_paths if p.is_file()]
    if not jpg_paths:
        raise RuntimeError("test split 没有可用图片文件")
    rng = random.Random(seed)
    sample_size = min(n, len(jpg_paths))
    return rng.sample(jpg_paths, sample_size)


def _parse_int(value: object) -> int | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _eval_parsed_expression(parsed) -> int | None:
    left = int(parsed.digit_left)
    right = int(parsed.digit_right)
    if parsed.operator == "+":
        return left + right
    if parsed.operator == "-":
        return left - right
    if parsed.operator == "*":
        return left * right
    return None


def load_ground_truth(src_path: Path) -> dict[str, object]:
    json_path = src_path.with_suffix(".json")
    if not json_path.is_file():
        return {"gt_expression": None, "gt_result": None, "gt_display": None, "ok": False}
    try:
        meta = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"gt_expression": None, "gt_result": None, "gt_display": None, "ok": False}

    expr = str(meta.get("expression", "")).strip()
    parsed = parse_captcha_expression(expr)
    if parsed is not None:
        gt_expr = f"{parsed.digit_left}{parsed.operator}{parsed.digit_right}"
        gt_result = parsed.answer
        if gt_result is None:
            gt_result = _parse_int(meta.get("answer"))
        if gt_result is None:
            gt_result = _eval_parsed_expression(parsed)
    else:
        gt_expr = expr or None
        gt_result = _parse_int(meta.get("answer"))

    gt_display = gt_expr
    if gt_expr is not None and gt_result is not None:
        gt_display = f"{gt_expr}={gt_result}"

    return {
        "gt_expression": gt_expr,
        "gt_result": gt_result,
        "gt_display": gt_display,
        "ok": True,
    }


def make_output_subdir(output_dir: Path, subdir: str | None, n: int, seed: int) -> Path:
    name = subdir or f"test_samples_n{n}_seed{seed}"
    out = output_dir / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def save_contact_sheet(records: list[dict], output_path: Path) -> None:
    if not records:
        return

    thumb_w = 192
    thumb_h = 64
    padding = 12
    caption_h = 82
    cols = min(4, len(records))
    rows = math.ceil(len(records) / cols)
    canvas_w = cols * (thumb_w + padding) + padding
    canvas_h = rows * (thumb_h + caption_h + padding) + padding
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    font_main = _load_font(14)
    font_small = _load_font(13)

    for idx, record in enumerate(records):
        row, col = divmod(idx, cols)
        x = padding + col * (thumb_w + padding)
        y = padding + row * (thumb_h + caption_h + padding)

        img = Image.open(record["source_path"]).convert("RGB")
        img = img.resize((thumb_w, thumb_h))
        canvas.paste(img, (x, y))

        status_text = "CORRECT" if record["is_correct"] else "WRONG"
        status_color = (24, 140, 62) if record["is_correct"] else (210, 48, 44)
        gt_expr = record["gt_display"] or "N/A"
        pred_expr = record["prediction_with_result"]
        conf_text = f'conf={record["confidence"]:.3f}'

        draw.text((x, y + thumb_h + 6), status_text, fill=status_color, font=font_main)
        draw.text((x + 92, y + thumb_h + 6), conf_text, fill=(60, 60, 60), font=font_small)
        draw.text((x, y + thumb_h + 28), f"GT: {gt_expr}", fill=(20, 20, 20), font=font_small)
        draw.text((x, y + thumb_h + 48), f"Pred: {pred_expr}", fill=(20, 20, 20), font=font_small)

    canvas.save(output_path)


def main() -> int:
    args = resolve_args(parse_args())
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    inferencer = build_inferencer(args)

    sample_paths = choose_test_samples(data_root, args.n, args.seed)
    results = inferencer.predict_batch(sample_paths)

    out_dir = make_output_subdir(output_dir, args.subdir, len(sample_paths), args.seed)
    name_counts: defaultdict[str, int] = defaultdict(int)
    records: list[dict] = []

    for src_path, pred in zip(sample_paths, results):
        gt = load_ground_truth(src_path)
        stem = sanitize_expr_filename(pred.expression, pred.result)
        name_counts[stem] += 1
        suffix = f"__{name_counts[stem]}" if name_counts[stem] > 1 else ""
        dst_name = f"{stem}{suffix}{src_path.suffix.lower()}"
        dst_path = out_dir / dst_name
        dst_path.write_bytes(src_path.read_bytes())

        prediction_with_result = (
            f"{pred.expression}={pred.result}" if pred.result is not None else pred.expression
        )
        gt_expression = gt["gt_expression"]
        gt_result = gt["gt_result"]
        is_correct = bool(gt["ok"]) and pred.expression == gt_expression
        if is_correct and gt_result is not None:
            is_correct = pred.result == gt_result

        records.append(
            {
                "source_image": src_path.name,
                "source_path": str(src_path),
                "saved_image": dst_name,
                "prediction": pred.expression,
                "prediction_with_result": prediction_with_result,
                "result": pred.result,
                "confidence": pred.confidence,
                "gt_expression": gt_expression,
                "gt_display": gt["gt_display"],
                "gt_result": gt["gt_result"],
                "is_correct": is_correct,
            }
        )

    save_contact_sheet(records, out_dir / "contact_sheet.jpg")

    manifest = {
        "data_root": str(data_root),
        "backend": args.backend,
        "checkpoint": args.checkpoint,
        "onnx": args.onnx,
        "ncnn_param": args.ncnn_param,
        "ncnn_bin": args.ncnn_bin,
        "n": len(records),
        "seed": args.seed,
        "output_dir": str(out_dir),
        "records": [
            {
                "source_image": r["source_image"],
                "saved_image": r["saved_image"],
                "prediction": r["prediction"],
                "prediction_with_result": r["prediction_with_result"],
                "result": r["result"],
                "confidence": r["confidence"],
                "gt_expression": r["gt_expression"],
                "gt_result": r["gt_result"],
                "is_correct": r["is_correct"],
            }
            for r in records
        ],
    }
    (out_dir / "predictions.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[saved] {out_dir}")
    print(f"[count] {len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
