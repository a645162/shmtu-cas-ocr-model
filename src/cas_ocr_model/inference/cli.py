"""推理 CLI (predict / evaluate / benchmark).

用法:
    # 1) 单图 / 批量目录预测
    python -m cas_ocr_model.inference.cli --mode predict \\
        --checkpoint runs/exp1/best.pt \\
        --image dataset/00000007.jpg

    # 2) 在带 ground-truth 的数据集上计算指标
    python -m cas_ocr_model.inference.cli --mode evaluate \\
        --checkpoint runs/exp1/best.pt \\
        --gt-dir dataset --output report.json

    # 3) 性能 benchmark
    python -m cas_ocr_model.inference.cli --mode benchmark \\
        --checkpoint runs/exp1/best.pt \\
        --num-samples 500 --batch-sizes 1,8,32,128
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cas_ocr_model.common.console import tag_print, get_console

from .inference import CaptchaInferencer, InferencerConfig


def build_backend(args: argparse.Namespace):
    from . import backends

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


def resolve_defaults_from_checkpoint(args: argparse.Namespace) -> argparse.Namespace:
    if args.backend != "pytorch" or not args.checkpoint:
        return args

    try:
        import torch
    except ImportError:
        return args

    raw = torch.load(args.checkpoint, map_location="cpu")
    cfg = raw.get("config", {}) if isinstance(raw, dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    train_cfg = cfg.get("train", {}) if isinstance(cfg, dict) else {}

    if args.backbone is None:
        args.backbone = str(model_cfg.get("backbone", "resnet18"))
    if args.image_size_h is None:
        args.image_size_h = int(data_cfg.get("image_size_h", 64))
    if args.image_size_w is None:
        args.image_size_w = int(data_cfg.get("image_size_w", 192))
    if args.threshold is None:
        args.threshold = int(data_cfg.get("threshold", 200))
    if args.binarize_mode is None:
        args.binarize_mode = str(data_cfg.get("binarize_mode", "min_channel_otsu"))
    if args.adaptive_block_size is None:
        args.adaptive_block_size = int(data_cfg.get("adaptive_block_size", 25))
    if args.adaptive_c is None:
        args.adaptive_c = int(data_cfg.get("adaptive_c", 15))
    if args.batch_size is None:
        args.batch_size = min(128, int(train_cfg.get("per_device_batch_size", 32)))
    return args


def fill_builtin_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.backbone is None:
        args.backbone = "resnet18"
    if args.image_size_h is None:
        args.image_size_h = 64
    if args.image_size_w is None:
        args.image_size_w = 192
    if args.threshold is None:
        args.threshold = 200
    if args.binarize_mode is None:
        args.binarize_mode = "min_channel_otsu"
    if args.adaptive_block_size is None:
        args.adaptive_block_size = 25
    if args.adaptive_c is None:
        args.adaptive_c = 15
    if args.batch_size is None:
        args.batch_size = 32
    return args


def common_parser(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", default="pytorch", choices=["pytorch", "onnx", "ncnn"])
    p.add_argument("--checkpoint", default=None, help="PyTorch backend 使用的 best.pt")
    p.add_argument("--onnx", default=None, help="ONNX backend 使用的 .onnx")
    p.add_argument("--ncnn-param", default=None, help="ncnn backend 使用的 .param")
    p.add_argument("--ncnn-bin", default=None, help="ncnn backend 使用的 .bin")
    p.add_argument("--backbone", default=None)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    p.add_argument("--image-size-h", type=int, default=None)
    p.add_argument("--image-size-w", type=int, default=None)
    p.add_argument("--threshold", type=int, default=None)
    p.add_argument("--binarize-mode", default=None)
    p.add_argument("--adaptive-block-size", type=int, default=None)
    p.add_argument("--adaptive-c", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)


def cmd_predict(args: argparse.Namespace) -> int:
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
    inferencer = CaptchaInferencer(backend=backend, config=cfg)

    records: list[dict] = []

    if args.image:
        result = inferencer.predict_one(args.image)
        record = {
            "image": str(args.image),
            "expression": result.expression,
            "digit_left": result.digit_left,
            "operator": result.operator,
            "digit_right": result.digit_right,
            "result": result.result,
            "confidence": result.confidence,
        }
        records.append(record)
        print(json.dumps(record, ensure_ascii=False, indent=2))

    if args.dir:
        items = inferencer.predict_dir(args.dir, pattern=args.pattern, limit=args.limit)
        for name, r in items:
            record = {
                "image": name,
                "expression": r.expression,
                "digit_left": r.digit_left,
                "operator": r.operator,
                "digit_right": r.digit_right,
                "result": r.result,
                "confidence": r.confidence,
            }
            records.append(record)
            print(json.dumps(record, ensure_ascii=False))
        tag_print("done", f"{len(items)} samples")

    if args.output and records:
        Path(args.output).write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tag_print("saved", str(args.output))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluate import evaluate, metrics_to_dict

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
    inferencer = CaptchaInferencer(backend=backend, config=cfg)

    metrics = evaluate(inferencer, dataset_dir=args.gt_dir, pattern=args.pattern, limit=args.limit)
    report = metrics_to_dict(metrics)

    # 命令行可读摘要
    tag_print(
        "eval",
        f"n={metrics.n_samples} "
        f"acc_dl={metrics.acc_digit_left:.4f} "
        f"acc_op={metrics.acc_operator:.4f} "
        f"acc_dr={metrics.acc_digit_right:.4f} "
        f"acc_full={metrics.acc_expression:.4f} "
        f"acc_eval={metrics.acc_eval_result:.4f} "
        f"ece={metrics.ece:.4f}",
    )

    if args.output:
        Path(args.output).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tag_print("saved", str(args.output))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from .benchmark import benchmark, print_report, report_to_dict

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
    inferencer = CaptchaInferencer(backend=backend, config=cfg)

    bs_list = tuple(int(x) for x in args.batch_sizes.split(",")) if args.batch_sizes else (1, 8, 32, 128)
    rpt = benchmark(
        inferencer,
        n_samples=args.num_samples,
        warmup=args.warmup,
        batch_sizes=bs_list,
        backend_name=args.backend,
    )
    print_report(rpt)

    if args.output:
        Path(args.output).write_text(
            json.dumps(report_to_dict(rpt), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tag_print("saved", str(args.output))
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="CAS CAPTCHA 3-head 推理 (predict / evaluate / benchmark)")
    p.add_argument("--mode", choices=["predict", "evaluate", "benchmark"], default="predict")
    common_parser(p)

    # predict
    p.add_argument("--image", default=None, help="单图推理 (predict 模式)")
    p.add_argument("--dir", default=None, help="批量目录 (predict 模式)")
    p.add_argument("--pattern", default="*.jpg")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output", default=None, help="结果/报告 json 输出路径")

    # evaluate
    p.add_argument("--gt-dir", default=None, help="evaluate 模式: 带 ground truth 的目录")

    # benchmark
    p.add_argument("--num-samples", type=int, default=500)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--batch-sizes", default="1,8,32,128", help="逗号分隔")

    args = p.parse_args()
    args = resolve_defaults_from_checkpoint(args)
    args = fill_builtin_defaults(args)

    if args.mode == "predict":
        if not args.image and not args.dir:
            p.error("predict 模式必须指定 --image 或 --dir 之一")
        sys.exit(cmd_predict(args))
    elif args.mode == "evaluate":
        if not args.gt_dir:
            p.error("evaluate 模式必须指定 --gt-dir")
        sys.exit(cmd_evaluate(args))
    elif args.mode == "benchmark":
        sys.exit(cmd_benchmark(args))


if __name__ == "__main__":
    main()
