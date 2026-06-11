"""多卡 DDP test 集并行推理与汇总报告.

通过 accelerate 启动后:
    * 只在 rank 间切分 test split, 不再每卡重复跑全集
    * 汇总全局 acc / confusion / ECE
    * 可选把推理结果图片复制到目录, 文件名以预测表达式/结果开头
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from cas_ocr_model.common.expression import parse_captcha_expression
from cas_ocr_model.datasets.format import DatasetManifest
from cas_ocr_model.trainer.config import DIGIT_LABELS, OPERATOR_LABELS

from .evaluate import EvalMetrics, metrics_to_dict
from .inference import CaptchaInferencer, InferencerConfig, InferenceResult


def build_backend(args: argparse.Namespace, accelerator: Accelerator):
    from .backends.pytorch_backend import PyTorchBackend

    device = str(accelerator.device) if torch.cuda.is_available() else "cpu"
    return PyTorchBackend(
        checkpoint=args.checkpoint,
        backbone=args.backbone,
        device=device,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="多卡 DDP 并行推理 test 集并汇总指标")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--backbone", default=None)
    p.add_argument("--data-root", required=True, help="含 manifest.json + test split 的目录")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output", default=None, help="汇总 JSON 报告输出路径")
    p.add_argument("--save-dir", default=None, help="可选: 保存预测图片到该目录")
    p.add_argument("--image-size-h", type=int, default=None)
    p.add_argument("--image-size-w", type=int, default=None)
    p.add_argument("--threshold", type=int, default=None)
    p.add_argument("--binarize-mode", default=None)
    p.add_argument("--adaptive-block-size", type=int, default=None)
    p.add_argument("--adaptive-c", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    return p.parse_args()


def resolve_defaults_from_checkpoint(args: argparse.Namespace) -> argparse.Namespace:
    raw = torch.load(args.checkpoint, map_location="cpu")
    cfg = raw.get("config", {}) if isinstance(raw, dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}

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
        train_cfg = cfg.get("train", {}) if isinstance(cfg, dict) else {}
        args.batch_size = min(128, int(train_cfg.get("per_device_batch_size", 64)))
    return args


def _parse_gt_expression(expr: str) -> tuple[int, str, int, int | None] | None:
    parsed = parse_captcha_expression(expr)
    if parsed is None:
        return None
    return int(parsed.digit_left), parsed.operator, int(parsed.digit_right), parsed.answer


def _safe_eval(d1: int, op: str, d2: int) -> int | None:
    try:
        if op == "+":
            return d1 + d2
        if op == "-":
            return d1 - d2
        if op == "*":
            return d1 * d2
    except Exception:
        return None
    return None


def _sanitize_prediction_filename(expr: str, result: int | None, original_stem: str) -> str:
    stem = f"{expr}={result}" if result is not None else expr
    safe = stem.replace("/", "_").replace("\\", "_").replace(" ", "")
    return f"{safe}__{original_stem}"


def load_test_samples(dataset_dir: str | Path, limit: int | None) -> list[tuple[Path, tuple[int, str, int, int | None]]]:
    root = Path(dataset_dir)
    manifest = DatasetManifest.load(root)
    test_files = list(manifest.splits.get("test", []))
    if limit is not None:
        test_files = test_files[:limit]

    samples: list[tuple[Path, tuple[int, str, int, int | None]]] = []
    for name in test_files:
        jpg = root / name
        json_path = jpg.with_suffix(".json")
        if not jpg.is_file() or not json_path.is_file():
            continue
        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        parsed = _parse_gt_expression(str(meta.get("expression", "")))
        if parsed is None:
            continue
        samples.append((jpg, parsed))
    if not samples:
        raise RuntimeError(f"no valid test samples found in {dataset_dir}")
    return samples


def shard_samples(
    samples: list[tuple[Path, tuple[int, str, int, int | None]]],
    rank: int,
    world_size: int,
) -> list[tuple[Path, tuple[int, str, int, int | None]]]:
    return samples[rank::world_size]


def save_prediction_image(
    *,
    src_path: Path,
    prediction: InferenceResult,
    save_dir: Path,
) -> str:
    save_dir.mkdir(parents=True, exist_ok=True)
    stem = _sanitize_prediction_filename(prediction.expression, prediction.result, src_path.stem)
    dst_name = f"{stem}{src_path.suffix.lower()}"
    dst_path = save_dir / dst_name
    dst_path.write_bytes(src_path.read_bytes())
    return dst_name


def evaluate_local_shard(
    inferencer: CaptchaInferencer,
    samples: list[tuple[Path, tuple[int, str, int, int | None]]],
    *,
    save_dir: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    digit2idx = {s: i for i, s in enumerate(DIGIT_LABELS)}
    op2idx = {s: i for i, s in enumerate(OPERATOR_LABELS)}

    results = inferencer.predict_batch([p for p, _ in samples]) if samples else []

    conf_dl = np.zeros((10, 10), dtype=np.int64)
    conf_op = np.zeros((len(OPERATOR_LABELS), len(OPERATOR_LABELS)), dtype=np.int64)
    conf_dr = np.zeros((10, 10), dtype=np.int64)

    n = n_dl = n_op = n_dr = n_full = n_eval = n_answered = 0
    sum_conf = 0.0
    n_bins = 15
    bin_counts = np.zeros(n_bins, dtype=np.int64)
    bin_conf_sums = np.zeros(n_bins, dtype=np.float64)
    bin_correct_sums = np.zeros(n_bins, dtype=np.float64)
    records: list[dict[str, Any]] = []

    for (jpg, (d1, op, d2, ans)), pred in zip(samples, results):
        n += 1
        pred_dl = int(pred.digit_left)
        pred_op = pred.operator
        pred_dr = int(pred.digit_right)

        ok_dl = pred_dl == d1
        ok_op = pred_op == op
        ok_dr = pred_dr == d2
        ok_full = ok_dl and ok_op and ok_dr

        n_dl += int(ok_dl)
        n_op += int(ok_op)
        n_dr += int(ok_dr)
        n_full += int(ok_full)

        eval_v = _safe_eval(pred_dl, pred_op, pred_dr)
        if ans is not None:
            n_answered += 1
            n_eval += int(eval_v is not None and eval_v == ans)

        conf_dl[d1, pred_dl] += 1
        conf_op[op2idx[op], op2idx.get(pred_op, 0)] += 1
        conf_dr[d2, pred_dr] += 1

        conf = float(pred.confidence)
        sum_conf += conf
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bin_counts[bin_idx] += 1
        bin_conf_sums[bin_idx] += conf
        bin_correct_sums[bin_idx] += float(ok_full)

        saved_image = None
        if save_dir is not None:
            saved_image = save_prediction_image(
                src_path=jpg,
                prediction=pred,
                save_dir=save_dir,
            )

        gt_expression = f"{d1}{op}{d2}" if ans is None else f"{d1}{op}{d2}={ans}"
        pred_expression = pred.expression if pred.result is None else f"{pred.expression}={pred.result}"
        records.append(
            {
                "image": jpg.name,
                "gt_expression": gt_expression,
                "pred_expression": pred_expression,
                "pred_result": pred.result,
                "gt_result": ans,
                "confidence": conf,
                "ok_digit_left": ok_dl,
                "ok_operator": ok_op,
                "ok_digit_right": ok_dr,
                "ok_full": ok_full,
                "saved_image": saved_image,
            }
        )

    return {
        "counts": np.array([n, n_dl, n_op, n_dr, n_full, n_eval, n_answered], dtype=np.int64),
        "sum_conf": np.array([sum_conf], dtype=np.float64),
        "bin_counts": bin_counts,
        "bin_conf_sums": bin_conf_sums,
        "bin_correct_sums": bin_correct_sums,
        "confusion_digit_left": conf_dl,
        "confusion_operator": conf_op,
        "confusion_digit_right": conf_dr,
    }, records


def reduce_numpy_array(accelerator: Accelerator, array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.as_tensor(array, device=accelerator.device, dtype=dtype)
    return accelerator.reduce(tensor, reduction="sum")


def gather_object_records(accelerator: Accelerator, local_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if accelerator.num_processes == 1 or not dist.is_available() or not dist.is_initialized():
        return local_records
    gathered: list[list[dict[str, Any]] | None] = [None for _ in range(accelerator.num_processes)]
    dist.all_gather_object(gathered, local_records)
    merged: list[dict[str, Any]] = []
    for chunk in gathered:
        if chunk:
            merged.extend(chunk)
    return merged


def compute_ece_from_bins(
    *,
    bin_counts: np.ndarray,
    bin_conf_sums: np.ndarray,
    bin_correct_sums: np.ndarray,
) -> float:
    total = int(bin_counts.sum())
    if total == 0:
        return 0.0
    ece = 0.0
    for count, conf_sum, correct_sum in zip(bin_counts, bin_conf_sums, bin_correct_sums):
        if count <= 0:
            continue
        avg_conf = float(conf_sum / count)
        avg_acc = float(correct_sum / count)
        ece += (count / total) * abs(avg_acc - avg_conf)
    return float(ece)


def main() -> None:
    args = resolve_defaults_from_checkpoint(parse_args())
    accelerator = Accelerator()
    backend = build_backend(args, accelerator)
    inferencer = CaptchaInferencer(
        backend=backend,
        config=InferencerConfig(
            image_size_h=args.image_size_h,
            image_size_w=args.image_size_w,
            threshold=args.threshold,
            binarize_mode=args.binarize_mode,
            adaptive_block_size=args.adaptive_block_size,
            adaptive_c=args.adaptive_c,
            batch_size=args.batch_size,
        ),
    )

    all_samples = load_test_samples(args.data_root, args.limit)
    local_samples = shard_samples(all_samples, accelerator.process_index, accelerator.num_processes)
    save_dir = Path(args.save_dir).resolve() if args.save_dir else None

    accelerator.print(
        f"[multi-gpu-bench] world_size={accelerator.num_processes} "
        f"rank={accelerator.process_index} device={accelerator.device} "
        f"n_total={len(all_samples)} n_local={len(local_samples)}"
    )

    local_stats, local_records = evaluate_local_shard(
        inferencer,
        local_samples,
        save_dir=save_dir,
    )

    global_counts = reduce_numpy_array(accelerator, local_stats["counts"], torch.int64).cpu().numpy()
    global_sum_conf = reduce_numpy_array(accelerator, local_stats["sum_conf"], torch.float64).cpu().numpy()
    global_bin_counts = reduce_numpy_array(accelerator, local_stats["bin_counts"], torch.int64).cpu().numpy()
    global_bin_conf_sums = reduce_numpy_array(accelerator, local_stats["bin_conf_sums"], torch.float64).cpu().numpy()
    global_bin_correct_sums = reduce_numpy_array(accelerator, local_stats["bin_correct_sums"], torch.float64).cpu().numpy()
    global_conf_dl = reduce_numpy_array(accelerator, local_stats["confusion_digit_left"], torch.int64).cpu().numpy()
    global_conf_op = reduce_numpy_array(accelerator, local_stats["confusion_operator"], torch.int64).cpu().numpy()
    global_conf_dr = reduce_numpy_array(accelerator, local_stats["confusion_digit_right"], torch.int64).cpu().numpy()

    accelerator.print(
        f"[multi-gpu-bench][rank={accelerator.process_index}] "
        f"n_seen_local={int(local_stats['counts'][0])}"
    )

    if accelerator.is_main_process:
        n = int(global_counts[0])
        n_answered = int(global_counts[6])
        metrics = EvalMetrics(
            n_samples=n,
            acc_digit_left=(int(global_counts[1]) / n) if n else 0.0,
            acc_operator=(int(global_counts[2]) / n) if n else 0.0,
            acc_digit_right=(int(global_counts[3]) / n) if n else 0.0,
            acc_expression=(int(global_counts[4]) / n) if n else 0.0,
            acc_eval_result=(int(global_counts[5]) / n_answered) if n_answered else 0.0,
            mean_confidence=float(global_sum_conf[0] / n) if n else 0.0,
            ece=compute_ece_from_bins(
                bin_counts=global_bin_counts,
                bin_conf_sums=global_bin_conf_sums,
                bin_correct_sums=global_bin_correct_sums,
            ),
            confusion={
                "digit_left": global_conf_dl,
                "operator": global_conf_op,
                "digit_right": global_conf_dr,
            },
        )
        report = metrics_to_dict(metrics)
        report["benchmark"] = {
            "mode": "multi-gpu-ddp-test-inference",
            "world_size": accelerator.num_processes,
            "n_samples_global": n,
            "n_answered_global": n_answered,
            "save_dir": str(save_dir) if save_dir is not None else None,
        }
        report["predictions"] = gather_object_records(accelerator, local_records)

        if args.output:
            Path(args.output).write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            accelerator.print(f"[saved] {args.output}")
        else:
            print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        gather_object_records(accelerator, local_records)

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
