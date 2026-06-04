"""指标计算: 在带 ground truth 的数据集上评估推理.

输入: dataset_dir (jpg + json 配对, json 含 expression)
输出: dict 报告
    * per-head top-1 准确率 (digit_left / operator / digit_right)
    * 全表达式正确率 (3 头全对)
    * 算式结果正确率 (求出的算式结果 == gt answer)
    * 类别级混淆矩阵 (digit / operator 各一份)
    * 置信度校准: ECE + 平均置信度
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .inference import CaptchaInferencer, InferenceResult

_EXPR_RE = re.compile(r"^(\d)([+\-*/])(\d)=$")


# ----------------------------------------------------------------------------
# 结果汇总
# ----------------------------------------------------------------------------


@dataclass
class EvalMetrics:
    n_samples: int = 0

    acc_digit_left: float = 0.0
    acc_operator: float = 0.0
    acc_digit_right: float = 0.0

    acc_expression: float = 0.0        # 3 头 argmax 全等于 gt
    acc_eval_result: float = 0.0       # 求出的算式结果 == gt answer

    mean_confidence: float = 0.0
    ece: float = 0.0                   # Expected Calibration Error (15 bins)

    confusion: dict[str, np.ndarray] = field(default_factory=dict)
    per_sample: list[dict] = field(default_factory=list)


# ----------------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------------


def _parse_gt_expression(expr: str) -> Optional[tuple[int, str, int, int]]:
    """从 expression "12+34=46" 解出 (d1, op, d2, answer). 失败返回 None."""
    m = _EXPR_RE.match(expr)
    if not m:
        return None
    d1, op, d2 = int(m.group(1)), m.group(2), int(m.group(3))
    rhs = expr[m.end():]
    try:
        ans = int(rhs)
    except ValueError:
        return None
    return d1, op, d2, ans


def _safe_eval(d1: int, op: str, d2: int) -> Optional[int]:
    try:
        if op == "+": return d1 + d2
        if op == "-": return d1 - d2
        if op == "*": return d1 * d2
        if op == "/": return d1 // d2 if d2 != 0 else None
    except Exception:
        return None
    return None


def _compute_ece(confidences: np.ndarray, corrects: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i > 0:
            in_bin = (confidences > lo) & (confidences <= hi)
        else:
            in_bin = (confidences >= lo) & (confidences <= hi)
        if in_bin.any():
            avg_conf = confidences[in_bin].mean()
            avg_acc = corrects[in_bin].mean()
            ece += (in_bin.sum() / len(confidences)) * abs(avg_acc - avg_conf)
    return float(ece)


# ----------------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------------


def evaluate(
    inferencer: CaptchaInferencer,
    dataset_dir: str | Path,
    pattern: str = "*.jpg",
    limit: Optional[int] = None,
    digit_labels: Optional[list[str]] = None,
    operator_labels: Optional[list[str]] = None,
) -> EvalMetrics:
    """在带 ground-truth 的数据集上评估.

    Args:
        inferencer: 已构造的 CaptchaInferencer
        dataset_dir: 包含 jpg + json 配对的目录
        pattern: glob 模式
        limit: 最多取多少张
    """
    if digit_labels is None:
        from cas_ocr_model.trainer.config import DIGIT_LABELS as digit_labels  # type: ignore
    if operator_labels is None:
        from cas_ocr_model.trainer.config import OPERATOR_LABELS as operator_labels  # type: ignore

    digit2idx = {s: i for i, s in enumerate(digit_labels)}
    op2idx = {s: i for i, s in enumerate(operator_labels)}

    paths = sorted(Path(dataset_dir).glob(pattern))
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise RuntimeError(f"no images found in {dataset_dir} matching {pattern}")

    samples: list[tuple[Path, tuple[int, str, int, int]]] = []
    for jpg in paths:
        json_path = jpg.with_suffix(".json")
        if not json_path.is_file():
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
        raise RuntimeError(f"no valid (jpg+json) samples in {dataset_dir}")

    results: list[InferenceResult] = inferencer.predict_batch([p for p, _ in samples])
    assert len(results) == len(samples)

    n_dl = n_op = n_dr = n_full = n_eval = 0
    n = len(samples)
    confs: list[float] = []
    corrects: list[int] = []

    conf_dl = np.zeros((10, 10), dtype=np.int64)
    conf_op = np.zeros((4, 4), dtype=np.int64)
    conf_dr = np.zeros((10, 10), dtype=np.int64)

    per_sample: list[dict] = []

    for (jpg, (d1, op, d2, ans)), r in zip(samples, results):
        try:
            pred_dl = int(r.digit_left)
            pred_op = r.operator
            pred_dr = int(r.digit_right)
        except ValueError:
            continue

        gt_dl_idx = d1
        gt_op_idx = op2idx.get(op, -1)
        gt_dr_idx = d2

        ok_dl = pred_dl == gt_dl_idx
        ok_op = pred_op == op
        ok_dr = pred_dr == gt_dr_idx
        ok_full = ok_dl and ok_op and ok_dr

        n_dl += int(ok_dl)
        n_op += int(ok_op)
        n_dr += int(ok_dr)
        n_full += int(ok_full)

        eval_v = _safe_eval(pred_dl, pred_op, pred_dr)
        n_eval += int(eval_v is not None and eval_v == ans)

        if ok_dl:
            conf_dl[gt_dl_idx, gt_dl_idx] += 1
        else:
            conf_dl[gt_dl_idx, pred_dl] += 1
        if gt_op_idx >= 0:
            if ok_op:
                conf_op[gt_op_idx, gt_op_idx] += 1
            else:
                pred_op_idx = op2idx.get(pred_op, -1)
                if pred_op_idx >= 0:
                    conf_op[gt_op_idx, pred_op_idx] += 1
        if ok_dr:
            conf_dr[gt_dr_idx, gt_dr_idx] += 1
        else:
            conf_dr[gt_dr_idx, pred_dr] += 1

        confs.append(r.confidence)
        corrects.append(int(ok_full))

        per_sample.append(
            {
                "image": jpg.name,
                "gt_expression": f"{d1}{op}{d2}={ans}",
                "pred_expression": r.expression,
                "ok_digit_left": ok_dl,
                "ok_operator": ok_op,
                "ok_digit_right": ok_dr,
                "ok_full": ok_full,
                "pred_result": eval_v,
                "gt_result": ans,
                "confidence": r.confidence,
            }
        )

    confs_arr = np.array(confs, dtype=np.float64) if confs else np.zeros(0)
    corrects_arr = np.array(corrects, dtype=np.float64) if corrects else np.zeros(0)

    return EvalMetrics(
        n_samples=n,
        acc_digit_left=n_dl / n,
        acc_operator=n_op / n,
        acc_digit_right=n_dr / n,
        acc_expression=n_full / n,
        acc_eval_result=n_eval / n,
        mean_confidence=float(confs_arr.mean()) if confs_arr.size else 0.0,
        ece=_compute_ece(confs_arr, corrects_arr) if confs_arr.size else 0.0,
        confusion={
            "digit_left": conf_dl,
            "operator": conf_op,
            "digit_right": conf_dr,
        },
        per_sample=per_sample,
    )


def metrics_to_dict(m: EvalMetrics) -> dict:
    """序列化为 json-friendly dict."""
    return {
        "n_samples": m.n_samples,
        "acc_digit_left": m.acc_digit_left,
        "acc_operator": m.acc_operator,
        "acc_digit_right": m.acc_digit_right,
        "acc_expression": m.acc_expression,
        "acc_eval_result": m.acc_eval_result,
        "mean_confidence": m.mean_confidence,
        "ece": m.ece,
        "confusion": {k: v.tolist() for k, v in m.confusion.items()},
    }
