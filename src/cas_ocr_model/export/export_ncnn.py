#!/usr/bin/env python3
"""直接使用 pnnx Python API 导出 ncnn.

用法:
    python -m cas_ocr_model.export.export_ncnn \
        --checkpoint ./runs/exp1/best.pt \
        --output ./runs/exp1/export/ncnn/best.fp16.pt \
        --image-size-h 64 --image-size-w 192

输出:
    best.pt
    best.param
    best.bin
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from cas_ocr_model.model.stats import collect_model_stats, format_model_stats
from cas_ocr_model.trainer.model import build_model_from_checkpoint


class ExportWrapper(nn.Module):
    """把 dict 输出包装成 pnnx 更稳定的 tuple 输出."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.model(x)
        return out["digit_left_logits"], out["operator_logits"], out["digit_right_logits"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="使用 pnnx Python API 导出 ncnn")
    p.add_argument("--checkpoint", required=True, help="训练得到的 best.pt / last.pt")
    p.add_argument("--output", required=True, help="导出的 pnnx 输入 pt 路径, 例如 ./export/best.fp16.pt")
    p.add_argument("--image-size-h", type=int, default=64)
    p.add_argument("--image-size-w", type=int, default=192)
    p.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    p.add_argument("--optlevel", type=int, default=2)
    p.add_argument("--ncnn-param", default=None)
    p.add_argument("--ncnn-bin", default=None)
    p.add_argument("--pnnx-param", default=None)
    p.add_argument("--pnnx-bin", default=None)
    p.add_argument("--pnnx-py", default=None)
    p.add_argument("--pnnx-onnx", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import pnnx
    except ImportError as exc:  # pragma: no cover - 取决于用户环境
        raise SystemExit(
            "未安装 pnnx Python 包. 请先执行: pip install pnnx"
        ) from exc

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ncnn_param = Path(args.ncnn_param).resolve() if args.ncnn_param else output_path.with_suffix(".param")
    ncnn_bin = Path(args.ncnn_bin).resolve() if args.ncnn_bin else output_path.with_suffix(".bin")
    pnnx_param = Path(args.pnnx_param).resolve() if args.pnnx_param else output_path.with_suffix(".pnnx.param")
    pnnx_bin = Path(args.pnnx_bin).resolve() if args.pnnx_bin else output_path.with_suffix(".pnnx.bin")
    pnnx_py = Path(args.pnnx_py).resolve() if args.pnnx_py else output_path.with_name(f"{output_path.stem}_pnnx.py")
    pnnx_onnx = Path(args.pnnx_onnx).resolve() if args.pnnx_onnx else output_path.with_suffix(".pnnx.onnx")

    model = build_model_from_checkpoint(args.checkpoint, device="cpu")
    model.eval()
    print(
        f"[model-stats] "
        f"{format_model_stats(collect_model_stats(model, args.image_size_h, args.image_size_w))}"
    )

    wrapper = ExportWrapper(model)
    wrapper.eval()
    dummy = torch.randn(1, 1, args.image_size_h, args.image_size_w, dtype=torch.float32)

    print(f"[export-ncnn-python] checkpoint = {args.checkpoint}")
    print(f"[export-ncnn-python] output     = {output_path}")
    print(f"[export-ncnn-python] precision  = {args.precision}")
    print(f"[export-ncnn-python] inputshape = {(1, 1, args.image_size_h, args.image_size_w)}")
    print(f"[export-ncnn-python] ncnnparam  = {ncnn_param}")
    print(f"[export-ncnn-python] ncnnbin    = {ncnn_bin}")
    pnnx.export(
        wrapper,
        str(output_path),
        (dummy,),
        fp16=args.precision == "fp16",
        optlevel=args.optlevel,
        pnnxparam=str(pnnx_param),
        pnnxbin=str(pnnx_bin),
        pnnxpy=str(pnnx_py),
        pnnxonnx=str(pnnx_onnx),
        ncnnparam=str(ncnn_param),
        ncnnbin=str(ncnn_bin),
    )
    print(f"[export-ncnn-python] done -> {output_path}")
    print(f"[export-ncnn-python] expect   -> {ncnn_param}")
    print(f"[export-ncnn-python] expect   -> {ncnn_bin}")


if __name__ == "__main__":
    main()
