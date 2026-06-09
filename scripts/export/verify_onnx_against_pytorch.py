#!/usr/bin/env python3
"""校验 ONNX 推理输出是否与 PyTorch 直接推理一致."""
from __future__ import annotations

import argparse

import numpy as np

from verify_common import (
    HEAD_NAMES,
    add_common_args,
    compare_outputs,
    describe_predictions,
    load_input_array,
    run_pytorch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="校验 ONNX 与 PyTorch 推理输出")
    parser.add_argument("--onnx", required=True, help="待校验的 ONNX 文件路径")
    add_common_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("未安装 onnxruntime. 请先执行: pip install onnxruntime") from exc

    input_array = load_input_array(args)
    reference = run_pytorch(args.checkpoint, input_array, args.device)

    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_dtype = np.float16 if "float16" in input_meta.type else np.float32
    ort_input = input_array.astype(input_dtype, copy=False)

    output_metas = session.get_outputs()
    output_names = [meta.name for meta in output_metas]
    raw_outputs = session.run(output_names, {input_name: ort_input})
    candidate = {
        head_name: np.asarray(value, dtype=np.float32)
        for head_name, value in zip(HEAD_NAMES, raw_outputs)
    }

    print(f"[verify-onnx] onnx        = {args.onnx}")
    print(f"[verify-onnx] checkpoint  = {args.checkpoint}")
    print(f"[verify-onnx] input_name  = {input_name}")
    print(f"[verify-onnx] output_name = {output_names}")
    print(f"[verify-onnx] input_shape = {tuple(input_array.shape)}")
    print(f"[verify-onnx] pytorch_pred= {describe_predictions(reference)}")
    print(f"[verify-onnx] onnx_pred   = {describe_predictions(candidate)}")

    passed, lines = compare_outputs(reference, candidate, atol=args.atol, rtol=args.rtol)
    for line in lines:
        print(line)

    if not passed:
        raise SystemExit("[verify-onnx] FAILED")
    print("[verify-onnx] PASSED")


if __name__ == "__main__":
    main()
