#!/usr/bin/env python3
"""校验 ncnn 推理输出是否与 PyTorch 直接推理一致."""
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
    parser = argparse.ArgumentParser(description="校验 ncnn 与 PyTorch 推理输出")
    parser.add_argument("--param", required=True, help="待校验的 ncnn .param 路径")
    parser.add_argument("--bin", dest="bin_path", required=True, help="待校验的 ncnn .bin 路径")
    parser.add_argument("--input-name", default="in0")
    parser.add_argument("--output-names", default="out0,out1,out2")
    add_common_args(parser)
    parser.set_defaults(atol=5e-2, rtol=5e-2)
    return parser.parse_args()


def run_ncnn(args, input_array: np.ndarray) -> dict[str, np.ndarray]:
    try:
        import ncnn
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("未安装 ncnn Python 包") from exc

    output_names = [name.strip() for name in args.output_names.split(",") if name.strip()]
    if len(output_names) != len(HEAD_NAMES):
        raise SystemExit(f"--output-names 需要提供 {len(HEAD_NAMES)} 个名字, 当前是 {output_names}")

    net = ncnn.Net()
    if net.load_param(args.param) != 0:
        raise SystemExit(f"加载 ncnn param 失败: {args.param}")
    if net.load_model(args.bin_path) != 0:
        raise SystemExit(f"加载 ncnn bin 失败: {args.bin_path}")

    outputs = {name: [] for name in HEAD_NAMES}
    for sample in input_array:
        extractor = net.create_extractor()
        mat = ncnn.Mat(np.ascontiguousarray(sample, dtype=np.float32))
        if extractor.input(args.input_name, mat) != 0:
            raise SystemExit(f"ncnn extractor.input 失败, input_name={args.input_name}")
        for head_name, blob_name in zip(HEAD_NAMES, output_names):
            ret, out = extractor.extract(blob_name)
            if ret != 0:
                raise SystemExit(f"ncnn extractor.extract 失败, blob_name={blob_name}, ret={ret}")
            outputs[head_name].append(np.asarray(out.numpy(), dtype=np.float32))

    return {name: np.stack(values, axis=0) for name, values in outputs.items()}


def main() -> None:
    args = parse_args()
    input_array = load_input_array(args)
    reference = run_pytorch(args.checkpoint, input_array, args.device)
    candidate = run_ncnn(args, input_array)

    output_names = [name.strip() for name in args.output_names.split(",") if name.strip()]
    print(f"[verify-ncnn] param       = {args.param}")
    print(f"[verify-ncnn] bin         = {args.bin_path}")
    print(f"[verify-ncnn] checkpoint  = {args.checkpoint}")
    print(f"[verify-ncnn] input_name  = {args.input_name}")
    print(f"[verify-ncnn] output_name = {output_names}")
    print(f"[verify-ncnn] input_shape = {tuple(input_array.shape)}")
    print(f"[verify-ncnn] pytorch_pred= {describe_predictions(reference)}")
    print(f"[verify-ncnn] ncnn_pred   = {describe_predictions(candidate)}")

    passed, lines = compare_outputs(reference, candidate, atol=args.atol, rtol=args.rtol)
    for line in lines:
        print(line)

    if not passed:
        raise SystemExit("[verify-ncnn] FAILED")
    print("[verify-ncnn] PASSED")


if __name__ == "__main__":
    main()
