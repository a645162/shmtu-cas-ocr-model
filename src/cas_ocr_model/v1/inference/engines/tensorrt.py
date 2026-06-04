"""TensorRT 引擎构建、转换、推理（FP32/FP16/INT8）。"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import cv2
import numpy as np


def build_trt_engine_from_onnx(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    max_workspace_size: int = 2 << 30,
) -> None:
    """把 ONNX 编译成 TensorRT 引擎并序列化到 engine_path。"""
    import tensorrt as trt

    trt_logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, trt_logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for err in range(parser.num_errors):
                print(parser.get_error(err))
            raise RuntimeError("Failed to parse ONNX")

    if os.path.isfile(engine_path):
        os.remove(engine_path)

    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    engine = builder.build_engine(network, config)
    with open(engine_path, "wb") as f:
        f.write(engine.serialize())
    print(f"[trt] engine saved at: {engine_path}")


def convert_onnx_to_trt(
    onnx_path: str,
    engine_path: str,
    mode: str = "fp16",
    batch_size: int = 1,
    calib=None,
) -> None:
    """通过 argparse 风格参数把 ONNX 转成 TensorRT engine。"""
    import tensorrt as trt

    assert mode.lower() in ("fp32", "fp16", "int8"), "mode must be fp32/fp16/int8"
    trt_logger = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(trt_logger) as builder, builder.create_network() as network, \
            trt.OnnxParser(network, trt_logger) as parser:
        builder.max_batch_size = batch_size
        builder.max_workspace_size = 1 << 30
        if mode.lower() == "int8":
            assert builder.platform_has_fast_int8, "platform lacks INT8"
            builder.int8_mode = True
            builder.int8_calibrator = calib
        elif mode.lower() == "fp16":
            assert builder.platform_has_fast_fp16, "platform lacks FP16"
            builder.fp16_mode = True

        with open(onnx_path, "rb") as f:
            parser.parse(f.read())
        engine = builder.build_cuda_engine(network)
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())
        print(f"[trt] engine -> {engine_path}")


def run_trt_inference(
    engine_path: str,
    image_path: str,
    image_size: int = 224,
) -> int:
    """加载 engine，对单张图做推理，返回 argmax。"""
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    import tensorrt as trt
    from PIL import Image

    trt_logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(trt_logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    h_in = cuda.pagelocked_empty(engine.get_binding_shape(0), dtype=trt.nptype(trt.float32))
    h_out = cuda.pagelocked_empty(engine.get_binding_shape(1), dtype=trt.nptype(trt.float32))

    image = np.array(Image.open(image_path).resize((image_size, image_size))) / 255.0
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = np.ascontiguousarray(image)
    np.copyto(h_in, image)

    cuda.memcpy_htod(cuda.mem_alloc(h_in.nbytes), h_in)
    context.execute_v2([int(h_in.ctypes.data), int(h_out.ctypes.data)])
    cuda.Context.synchronize()

    out = np.empty_like(h_out)
    cuda.memcpy_dtoh(out, cuda.mem_alloc(h_out.nbytes))
    return int(out.argmax())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--mode", default="fp16", choices=["fp32", "fp16", "int8"])
    args = parser.parse_args()
    convert_onnx_to_trt(args.onnx, args.engine, args.mode)
