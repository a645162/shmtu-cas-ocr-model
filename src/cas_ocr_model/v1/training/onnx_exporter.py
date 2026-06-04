"""把 PyTorch 模型导出为 ONNX。"""

from __future__ import annotations

import os

import torch

from ..configs.defaults import input_name, output_name
from ..configs.model import ModelType
from ..models.resnet import init_model


def export_to_onnx(
    model_type: ModelType,
    output_features_count: int,
    pth_path: str,
    onnx_file_path: str = "",
) -> str:
    """
    加载 pth_path 权重，导出为 ONNX。

    Args:
        model_type: 模型类型。
        output_features_count: 输出维度。
        pth_path: 权重文件路径。
        onnx_file_path: 目标 ONNX 路径；为空则与 pth_path 同名 .onnx。

    Returns:
        实际写入的 ONNX 路径。
    """
    if not onnx_file_path:
        base = os.path.basename(pth_path)
        dir_path = os.path.dirname(pth_path)
        onnx_file_path = os.path.join(dir_path, os.path.splitext(base)[0] + ".onnx")

    print(f"[onnx] input:  {pth_path}")
    model = init_model(model_type, output_features_count, pretrained=False)
    model.load_state_dict(torch.load(pth_path))
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        f=onnx_file_path,
        input_names=[input_name],
        output_names=[output_name],
        opset_version=11,
    )
    print(f"[onnx] output: {onnx_file_path}")
    return onnx_file_path
