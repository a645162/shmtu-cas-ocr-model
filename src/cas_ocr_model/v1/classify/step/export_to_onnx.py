import os.path

import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

from cas_ocr_model.v1.classify.model.model_type import ModelType
from cas_ocr_model.v1.classify.model.my_model import init_model
from cas_ocr_model.v1.config import config


def export_to_onnx(
        model_type: ModelType,
        output_features_count: int,
        pth_path: str,
        onnx_file_path: str = ""
):
    onnx_file_path = onnx_file_path.strip()
    if len(onnx_file_path) == 0:
        base_name = os.path.basename(pth_path)
        dir_path = os.path.dirname(pth_path)
        no_ext = os.path.splitext(base_name)[0]
        onnx_file_name = f"{no_ext}.onnx"
        onnx_file_path = os.path.join(dir_path, onnx_file_name)

    print(f"Input:", pth_path)

    model = init_model(model_type, output_features_count, False)
    model.load_state_dict(torch.load(pth_path))

    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    input_names = [config.input_name]
    output_names = [config.output_name]

    torch.onnx.export(
        model,
        dummy_input,
        f=onnx_file_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=11
    )

    print(f"Output:", onnx_file_path)
