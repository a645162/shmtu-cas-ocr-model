"""一键导出三个模型到 ONNX。"""

from cas_ocr_model.v1.configs.paths import pth_save_dir_path
from cas_ocr_model.v1.tasks.digit.onnx_export import export_digit_to_onnx
from cas_ocr_model.v1.tasks.equal_symbol.onnx_export import export_equal_symbol_to_onnx
from cas_ocr_model.v1.tasks.operator.onnx_export import export_operator_to_onnx


def export_all_to_onnx(save_dir: str = pth_save_dir_path) -> None:
    export_equal_symbol_to_onnx(save_dir)
    export_operator_to_onnx(save_dir)
    export_digit_to_onnx(save_dir)


if __name__ == "__main__":
    export_all_to_onnx()
