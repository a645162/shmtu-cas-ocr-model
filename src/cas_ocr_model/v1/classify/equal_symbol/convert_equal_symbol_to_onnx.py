import os

from cas_ocr_model.v1.classify.model.model_type import get_pth_name
from cas_ocr_model.v1.classify.step.export_to_onnx import export_to_onnx
from cas_ocr_model.v1.config import config


def export_equal_symbol_to_onnx(save_dir_path: str = "../../../workdir/Models"):
    pth_path = os.path.join(
        save_dir_path,
        get_pth_name(
            config.model_equal_symbol_type,
            "equal_symbol",
            str("latest")
        )
    )

    export_to_onnx(
        config.model_equal_symbol_type,
        output_features_count=2,
        pth_path=pth_path
    )


if __name__ == "__main__":
    export_equal_symbol_to_onnx(
        config.pth_save_dir_path
    )
