import os

from cas_ocr_model.v1.classify.model.model_type import get_pth_name
from cas_ocr_model.v1.classify.step.export_to_onnx import export_to_onnx
from cas_ocr_model.v1.config import config


def export_digit_to_onnx(save_dir_path: str = "../../../workdir/Models"):
    pth_path = os.path.join(
        save_dir_path,
        get_pth_name(
            config.model_digit_type,
            "digit",
            str("latest")
        )
    )

    export_to_onnx(
        config.model_digit_type,
        output_features_count=10,
        pth_path=pth_path
    )


if __name__ == "__main__":
    export_digit_to_onnx(
        config.pth_save_dir_path
    )
