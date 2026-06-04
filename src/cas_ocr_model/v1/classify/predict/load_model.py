import os

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms

from cas_ocr_model.v1.classify.model.model_type import get_pth_name
from cas_ocr_model.v1.classify.model.my_model import init_model
from cas_ocr_model.v1.classify.utils.devices_selector import get_recommended_device
from cas_ocr_model.v1.config import config
from cas_ocr_model.v1.utils.pic.cv2plt import show_opencv_img_by_plt


def load_model(device: torch.device):
    # 定义模型
    model_equal_symbol = (
        init_model(
            model_type=config.model_equal_symbol_type,
            output_features_count=2,
            pretrained=False
        )
    )
    model_operator = (
        init_model(
            model_type=config.model_operator_type,
            output_features_count=6,
            pretrained=False
        )
    )
    model_digit = (
        init_model(
            model_type=config.model_digit_type,
            output_features_count=10,
            pretrained=False
        )
    )

    # 将模型加载到device中
    model_equal_symbol = model_equal_symbol.to(device)
    model_operator = model_operator.to(device)
    model_digit = model_digit.to(device)

    # 加载已训练的模型参数pth权重文件
    model_equal_symbol.load_state_dict(
        torch.load(
            str(
                os.path.join(
                    config.pth_save_dir_path,
                    get_pth_name(
                        config.model_equal_symbol_type,
                        "equal_symbol",
                        "latest"
                    )
                )
            ),
            map_location=device
        )
    )
    model_operator.load_state_dict(
        torch.load(
            str(
                os.path.join(
                    config.pth_save_dir_path,
                    get_pth_name(
                        config.model_operator_type,
                        "operator",
                        "latest"
                    )
                )
            ),
            map_location=device
        )
    )
    model_digit.load_state_dict(
        torch.load(
            str(
                os.path.join(
                    config.pth_save_dir_path,
                    get_pth_name(
                        config.model_digit_type,
                        "digit",
                        "latest"
                    )
                )
            ),
            map_location=device
        )
    )

    model_equal_symbol.eval()
    model_operator.eval()
    model_digit.eval()

    return model_equal_symbol, model_operator, model_digit


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def predict_cv_image(
        model: torch.nn.Module,
        device: torch.device,
        image_cv_rgb: np.ndarray
) -> int:
    # show_opencv_img_by_plt(image_cv_rgb)
    img = Image.fromarray(image_cv_rgb)

    img_tensor = transform(img).unsqueeze(0)

    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)

    return predicted.item()


if __name__ == "__main__":
    device = get_recommended_device()

    model_equal_symbol, model_operator, model_digit = load_model(device)
