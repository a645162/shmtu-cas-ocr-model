"""运算符分类：单张图像推理。"""

from __future__ import annotations

import os

import cv2
import torch
from PIL import Image
from torchvision import transforms

from ...configs.defaults import model_operator_type, pth_save_dir_path
from ...configs.model import get_pth_name
from ...data_modules.device import get_recommended_device
from ...models.resnet import init_model

_DEVICE = None
_MODEL: torch.nn.Module | None = None
_TRANSFORM = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])


def _ensure_loaded() -> tuple[torch.device, torch.nn.Module]:
    global _DEVICE, _MODEL
    if _MODEL is None:
        _DEVICE = get_recommended_device()
        _MODEL = init_model(model_operator_type, 6, pretrained=False).to(_DEVICE)
        pth = os.path.join(
            pth_save_dir_path,
            get_pth_name(model_operator_type, "operator", "latest"),
        )
        _MODEL.load_state_dict(torch.load(pth, map_location=_DEVICE))
        _MODEL.eval()
    return _DEVICE, _MODEL


def predict_operator(image_path: str) -> int:
    device, model = _ensure_loaded()
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    tensor = _TRANSFORM(Image.fromarray(img)).unsqueeze(0).to(device)
    with torch.no_grad():
        _, pred = torch.max(model(tensor), 1)
    print(f"Prediction for {image_path}: Class {pred.item()}")
    return pred.item()


if __name__ == "__main__":
    for img in ["add", "add_chs", "minus", "minus_chs", "multiply", "multiply_chs"]:
        predict_operator(f"test/{img}.png")
