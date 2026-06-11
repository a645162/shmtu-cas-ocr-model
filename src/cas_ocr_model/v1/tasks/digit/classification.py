"""数字分类：把已切好的数字子图按模型预测结果归档。"""

from __future__ import annotations

import os

from torch.utils.data import DataLoader
from torchvision import transforms

from ...configs.defaults import model_digit_type, pth_save_dir_path
from ...data_modules.dataset import CustomDataset
from ...data_modules.device import get_recommended_device
from ...training.classification import classify_by_model

_TRANSFORM = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])


def classify_digits_in_split(
    split_dir: str = "../workdir/Spilt/MainBody_symbol/0",
    output_dir: str = "../workdir/Test/Digit_Predictions/Digit_Predictions_symbol_0_test",
) -> None:
    """对一组切好的数字子图做分类。"""
    dataset = CustomDataset(root=split_dir, transform=_TRANSFORM)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    device = get_recommended_device()

    classify_by_model(
        device=device,
        class_count=10,
        model_type=model_digit_type,
        pth_path=os.path.join(pth_save_dir_path, "resnet34_digit_latest.pth"),
        data_loader=loader,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    classify_digits_in_split()
