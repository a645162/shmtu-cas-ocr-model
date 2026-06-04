"""等号分类：训练。"""

from __future__ import annotations

import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

from ...configs.defaults import (
    dataset_dir_path,
    epoch_equal_symbol,
    model_equal_symbol_type,
    pth_save_dir_path,
)
from ...data_modules.device import get_recommended_device
from ...training.trainer import train_model


def train_equal_symbol(device: torch.device | None = None) -> None:
    """训练"等号 vs 中文等于"二分类模型。"""
    if device is None:
        device = get_recommended_device()
    print("[equal_symbol] training...")

    dataset_dir = os.path.join(dataset_dir_path, "EqualSymbol")
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    train_set = datasets.ImageFolder(
        root=os.path.join(dataset_dir, "train"), transform=transform
    )
    val_set = datasets.ImageFolder(
        root=os.path.join(dataset_dir, "val"), transform=transform
    )
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

    train_model(
        device=device,
        data_loader_train=train_loader,
        data_loader_val=val_loader,
        model_type=model_equal_symbol_type,
        output_features_count=2,
        pretrain=True,
        pth_path="",
        model_label="equal_symbol",
        num_epochs=epoch_equal_symbol,
        pth_save_dir_path=pth_save_dir_path,
    )


if __name__ == "__main__":
    train_equal_symbol()
