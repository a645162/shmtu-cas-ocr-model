import os

import torch

from cas_ocr_model.v1.config import config

import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

from cas_ocr_model.v1.classify.step.train import train_model
from cas_ocr_model.v1.classify.utils.devices_selector import get_recommended_device


def train_operator(device: torch.device):
    """训练运算符分类模型"""

    print("Training operator...")

    dataset_path = os.path.join(config.dataset_dir_path, "Operator")

    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 加载数据集
    dataset_train = datasets.ImageFolder(root=os.path.join(dataset_path, "train"), transform=transform)
    dataset_val = datasets.ImageFolder(root=os.path.join(dataset_path, "val"), transform=transform)

    # 创建数据加载器
    batch_size = 64
    data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    data_loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    train_model(
        device,
        data_loader_train,
        data_loader_val,
        config.model_operator_type,
        6,
        True, "",
        "operator",
        3,
        config.pth_save_dir_path
    )


if __name__ == '__main__':
    device = get_recommended_device()
    train_operator(device)
