import os

import torch

from cas_ocr_model.v1.config import config

import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

from cas_ocr_model.v1.classify.model.model_type import get_pth_name
from cas_ocr_model.v1.classify.step.train import train_model
from cas_ocr_model.v1.classify.utils.devices_selector import get_recommended_device

dataset_path = os.path.join(config.dataset_dir_path, "Digit")

max_degree = config.data_transform_rotate_degree


def train_digit(device: torch.device):

    print("Training digit on SHMTU CAS dataset...")

    transform_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((240, 240)),
        transforms.RandomCrop(224),
        transforms.RandomRotation(degrees=(-max_degree, max_degree)),
        transforms.ToTensor(),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 加载数据集
    dataset_train = datasets.ImageFolder(root=os.path.join(dataset_path, "train"), transform=transform_train)
    dataset_val = datasets.ImageFolder(root=os.path.join(dataset_path, "val"), transform=transform_val)

    # 创建数据加载器
    batch_size = 64
    data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    data_loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    if config.pretrain_on_mnist:
        mnist_pth_name: str = get_pth_name(
            config.model_digit_type,
            "digit_mnist",
            "latest"
        )
        mnist_pth_path: str = str(os.path.join(config.pth_save_dir_path, mnist_pth_name))
    else:
        mnist_pth_path = ""

    train_model(
        device,
        data_loader_train,
        data_loader_val,
        config.model_digit_type,
        10,
        True, mnist_pth_path,
        "digit",
        config.epoch_digit,
        config.pth_save_dir_path
    )


if __name__ == "__main__":
    device = get_recommended_device()
    train_digit(device)
