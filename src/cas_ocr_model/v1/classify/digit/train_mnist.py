import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cas_ocr_model.v1.classify.step.train import train_model
from cas_ocr_model.v1.classify.utils.devices_selector import get_recommended_device
from cas_ocr_model.v1.config import config

dataset_path = os.path.join(config.dataset_dir_path, "MNIST")

max_degree = config.data_transform_rotate_degree


def train_mnist(device: torch.device):

    print("Training digit on MNIST dataset...")

    transform_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # 将MNIST图像转换为RGB格式
        transforms.Resize((240, 240)),
        transforms.RandomCrop(224),
        transforms.RandomRotation(degrees=(-max_degree, max_degree)),
        transforms.ToTensor(),
    ])

    transform_val = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # 将MNIST图像转换为RGB格式
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    mnist_train = datasets.MNIST(
        root=dataset_path,
        train=True, download=True,
        transform=transform_train
    )
    mnist_val = datasets.MNIST(
        root=dataset_path,
        train=False, download=True,
        transform=transform_val
    )

    batch_size = config.batch_size
    data_loader_train = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    data_loader_val = DataLoader(mnist_val, batch_size=batch_size, shuffle=False)

    train_model(
        device,
        data_loader_train,
        data_loader_val,
        config.model_digit_type,
        10,
        True, "",
        "digit_mnist",
        config.epoch_mnist,
        config.pth_save_dir_path
    )


if __name__ == "__main__":
    device = get_recommended_device()
    train_mnist(device)
