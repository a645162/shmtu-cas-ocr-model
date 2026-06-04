"""数字分类：训练（含 MNIST 预训练）。"""

from __future__ import annotations

import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

from ...configs.defaults import (
    batch_size,
    data_transform_rotate_degree,
    dataset_dir_path,
    epoch_digit,
    model_digit_type,
    pretrain_on_mnist,
    pth_save_dir_path,
)
from ...configs.model import get_pth_name
from ...data_modules.device import get_recommended_device
from ...training.trainer import train_model

MAX_ROTATE_DEGREE = data_transform_rotate_degree
DATASET_DIR = os.path.join(dataset_dir_path, "Digit")
MNIST_DIR = os.path.join(dataset_dir_path, "MNIST")


def _build_digit_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_t = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((240, 240)),
            transforms.RandomCrop(224),
            transforms.RandomRotation(degrees=(-MAX_ROTATE_DEGREE, MAX_ROTATE_DEGREE)),
            transforms.ToTensor(),
        ]
    )
    val_t = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )
    return train_t, val_t


def train_digit(device: torch.device | None = None) -> None:
    if device is None:
        device = get_recommended_device()
    print("[digit] training on SHMTU CAS dataset...")

    train_t, val_t = _build_digit_transforms()
    train_set = datasets.ImageFolder(
        os.path.join(DATASET_DIR, "train"), transform=train_t
    )
    val_set = datasets.ImageFolder(os.path.join(DATASET_DIR, "val"), transform=val_t)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

    mnist_pretrained_path = ""
    if pretrain_on_mnist:
        mnist_pretrained_path = os.path.join(
            pth_save_dir_path,
            get_pth_name(model_digit_type, "digit_mnist", "latest"),
        )

    train_model(
        device=device,
        data_loader_train=train_loader,
        data_loader_val=val_loader,
        model_type=model_digit_type,
        output_features_count=10,
        pretrain=True,
        pth_path=mnist_pretrained_path,
        model_label="digit",
        num_epochs=epoch_digit,
        pth_save_dir_path=pth_save_dir_path,
    )


def train_mnist(device: torch.device | None = None) -> None:
    if device is None:
        device = get_recommended_device()
    print("[digit] pretraining on MNIST...")

    train_t = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((240, 240)),
            transforms.RandomCrop(224),
            transforms.RandomRotation(degrees=(-MAX_ROTATE_DEGREE, MAX_ROTATE_DEGREE)),
            transforms.ToTensor(),
        ]
    )
    val_t = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    train_set = datasets.MNIST(
        root=MNIST_DIR, train=True, download=True, transform=train_t
    )
    val_set = datasets.MNIST(
        root=MNIST_DIR, train=False, download=True, transform=val_t
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    train_model(
        device=device,
        data_loader_train=train_loader,
        data_loader_val=val_loader,
        model_type=model_digit_type,
        output_features_count=10,
        pretrain=True,
        pth_path="",
        model_label="digit_mnist",
        num_epochs=3,
        pth_save_dir_path=pth_save_dir_path,
    )


def train_digit_all(device: torch.device | None = None) -> None:
    if device is None:
        device = get_recommended_device()
    print("[digit] full pipeline: MNIST pretrain -> SHMTU fine-tune")
    if pretrain_on_mnist:
        train_mnist(device)
    else:
        print("[digit] skip MNIST pretrain")
    train_digit(device)


if __name__ == "__main__":
    train_digit_all()
