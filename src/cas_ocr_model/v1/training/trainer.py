"""通用训练循环：支持任意 (model_type, output_features_count) 的 ResNet 微调。"""

from __future__ import annotations

import os
from time import sleep as time_sleep

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..configs.model import ModelType, get_pth_name
from ..models.resnet import init_model


def train_model(
    device: torch.device,
    data_loader_train: DataLoader,
    data_loader_val: DataLoader | None = None,
    model_type: ModelType = ModelType.ResNet_18,
    output_features_count: int = 6,
    pretrain: bool = True,
    pth_path: str = "",
    model_label: str = "",
    num_epochs: int = 3,
    pth_save_dir_path: str = "../../workdir/Models",
) -> None:
    """
    通用 ResNet 训练循环。

    每个 epoch 保存一份 {model_type}_{label}_{epoch}.pth，
    训练结束后额外保存 {model_type}_{label}_latest.pth。
    """
    if data_loader_train is None:
        return

    model_label = model_label.strip().lower()
    pth_save_dir_path = pth_save_dir_path.strip()
    pth_path = pth_path.strip()

    if pth_path and os.path.exists(pth_path):
        pretrain = False
    else:
        pth_path = ""

    model = init_model(model_type, output_features_count, pretrain).to(device)

    if pth_path:
        print(f"[trainer] loading checkpoint: {pth_path}")
        model.load_state_dict(torch.load(pth_path, map_location=device))

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        time_sleep(1)
        total_loss = 0.0
        model.train()
        for inputs, labels in tqdm(
            data_loader_train, desc=f"Epoch {epoch + 1}/{num_epochs}"
        ):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(data_loader_train)

        if pth_save_dir_path:
            torch.save(
                model.state_dict(),
                os.path.join(
                    pth_save_dir_path,
                    get_pth_name(model_type, model_label, str(epoch + 1)),
                ),
            )

        if data_loader_val is None:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")
            continue

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(data_loader_val, desc="Validating"):
                inputs, labels = inputs.to(device), labels.to(device)
                _, predicted = torch.max(model(inputs), 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Loss: {average_loss:.4f}, Acc: {accuracy * 100:.4f}%"
        )

    if pth_save_dir_path:
        latest_path = os.path.join(
            pth_save_dir_path, get_pth_name(model_type, model_label, "latest")
        )
        torch.save(model.state_dict(), latest_path)
        print(f"[trainer] latest saved at: {latest_path}")
