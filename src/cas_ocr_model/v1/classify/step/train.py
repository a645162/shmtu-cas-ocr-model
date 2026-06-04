import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import sleep as time_sleep

from cas_ocr_model.v1.classify.model.model_type import ModelType, get_pth_name
from cas_ocr_model.v1.classify.model.my_model import init_model


def train_model(
        device: torch.device,
        data_loader_train: DataLoader = None,
        data_loader_val: DataLoader = None,
        model_type: ModelType = ModelType.ResNet_18,
        output_features_count: int = 6,
        pretrain: bool = True,
        pth_path: str = "",
        model_label: str = "",
        num_epochs: int = 3,
        pth_save_dir_path: str = "../../workdir/Models"
):
    if data_loader_train is None:
        return

    model_label = model_label.strip().lower()
    pth_save_dir_path = pth_save_dir_path.strip()

    pth_path = pth_path.strip()
    if len(pth_path) > 0 and os.path.exists(pth_path):
        pretrain = False
    else:
        pth_path = ""

    model = init_model(model_type, output_features_count, pretrain)

    # 将模型和数据移动到GPU上
    model = model.to(device)

    if len(pth_path) > 0:
        print(f"Load model from {pth_path}")
        model.load_state_dict(torch.load(pth_path, map_location=device))

    criterion = nn.CrossEntropyLoss().to(device)

    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    for epoch in range(num_epochs):
        time_sleep(1)

        total_loss = 0.0
        model.train()
        # 使用tqdm显示进度条
        for inputs, labels in tqdm(data_loader_train, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            # 将数据移动到GPU
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(data_loader_train)

        if len(pth_save_dir_path) > 0:
            torch.save(
                model.state_dict(),
                os.path.join(
                    pth_save_dir_path,
                    get_pth_name(
                        model_type,
                        model_label,
                        str(epoch + 1)
                    )
                )
            )

        if data_loader_val is None:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {average_loss}')
            continue

        model.eval()
        count_correct = 0
        count_total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(data_loader_val, desc='Testing'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                count_total += labels.size(0)
                count_correct += (predicted == labels).sum().item()

        accuracy = count_correct / count_total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {average_loss}, Accuracy: {(accuracy * 100):.4f}%')

    latest_pth_path = ""
    if len(pth_save_dir_path) > 0:
        latest_pth_name = get_pth_name(
            model_type,
            model_label,
            "latest"
        )
        latest_pth_path = os.path.join(
            pth_save_dir_path,
            latest_pth_name
        )
        torch.save(
            model.state_dict(),
            latest_pth_path
        )

    print('Training finished.')
    if len(model_label) > 0:
        print(f'Model: {model_label} Total Epochs: {num_epochs}')
    if len(latest_pth_path) > 0:
        print(f'Latest Model Path: {latest_pth_path}')
