"""按已有模型对一组图像做预测并按预测类别归档。"""

from __future__ import annotations

import os
import shutil

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..configs.model import ModelType
from ..helpers.filesystem import create_dirs
from ..models.resnet import init_model


def classify_by_model(
    device: torch.device,
    class_count: int,
    model_type: ModelType,
    pth_path: str,
    data_loader: DataLoader,
    output_dir: str,
) -> None:
    """
    用指定模型对一批图像做预测，把图像按预测类别复制到 output_dir/{class_id}/。

    Args:
        device: torch device。
        class_count: 类别数。
        model_type: 模型类型。
        pth_path: 权重文件路径。
        data_loader: 一次送 1 张图的 DataLoader。
        output_dir: 输出根目录。
    """
    model = init_model(model_type, class_count).to(device)
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    create_dirs([os.path.join(output_dir, str(i)) for i in range(class_count)])

    with torch.no_grad():
        for inputs, img_path in tqdm(data_loader, desc="Classifying"):
            inputs = inputs.to(device)
            _, predicted = torch.max(model(inputs), 1)
            original_filename = os.path.basename(img_path[0])
            target = os.path.join(output_dir, str(predicted.item()), original_filename)
            shutil.copy(img_path[0], target)

    print(f"[classify] done. Output -> {output_dir}")
