"""通用图像文件夹 Dataset：扫描根目录下的所有图像。"""

from __future__ import annotations

from PIL import Image
from torch.utils.data import Dataset

from ..helpers.filesystem import get_all_files


class CustomDataset(Dataset):
    """自动收集目录下的 .jpg/.jpeg/.png/.bmp 图像。"""

    def __init__(self, root: str, transform=None, include_subdir: bool = False) -> None:
        self.root = root
        self.transform = transform
        self.image_paths = [
            p
            for p in get_all_files(root, include_subdir=include_subdir)
            if p.endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, img_path
