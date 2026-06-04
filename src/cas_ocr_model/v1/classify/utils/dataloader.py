from PIL import Image
from torch.utils.data import Dataset

from cas_ocr_model.v1.utils.files.get_files import get_all_files


class CustomDataset(Dataset):
    def __init__(self, root, transform=None, include_subdir: bool = False):
        self.root = root
        self.transform = transform
        self.image_paths = [
            file_path
            for file_path in get_all_files(root, include_subdir=include_subdir)
            if file_path.endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, img_path  # 返回图像和路径
