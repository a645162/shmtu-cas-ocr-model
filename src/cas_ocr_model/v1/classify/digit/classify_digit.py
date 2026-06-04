import os.path

from torchvision import transforms
from torch.utils.data import DataLoader

from cas_ocr_model.v1.classify.step.classify import classify_by_model
from cas_ocr_model.v1.classify.utils.devices_selector import get_recommended_device
from cas_ocr_model.v1.config import config
from cas_ocr_model.v1.classify.utils.dataloader import CustomDataset

# 数据变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset_path = os.path.join(config.work_dir_path, "Spilt/MainBody_symbol/0")

test_dataset = CustomDataset(
    root=dataset_path,
    transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

device = get_recommended_device()

classify_by_model(
    device,
    10,
    config.model_digit_type,
    os.path.join(config.pth_save_dir_path, "resnet34_digit_latest.pth"),
    test_loader,
    os.path.join(config.work_dir_path, "Test/Digit_Predictions/Digit_Predictions_symbol_0_test")
)
