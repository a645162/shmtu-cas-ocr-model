import os

from torchvision import transforms
from PIL import Image
import cv2
import torch

from cas_ocr_model.v1.classify.model.model_type import get_pth_name
from cas_ocr_model.v1.classify.model.my_model import init_model
from cas_ocr_model.v1.classify.utils.devices_selector import get_recommended_device
from cas_ocr_model.v1.config import config

device = get_recommended_device()

# 数据变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

model_type = config.model_digit_type
model_label = "digit"

model = init_model(model_type, 10, False)

# 将模型和数据移动到GPU上
model = model.to(device)

# 加载已训练的模型参数
model.load_state_dict(
    torch.load(
        str(
            os.path.join(
                config.pth_save_dir_path,
                get_pth_name(
                    model_type,
                    model_label,
                    "latest"
                )
            )
        ),
        map_location=device
    )
)

# 设置模型为评估模式
model.eval()


def predict_image(image_path):
    # img = Image.open(image_path).convert('RGB')
    img_cv = cv2.imread(image_path)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(img_cv)

    # 对图像进行预处理
    img_tensor = transform(img).unsqueeze(0)  # 添加批量维度

    # 将图像移动到GPU上
    img_tensor = img_tensor.to(device)

    # 进行预测
    with torch.no_grad():
        output = model(img_tensor)
        # print(output)
        _, predicted = torch.max(output, 1)

    # 打印预测结果
    print(f'Prediction for {image_path}: Class {predicted.item()}')


if __name__ == '__main__':
    for i in range(10):
        index = 9 - i
        predict_image(f'test/{index}.png')
