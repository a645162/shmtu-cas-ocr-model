import os
import shutil
from tqdm import tqdm
import torch

from cas_ocr_model.v1.classify.model.model_type import ModelType
from cas_ocr_model.v1.classify.model.my_model import init_model
from cas_ocr_model.v1.utils.files.dirs import create_dirs


def classify_by_model(
        device,
        class_count=10,
        model_type=ModelType.ResNet_34,
        pth_path="../../workdir/Models/resnet34_digit_4.pth",
        data_loader=None,
        output_dir=""
):
    model = init_model(model_type, class_count)

    model = model.to(device)

    # 加载已训练的模型参数
    model.load_state_dict(torch.load(pth_path, map_location=device))

    # 设置模型为评估模式
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    create_dirs(list(
        [
            str(os.path.join(output_dir, str(i)))
            for i in range(class_count)
        ]
    ))

    # 遍历测试集进行预测并保存图像
    with torch.no_grad():
        for i, (inputs, img_path) in enumerate(tqdm(data_loader, desc='Predicting')):
            # 将数据移动到GPU
            inputs = inputs.to(device)

            # 进行预测
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # 获取原始图像文件名
            original_filename = os.path.basename(img_path[0])

            # 构建新的文件名，将预测结果添加到文件名开头
            # new_filename = f'{str(predicted.item())}_{original_filename}'

            # 构建新的文件路径
            # new_filepath = os.path.join(output_dir, new_filename)
            new_filepath = os.path.join(output_dir, str(predicted.item()), original_filename)

            # 复制原始图像到目标目录，并使用新的文件名
            shutil.copy(img_path[0], new_filepath)

    print('Prediction finished.')
    print(f'Prediction result saved to {output_dir}')
