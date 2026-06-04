import os.path

import cv2
import numpy as np
from openvino.inference_engine import IECore

from cas_ocr_model.v1.config import config

# 加载 OpenVINO 推理引擎
ie = IECore()

# 加载转换后的模型
model = ie.read_network(
    model=os.path.join(
        config.pth_save_dir_path,
        "resnet34_digit_latest",
        "resnet34_digit_latest.xml"
    ),
    weights=os.path.join(
        config.pth_save_dir_path,
        "resnet34_digit_latest",
        "resnet34_digit_latest.bin"
    )
)

input_info = model.input_info
print("模型输入信息:")
for input_name, input_data in input_info.items():
    print(f"Input name: {input_name}")
    print(f"Input shape: {input_data.input_data.shape}")

# 加载模型到设备
exec_net = ie.load_network(network=model, device_name="CPU")

# 准备输入数据
input_image = cv2.imread('test/1.png')  # 读取输入图像
input_image = cv2.resize(input_image, (224, 224))  # 调整图像大小为模型期望的尺寸
input_image = np.transpose(input_image, (2, 0, 1))  # 调整通道顺序，从 HWC 到 CHW
input_image = np.expand_dims(input_image, axis=0)  # 添加批量维度

# 执行推理
output = exec_net.infer(inputs={'input': input_image})

print("推理结果:", output)

key_list = list(output.keys())

array_list = output[key_list[0]]

# get max index
max_index = np.argmax(array_list)
print("max_index:", max_index)
