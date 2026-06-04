import os

import cv2
import numpy as np
from openvino.inference_engine import IECore

from cas_ocr_model.v1.config import config

# 加载OpenVINO插件
ie = IECore()

# 加载网络模型
net = ie.read_network(
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

# 配置输入
input_blob = next(iter(net.input_info))
input_shape = net.input_info[input_blob].input_data.shape
input_precision = net.input_info[input_blob].precision

# 加载执行网络
exec_net = ie.load_network(network=net, device_name="CPU")


# 预处理图像
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # ResNet-34期望的输入大小
    image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    image = image.reshape(1, 3, 224, 224)
    image = image.astype(np.float32)
    image /= 255.0
    return image


# 推理
def infer(image_path):
    image = preprocess_image(image_path)
    res = exec_net.infer(inputs={input_blob: image})
    return res


# 获取输出
def get_output(res):
    output_blob = next(iter(net.outputs))
    output = res[output_blob]
    return output


def infer_img_path(image_path):
    result = infer(image_path)
    output = get_output(result)

    # list to np.array
    p_list = np.array(list(output[0]))

    # output max index
    print("Output:", p_list.argmax())


if __name__ == "__main__":
    for i in range(10):
        infer_img_path(f"test/{i}.png")
