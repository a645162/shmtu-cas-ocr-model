import cv2
import onnxruntime
import numpy as np
import time

type = 32
# type = 16
type = 8

if type == 32:
    print("Using FP32 model")
elif type == 16:
    print("Using FP16 model")
elif type == 8:
    print("Using INT8 model")
else:
    raise ValueError("type must be 32, 16 or 8")


# 加载并预处理图片
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # ResNet-34期望的输入大小
    image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    image = image.reshape(1, 3, 224, 224)
    image = image.astype(np.float32)
    image /= 255.0
    if type == 32:
        return image.astype(np.float32)
    elif type == 16:
        return image.astype(np.float16)
    elif type == 8:
        # return image.astype(np.uint8)
        return image
    else:
        raise ValueError("type must be 32, 16 or 8")


# 加载ONNX模型

if type == 32:
    onnx_model_path = 'resnet34_digit_latest.onnx'
elif type == 16:
    onnx_model_path = 'resnet34_digit_latest_fp16.onnx'
elif type == 8:
    onnx_model_path = 'resnet34_digit_latest_int8.onnx'
else:
    raise ValueError("type must be 32, 16 or 8")

# session = onnxruntime.InferenceSession(onnx_model_path)
session = onnxruntime.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

image_array = []

for i in range(10):
    image_path = f"test/{i}.png"
    input_image = preprocess_image(image_path)
    image_array.append(input_image)

for i in range(1):
    for i in range(10):
        input_image = image_array[i]

        # 进行推理
        start_time = time.time()
        output = session.run(None, {'input': input_image})
        end_time = time.time()

        # 计算推理速度
        inference_time = end_time - start_time
        print(i, inference_time, "秒")

        # print(output)

        print(np.array(output[0]).argmax())
