import cv2
import numpy as np
import tensorrt as trt
import pycuda.autoinit  # 初始化CUDA
import pycuda.driver as cuda
from PIL import Image

# 确保TensorRT版本与ONNX模型兼容
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)

# 加载TensorRT引擎
with open("resnet34_int8.engine", "rb") as f:
    engine = trt_runtime.deserialize_cuda_engine(f.read())

# 创建执行上下文
context = engine.create_execution_context()

# 分配GPU内存
host_inputs = [cuda.pagelocked_empty(i.shape, dtype=trt.nptype(trt.float32)) for i in engine.get_binding_shape(0)]
host_outputs = [cuda.pagelocked_empty(i.shape, dtype=trt.nptype(trt.float32)) for i in engine.get_binding_shape(1)]

# 读取图片并进行预处理
image_path = "path_to_your_image.jpg"
image = Image.open(image_path).resize((224, 224))  # 根据模型输入大小调整图片大小
image_np = np.array(image) / 255.0  # 归一化到[0, 1]
image_np = np.transpose(image_np, (2, 0, 1))  # 修改通道顺序为CHW
image_np = np.ascontiguousarray(image_np)

# 将图片数据复制到GPU内存中
cuda.memcpy_htod(host_inputs[0], image_np)

# 设置输入和输出
context.set_binding_shape(0, host_inputs[0].shape)  # 输入
context.set_binding_shape(1, host_outputs[0].shape)  # 输出

# 执行推理
cuda.Context.synchronize()  # 确保CUDA操作完成
stream = cuda.Stream()
context.execute_async(bindings=[int(host_inputs[0]), int(host_outputs[0])], stream_handle=stream.handle)
stream.synchronize()  # 等待推理完成

# 获取输出
output = cuda.pagelocked_empty(host_outputs[0].shape, dtype=trt.nptype(trt.float32))
cuda.memcpy_dtoh(output, host_outputs[0])

# 处理输出，例如取概率最高的类别
predictions = np.argmax(output, axis=1)
print("Predicted class:", predictions[0])