"""训练/预处理默认值（超参、关键点、阈值等）。"""

from .model import ModelType

cpu_process_count = 10

# 二值化阈值
thresh = 200

# 等号区域水平裁剪比例
equal_symbol_key_start = 0.7
equal_symbol_key_end = 1.0

# 主体裁剪关键点（按比例）：数字 / 运算符 / 数字 三个区段
key_point_symbol = [0.25, 0.58, 0.75]
key_point_chs = [0.15, 0.33, 0.46]

# ============== 训练 ==============
batch_size = 64
data_transform_rotate_degree = 15

# 等号分类
model_equal_symbol_type = ModelType.ResNet_18
epoch_equal_symbol = 2

# 运算符分类
model_operator_type = ModelType.ResNet_18
epoch_operator = 3

# 数字分类
model_digit_type = ModelType.ResNet_34
pretrain_on_mnist = True
epoch_mnist = 3
epoch_digit = 2

# ============== 导出 ONNX ==============
input_name = "input"
output_name = "output"

if len(key_point_symbol) != 3 or len(key_point_chs) != 3:
    raise ValueError("key_point_symbol and key_point_chs must have 3 elements")
