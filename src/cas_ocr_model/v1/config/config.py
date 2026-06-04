import os
import tempfile

from cas_ocr_model.v1.classify.model.model_type import ModelType

prj_root_path = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)
print("Root Path:", prj_root_path)

work_dir_path = os.path.join(prj_root_path, "workdir")
pth_save_dir_path = os.path.join(work_dir_path, "Models")
dataset_dir_path = os.path.join(work_dir_path, "Datasets")

print("Work Dir Path:", work_dir_path)
print("Pth Save Dir Path:", pth_save_dir_path)
print("Dataset Dir Path:", dataset_dir_path)

cpu_process_count = 10

# preprocess

# Original Image

thresh = 200

# equal symbol
equal_symbol_key_start = 0.7
equal_symbol_key_end = 1

# Spilt Key Point
key_point_symbol = [0.25, 0.58, 0.75]
key_point_chs = [0.15, 0.33, 0.46]

# Train

batch_size = 64

# Equal Symbol

model_equal_symbol_type = ModelType.ResNet_18

epoch_equal_symbol = 2

# Operator

model_operator_type = ModelType.ResNet_18

epoch_operator = 3

# Digit

data_transform_rotate_degree = 15
model_digit_type = ModelType.ResNet_34
pretrain_on_mnist = True

epoch_mnist = 3
epoch_digit = 2

if len(key_point_symbol) != 3 or len(key_point_chs) != 3:
    raise ValueError("key_point_symbol and key_point_chs must have 3 elements")

system_tmp_dir_path = tempfile.gettempdir()

# onnx

input_name = "input"
output_name = "output"
