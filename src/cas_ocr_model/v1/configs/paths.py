"""路径配置：项目根目录、workdir、数据集、模型输出路径。"""

import os
import tempfile

# 本包 (cas_ocr_model.v1) 所在目录
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
# src/cas_ocr_model
PKG_PARENT_DIR = os.path.dirname(PACKAGE_DIR)
# src
SRC_DIR = os.path.dirname(PKG_PARENT_DIR)
# 项目根目录
prj_root_path = os.path.dirname(SRC_DIR)

work_dir_path = os.path.join(prj_root_path, "workdir")
pth_save_dir_path = os.path.join(work_dir_path, "Models")
dataset_dir_path = os.path.join(work_dir_path, "Datasets")
system_tmp_dir_path = tempfile.gettempdir()

print("[configs.paths] project root:", prj_root_path)
print("[configs.paths] workdir:    ", work_dir_path)
