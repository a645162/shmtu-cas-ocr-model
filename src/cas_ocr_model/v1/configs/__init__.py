"""集中配置：路径、默认超参、模型注册。"""

from .paths import (
    prj_root_path,
    work_dir_path,
    pth_save_dir_path,
    dataset_dir_path,
    system_tmp_dir_path,
)
from .defaults import (
    cpu_process_count,
    thresh,
    equal_symbol_key_start,
    equal_symbol_key_end,
    key_point_symbol,
    key_point_chs,
    batch_size,
    epoch_equal_symbol,
    epoch_operator,
    epoch_mnist,
    epoch_digit,
    data_transform_rotate_degree,
    pretrain_on_mnist,
    input_name,
    output_name,
)
from .model import ModelType, model_type_to_path_str, get_pth_name
