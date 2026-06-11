"""集中配置：路径、默认超参、模型注册。"""

from .defaults import (
    batch_size,
    cpu_process_count,
    data_transform_rotate_degree,
    epoch_digit,
    epoch_equal_symbol,
    epoch_mnist,
    epoch_operator,
    equal_symbol_key_end,
    equal_symbol_key_start,
    input_name,
    key_point_chs,
    key_point_symbol,
    output_name,
    pretrain_on_mnist,
    thresh,
)
from .model import ModelType, get_pth_name, model_type_to_path_str
from .paths import (
    dataset_dir_path,
    prj_root_path,
    pth_save_dir_path,
    system_tmp_dir_path,
    work_dir_path,
)
