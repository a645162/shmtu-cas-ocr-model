"""一键训练三个模型：等号 -> 运算符 -> 数字。"""

from cas_ocr_model.v1.data_modules.device import get_recommended_device
from cas_ocr_model.v1.tasks.digit.training import train_digit_all
from cas_ocr_model.v1.tasks.equal_symbol.training import train_equal_symbol
from cas_ocr_model.v1.tasks.operator.training import train_operator

if __name__ == "__main__":
    device = get_recommended_device()
    train_equal_symbol(device)
    train_operator(device)
    train_digit_all(device)
