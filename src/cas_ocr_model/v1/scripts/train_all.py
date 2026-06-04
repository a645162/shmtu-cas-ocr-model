"""一键训练三个模型：等号 -> 运算符 -> 数字。"""

from __future__ import annotations

from ..data_modules.device import get_recommended_device
from ..tasks.digit.training import train_digit_all
from ..tasks.equal_symbol.training import train_equal_symbol
from ..tasks.operator.training import train_operator


def main() -> None:
    device = get_recommended_device()
    train_equal_symbol(device)
    train_operator(device)
    train_digit_all(device)


if __name__ == "__main__":
    main()
