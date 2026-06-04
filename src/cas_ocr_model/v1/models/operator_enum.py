"""运算符枚举与运算工具。"""

from __future__ import annotations

from enum import Enum


class OperatorEnum(Enum):
    Add = 0
    Add_CHS = 1
    Minus = 2
    Minus_CHS = 3
    Multiply = 4
    Multiply_CHS = 5


def get_operator_type_by_int(int_type: OperatorEnum) -> OperatorEnum:
    """把 Add_CHS/Minus_CHS/Multiply_CHS 归一为对应的英文形式。"""
    if int_type in (OperatorEnum.Add, OperatorEnum.Add_CHS):
        return OperatorEnum.Add
    if int_type in (OperatorEnum.Minus, OperatorEnum.Minus_CHS):
        return OperatorEnum.Minus
    if int_type in (OperatorEnum.Multiply, OperatorEnum.Multiply_CHS):
        return OperatorEnum.Multiply
    raise ValueError(f"Invalid operator: {int_type}")


def get_operator_type_str(int_type: OperatorEnum) -> str:
    """返回运算符字符串 (+ - *)。"""
    canonical = get_operator_type_by_int(int_type)
    if canonical == OperatorEnum.Add:
        return "+"
    if canonical == OperatorEnum.Minus:
        return "-"
    if canonical == OperatorEnum.Multiply:
        return "*"
    raise ValueError(f"Invalid operator: {int_type}")


def calculate_operator(left: int, right: int, operator_type: OperatorEnum) -> int:
    """根据运算符计算 left OP right。"""
    canonical = get_operator_type_by_int(operator_type)
    if canonical == OperatorEnum.Add:
        return left + right
    if canonical == OperatorEnum.Minus:
        return left - right
    if canonical == OperatorEnum.Multiply:
        return left * right
    raise ValueError(f"Invalid operator: {operator_type}")
