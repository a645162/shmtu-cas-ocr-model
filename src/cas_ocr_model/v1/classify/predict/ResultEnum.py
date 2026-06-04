from enum import Enum


class OperatorEnum(Enum):
    Add = 0
    Add_CHS = 1
    Minus = 2
    Minus_CHS = 3
    Multiply = 4
    Multiply_CHS = 5


def get_operator_type_by_int(int_type: OperatorEnum):
    if int_type == OperatorEnum.Add or int_type == OperatorEnum.Add_CHS:
        return OperatorEnum.Add
    elif int_type == OperatorEnum.Minus or int_type == OperatorEnum.Minus_CHS:
        return OperatorEnum.Minus
    elif int_type == OperatorEnum.Multiply or int_type == OperatorEnum.Multiply_CHS:
        return OperatorEnum.Multiply
    else:
        raise ValueError("int_type must be in OperatorEnum")


def get_operator_type_str(int_type: OperatorEnum):
    int_type = get_operator_type_by_int(int_type)
    if int_type == OperatorEnum.Add:
        return "+"
    elif int_type == OperatorEnum.Minus:
        return "-"
    elif int_type == OperatorEnum.Multiply:
        return "*"
    else:
        raise ValueError("int_type must be in OperatorEnum")


def calculate_operator(left: int, right: int, operator_type: OperatorEnum):
    operator_type = get_operator_type_by_int(operator_type)
    if operator_type == OperatorEnum.Add:
        return left + right
    elif operator_type == OperatorEnum.Minus:
        return left - right
    elif operator_type == OperatorEnum.Multiply:
        return left * right
    else:
        raise ValueError("operator_type must be in OperatorEnum")
