"""一键导出三个模型到 ONNX。"""

from __future__ import annotations

from ..tasks.digit.onnx_export import export_digit_to_onnx
from ..tasks.equal_symbol.onnx_export import export_equal_symbol_to_onnx
from ..tasks.operator.onnx_export import export_operator_to_onnx


def main() -> None:
    export_equal_symbol_to_onnx()
    export_operator_to_onnx()
    export_digit_to_onnx()


if __name__ == "__main__":
    main()
