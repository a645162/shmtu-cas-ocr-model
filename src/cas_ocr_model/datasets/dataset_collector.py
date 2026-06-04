"""Backwards-compat shim.

真实实现已迁到 ``cas_ocr_model.datasets.maker``. 本模块只保留旧 import 路径.

调用方式不变:
    python -m cas_ocr_model.datasets.dataset_collector --backend restful ...
"""
from cas_ocr_model.datasets.maker.cli import main

if __name__ == "__main__":
    main()
