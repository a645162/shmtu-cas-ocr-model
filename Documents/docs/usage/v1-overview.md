# V1 架构与使用 (旧版)

> **注意**: V1 为历史版本，当前主力开发已迁移至 V2 (TriSlot Decoder)。V1 代码完整保留在 `src/cas_ocr_model/v1/` 和 `deprecated/` 目录中。

## V1 架构概述

V1 采用 **三模型分离** 策略：先将验证码图像切割为 3 段，然后分别送入 3 个独立的 ResNet 分类器。

```
原始验证码图像
    │
    ▼
┌──────────┐
│ 图像预处理 │  灰度化 → 二值化 → 7:3 分割
└──────────┘
    │
    ├──► 左段 (数字)  ──► ResNet-18 (10类: 0-9)
    ├──► 中段 (运算符) ──► ResNet-18 (6类: +,-,×,÷,=,其他)
    └──► 右段 (数字)  ──► ResNet-18 (10类: 0-9)
```

## 与 V2 的核心区别

| 特性 | V1 | V2 |
|---|---|---|
| 模型数量 | 3 个独立 ResNet | 1 个统一 CNN + TriSlot Decoder |
| 图像处理 | 需切割为 3 段 | 整图输入，无需切割 |
| 运算符类别 | 6 类 (含等号) | 3 类 (`+`, `-`, `*`) |
| 训练方式 | 分别训练 3 个模型 | 端到端联合训练 |
| 推理效率 | 3 次前向 | 1 次前向 |
| 部署复杂度 | 需管理 3 个模型文件 | 单模型文件 |

## V1 代码结构

```
src/cas_ocr_model/v1/
├── configs/          # 配置 (defaults, model, paths)
├── data/
│   ├── preparation/  # 图像预处理 (提取段、去白边、缩放、灰度化)
│   ├── splitting/    # 验证码切割
│   └── clustering/   # K-means 字符聚类
├── data_modules/     # PyTorch Dataset / Device 工具
├── helpers/          # 文件系统、图像工具
├── inference/        # 推理
│   ├── engines/      # ONNX Runtime / OpenVINO / TensorRT / 量化
│   └── predictor.py  # 统一推理器
├── models/           # ResNet 模型定义
├── tasks/
│   ├── digit/        # 数字分类 (10类)
│   ├── operator/     # 运算符分类 (6类)
│   └── equal_symbol/ # 等号检测
├── training/         # 训练逻辑
├── verification/     # 验证码求解、CAS 登录测试
└── scripts/          # 一键训练/导出脚本
```

## V1 使用方式

### 训练所有模型

```bash
python -m cas_ocr_model.v1.scripts.train_all
```

### 导出 ONNX

```bash
python -m cas_ocr_model.v1.scripts.export_all_onnx
```

### 推理

```python
from cas_ocr_model.v1.inference.predictor import Predictor

predictor = Predictor(model_dir="workdir/Models")
result = predictor.predict(image_path="test.jpg")
```

## 早期代码

`deprecated/` 目录保留了 V1 最早期的独立脚本：

- `deprecated/digit/` — 数字识别训练 (含 MNIST 对比)
- `deprecated/equal_symbol/` — 等号检测
- `deprecated/operator/` — 运算符识别
- `deprecated/pre/` — 图像预处理

这些代码仅作历史参考，不建议用于新项目。

## V1 Release 资产

V1 模型权重发布在 GitHub Release 上，包含原始 PyTorch checkpoint 和 ONNX 导出：

| Release | 文件 | 大小 |
|---|---|---|
| `v1.0` | `resnet34_digit_latest.pth` | 81.4 MB |
| `v1.0` | `resnet18_operator_latest.pth` | 42.8 MB |
| `v1.0` | `resnet18_equal_symbol_latest.pth` | 42.7 MB |
| `v1.0-ONNX` | `resnet34_digit_latest.onnx` | 81.1 MB |
| `v1.0-ONNX` | `resnet18_operator_latest.onnx` | 42.7 MB |
| `v1.0-ONNX` | `resnet18_equal_symbol_latest.onnx` | 42.6 MB |

下载地址：

- PyTorch 权重：`https://github.com/a645162/shmtu-cas-ocr-model/releases/tag/v1.0`
- ONNX 权重：`https://github.com/a645162/shmtu-cas-ocr-model/releases/tag/v1.0-ONNX`
- SHA256 校验：各 release 中附带 `SHA256SUMS.txt`

V1 采用硬编码文件列表 + `SHA256SUMS.txt` 校验，不支持 `model-assets.json` 智能下载（该功能仅 V2 支持）。
