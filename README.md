# SHMTU CAS OCR Model

上海海事大学统一认证平台验证码识别模型训练与推理项目。

## 模型版本

### V2 — TriSlot Decoder（当前版本）

单 CNN 端到端识别：共享 backbone + 槽位注意力解码器，一次前向输出数字/运算符/数字三个分类头。

- **架构**：MobileNetV3-Small + TriSlot Decoder
- **输入**：`(B, 1, 64, 192)` 灰度图
- **输出**：`digit_left` (10类) + `operator` (3类: `+`, `-`, `*`) + `digit_right` (10类)
- **无需图像切割**，整图输入

### V1 — 三模型分离（历史版本）

图像切割 → 独立 ResNet 分类，代码完整保留在 `src/cas_ocr_model/v1/` 中。

## 快速开始

```bash
# 克隆项目
git clone https://github.com/a645162/shmtu-cas-ocr-model.git
cd shmtu-cas-ocr-model

# 安装依赖 (Python >= 3.10)
pip install -e .            # 核心依赖 (训练 + PyTorch 推理)
pip install -e ".[onnx]"   # ONNX 导出/推理
pip install -e ".[ncnn]"   # NCNN 导出/推理
pip install -e ".[wandb]"  # wandb 实验追踪
```

详细使用说明请参阅 [文档站点](https://a645162.github.io/shmtu-cas-ocr-model/)。

## 训练

```bash
# 单卡训练
python -m cas_ocr_model.trainer.train \
    --data-root ./dataset \
    --output-dir ./runs/exp1 \
    --epochs 200

# 8 卡 DDP 训练（推荐）
accelerate launch --num_processes 8 --mixed_precision bf16 \
    -m cas_ocr_model.trainer.train \
    --config src/cas_ocr_model/trainer/configs/8gpu_ddp.yaml
```

## 推理

```bash
# PyTorch
python -m cas_ocr_model.inference --checkpoint ./runs/exp1/best.pt --image ./test.jpg

# ONNX
python -m cas_ocr_model.inference --backend onnx --onnx-path ./model.onnx --image ./test.jpg

# NCNN
python -m cas_ocr_model.inference --backend ncnn --ncnn-param ./model.param --ncnn-bin ./model.bin --image ./test.jpg
```

## Release 资产

### V2 (v2.0.2)

| 引擎 | 精度 | 文件 | 大小 |
|---|---|---|---|
| PyTorch | fp32 | `mobilenet_v3_small.trislot_decoder.v2_0.pt` | 17.2 MB |
| ONNX | fp16 | `*.fp16.onnx` | 2.9 MB |
| ONNX | fp32 | `*.fp32.onnx` | 5.7 MB |
| NCNN | fp16 | `*.fp16.param` + `*.fp16.bin` | 13 KB + 2.8 MB |
| NCNN | fp32 | `*.fp32.param` + `*.fp32.bin` | 13 KB + 5.6 MB |

Release 中附带 `model-assets.json`（元数据 + SHA256）与 `SHA256SUMS.txt`。

### V1

| Release | 内容 |
|---|---|
| `v1.0` | 3 × PyTorch checkpoint (ResNet-34/18) |
| `v1.0-ONNX` | 3 × ONNX 导出 |

## 数据集

- **Hugging Face**: https://huggingface.co/datasets/a645162/shmtu_cas_validate_code
- **Gitee AI (国内较快)**: https://ai.gitee.com/datasets/a645162/shmtu_cas_validate_code

## 相关项目

- [shmtu-cas-ocr-server](https://github.com/a645162/shmtu-cas-ocr-server) — C++ OCR 服务 (Drogon + ncnn)
- [shmtu-terminal-tauri](https://github.com/a645162/shmtu-terminal-tauri) — Tauri v2 桌面应用
- [shmtu-cas-kotlin](https://github.com/a645162/shmtu-cas-kotlin) — 统一认证登录库

## 免责声明

本项目仅供学习交流使用，不得用于商业用途。本项目为个人开发，与上海海事大学无关，仅供学习参考，请勿用于非法用途。
