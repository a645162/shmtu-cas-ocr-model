# V2 快速开始

当前版本 (v2.0) 采用 **TriSlot Decoder** 架构：单 CNN 一次前向输出 3 个分类头（数字/运算符/数字），无需图像切割。

## 环境准备

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

## 数据集

数据集由 jpg+json 对组成：

```
dataset/
  00000000.jpg
  00000000.json   # {"expression": "9 - 2 = 7", "answer": "7", ...}
  ...
```

可通过以下方式获取：

- **Hugging Face**: https://huggingface.co/datasets/a645162/shmtu_cas_validate_code
- **Gitee AI (国内较快)**: https://ai.gitee.com/datasets/a645162/shmtu_cas_validate_code
- **自行采集**: 参见 [数据采集](./v2-data-collection)

## 训练

### 单卡训练

```bash
python -m cas_ocr_model.trainer.train \
    --data-root ./dataset \
    --output-dir ./runs/exp1 \
    --epochs 200 \
    --per-device-batch-size 128 \
    --learning-rate 1e-3
```

### 8 卡 DDP 训练（推荐）

```bash
# 使用 accelerate launch
accelerate launch --num_processes 8 --num_machines 1 \
    --main_process_port "$(torch_ddp_port)" \
    --dynamo_backend inductor --mixed_precision bf16 \
    -m cas_ocr_model.trainer.train \
    --config src/cas_ocr_model/trainer/configs/8gpu_ddp.yaml

# 或使用 torchrun
torchrun --nproc_per_node=8 \
    -m cas_ocr_model.trainer.train \
    --config src/cas_ocr_model/trainer/configs/8gpu_ddp.yaml
```

关键参数说明：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--per-device-batch-size` | 256 | 单卡批次大小，8 卡 × 256 = 2048 effective |
| `--learning-rate` | 8e-3 | 线性缩放: 1e-3 × GPU 数 |
| `--mixed-precision` | bf16 | 混合精度训练 |
| `--early-stop-patience` | -1 | -1 = 自动取总 epoch 的 20% |

### 断点续训

```bash
torchrun --nproc_per_node=8 \
    -m cas_ocr_model.trainer.train \
    --config src/cas_ocr_model/trainer/configs/8gpu_ddp.yaml \
    --output-dir ./runs/exp1 \
    --resume-from ./runs/exp1/last.pt
```

## 评估

```bash
python -m cas_ocr_model.trainer.eval \
    --checkpoint ./runs/exp1/best.pt \
    --data-root ./dataset
```

输出示例：

```
[eval] checkpoint=... n_val=500 loss=0.0312 acc_dl=0.9912 acc_op=0.9978 acc_dr=0.9896 acc_full=0.9802
```

## 导出 ONNX

```bash
python -m cas_ocr_model.trainer.export \
    --checkpoint ./runs/exp1/best.pt \
    --output ./runs/exp1/model.onnx \
    --image-size-h 64 --image-size-w 192 --dynamic-batch
```

- 输入: `(B, 1, 64, 192)` float32 ∈ [0, 1]
- 输出: 3 个 logits `(B, 10)`, `(B, 3)`, `(B, 10)`

## 推理

```bash
# PyTorch 推理
python -m cas_ocr_model.inference --checkpoint ./runs/exp1/best.pt --image ./test.jpg

# ONNX 推理
python -m cas_ocr_model.inference --backend onnx --onnx-path ./runs/exp1/model.onnx --image ./test.jpg

# NCNN 推理
python -m cas_ocr_model.inference --backend ncnn --ncnn-param ./runs/exp1/model.param --ncnn-bin ./runs/exp1/model.bin --image ./test.jpg
```

## 接入 wandb

```bash
pip install -e ".[wandb]"

accelerate launch --num_processes 8 --mixed_precision bf16 \
    -m cas_ocr_model.trainer.train \
    --config src/cas_ocr_model/trainer/configs/8gpu_ddp.yaml \
    --report-to wandb \
    --tracker-project-name cas-ocr-train \
    --wandb-run-name mobilenet-ddp-exp1
```
