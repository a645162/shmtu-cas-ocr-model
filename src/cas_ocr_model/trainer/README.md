# CAS CAPTCHA 3-head Trainer

基于 **HuggingFace Accelerate** 的 8 卡 DDP 训练项目。**单 CNN 一次前向** 同时输出 3 个分类头:

| 头 | 类别数 | 说明 |
|---|---|---|
| `digit_left_logits`  | 10  | 第一个数字 (0-9) |
| `operator_logits`    | 4   | 运算符 (`+ - * /`, 4 类统一) |
| `digit_right_logits` | 10  | 第二个数字 (0-9) |

与 `v1` 区别: 不再切 3 段、不再单独训练等号识别模型, 一个 ResNet-18 backbone 一次出 3 个 logits。

## 目录

```
trainer/
├── config.py     # 配置 dataclass + CLI + YAML 加载
├── model.py      # CaptchaTripleHeadCNN (ResNet-18/34 + 3 头)
├── data.py       # CaptchaPairDataset: 扫描 jpg+json, 灰度+二值化
├── losses.py     # 3-head 联合 CE + 准确率
├── train.py      # accelerate 训练入口 (DDP + fp16 + 线性 warmup + AdamW)
├── eval.py       # 单/多卡评估
├── export.py     # ONNX 导出
├── configs/
│   └── 8gpu_ddp.yaml
└── requirements.txt
```

## 数据集

由 `cas_ocr_model.datasets.dataset_collector` 采集:

```bash
python -m cas_ocr_model.datasets.dataset_collector \
    --backend restful --ocr-url http://127.0.0.1:21600 \
    --output ./dataset --count 5000 --processes 4 --per-process 8
```

目录结构:
```
dataset/
  00000000.jpg
  00000000.json   {"expression": "12+34=46", ...}
  ...
```

`data.py` 自动按 `^(\d)([+\-*/])(\d)=$` 正则解析 expression; 不符合则跳过。

## 训练 (8 卡 DDP)

```bash
# accelerate launch (推荐)
accelerate launch --num_processes 8 --mixed_precision fp16 \
    -m cas_ocr_model.trainer.train \
    --config src/cas_ocr_model/trainer/configs/8gpu_ddp.yaml

# 或 torchrun
torchrun --nproc_per_node=8 -m cas_ocr_model.trainer.train \
    --config src/cas_ocr_model/trainer/configs/8gpu_ddp.yaml

# 或完全 CLI (不走 YAML)
torchrun --nproc_per_node=8 -m cas_ocr_model.trainer.train \
    --data-root /path/to/dataset --output-dir ./runs/exp1 \
    --epochs 30 --per-device-batch-size 256 --learning-rate 8e-3 \
    --mixed-precision fp16 --label-smoothing 0.05
```

要点:
* **fp16 混合精度** (用户偏好; A100/H100/RTX40 都支持)
* **per_device_batch_size=256** × 8 卡 = 2048 effective
* **学习率 8e-3** = 1e-3 × 8 (线性缩放规则)
* **5% 线性 warmup** + cosine 衰减
* **梯度裁剪** L2=1.0
* **AdamW** + 1e-4 weight decay
* **ImageNet 预训练 backbone** (`resnet18` 或 `resnet34`)
* **标签平滑** 0.05

## 评估

```bash
python -m cas_ocr_model.trainer.eval \
    --checkpoint ./runs/8gpu_ddp/best.pt \
    --data-root /path/to/dataset
```

输出:
```
[eval] checkpoint=... n_val=500 loss=0.0312 acc_dl=0.9912 acc_op=0.9978 acc_dr=0.9896 acc_full=0.9802
```

## 导出 ONNX

```bash
python -m cas_ocr_model.trainer.export \
    --checkpoint ./runs/8gpu_ddp/best.pt \
    --output ./runs/8gpu_ddp/model.onnx \
    --image-size-h 64 --image-size-w 192 --dynamic-batch
```

输入: `(B, 1, 64, 192) float32 ∈ [0, 1]`
输出: 3 个 logits `(B, 10)`, `(B, 4)`, `(B, 10)`

## 依赖

```bash
pip install -r requirements.txt
```

`accelerate>=0.27` 是 DDP 入口, 第一次跑前执行 `accelerate config` 选择本机环境。
