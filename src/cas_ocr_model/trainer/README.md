# CAS CAPTCHA 3-head Trainer

基于 **HuggingFace Accelerate** 的 8 卡 DDP 训练项目。**单 CNN 一次前向** 同时输出 3 个分类头:

| 头 | 类别数 | 说明 |
|---|---|---|
| `digit_left_logits`  | 10  | 第一个数字 (0-9) |
| `operator_logits`    | 3   | 运算符 (`+ - *`, 自动兼容 `加/减/乘`) |
| `digit_right_logits` | 10  | 第二个数字 (0-9) |

与 `v1` 区别: 不再切 3 段、不再单独训练等号识别模型, 一个 ResNet-18 backbone 一次出 3 个 logits。

## 目录

```
trainer/
├── config.py     # 配置 dataclass + CLI + YAML/TOML 加载
├── model.py      # CaptchaTripleHeadCNN (ResNet-18/34 + 3 头)
├── data.py       # CaptchaPairDataset: 扫描 jpg+json, 灰度+二值化
├── losses.py     # 3-head 联合 CE + 准确率
├── train.py      # accelerate 训练入口 (DDP + fp16 + 线性 warmup + AdamW)
├── eval.py       # 单/多卡评估
├── export.py     # ONNX 导出
├── configs/
│   └── 8gpu_ddp.yaml
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
  00000000.json   {"expression": "9 - 2 = 7", "answer": "7", ...}
  ...
```

`data.py` 会自动解析 `expression`，支持这几类格式并统一映射到 3 类运算符:
- `9 - 2 = 7`
- `9-2=7`
- `9减2等于7`
- `9 乘 2 = 18`

解析失败的样本会被自动跳过。

## 训练 (8 卡 DDP)

```bash
# accelerate launch (推荐)
accelerate launch --num_processes 8 --num_machines 1 --dynamo_backend no --mixed_precision fp16 \
    -m cas_ocr_model.trainer.train \
    --config src/cas_ocr_model/trainer/configs/8gpu_ddp.yaml

# 或 torchrun
torchrun --nproc_per_node=8 -m cas_ocr_model.trainer.train \
    --config src/cas_ocr_model/trainer/configs/8gpu_ddp.yaml

# 或 TOML 配置
accelerate launch --num_processes 8 --num_machines 1 --dynamo_backend no --mixed_precision fp16 \
    -m cas_ocr_model.trainer.train \
    --config src/cas_ocr_model/trainer/configs/mobile_small.toml

# 或完全 CLI (不走配置文件)
torchrun --nproc_per_node=8 -m cas_ocr_model.trainer.train \
    --data-root /path/to/dataset --output-dir ./runs/exp1 \
    --epochs 500 --early-stop-patience -1 --per-device-batch-size 256 --learning-rate 8e-3 \
    --mixed-precision fp16 --label-smoothing 0.05

# 从 last.pt 断点续训
torchrun --nproc_per_node=8 -m cas_ocr_model.trainer.train \
    --config src/cas_ocr_model/trainer/configs/8gpu_ddp.yaml \
    --output-dir ./runs/exp1 \
    --resume-from ./runs/exp1/last.pt

# 接入 wandb
accelerate launch --num_processes 8 --num_machines 1 --dynamo_backend no --mixed_precision fp16 \
    -m cas_ocr_model.trainer.train \
    --config src/cas_ocr_model/trainer/configs/8gpu_ddp.yaml \
    --report-to wandb \
    --tracker-project-name cas-ocr-train \
    --wandb-run-name mobilenet-ddp-exp1 \
    --wandb-tags ddp,captcha,mobilenet
```

要点:
* **fp16 混合精度** (用户偏好; A100/H100/RTX40 都支持)
* **per_device_batch_size=256** × 8 卡 = 2048 effective
* **学习率 8e-3** = 1e-3 × 8 (线性缩放规则)
* **5% 线性 warmup** + cosine 衰减
* **Early stop**: `train.early_stop_patience=0` 关闭; `-1` 自动取总 epoch 的 20%; 正整数表示连续多少个验证 epoch 不提升后停止
* **梯度裁剪** L2=1.0
* **AdamW** + 1e-4 weight decay
* **ImageNet 预训练 backbone** (`resnet18` 或 `resnet34`)
* **标签平滑** 0.05
* **主进程 rich 进度条**: 交互式终端显示 step/loss/acc/lr/吞吐, 非 TTY 自动回退文本日志
* **DDP 全局聚合日志**: train/val/test 指标按所有 rank 汇总, 可直接用于 console 和 wandb
* **断点续训**: `--resume-from last.pt` 会恢复 model / optimizer / scheduler / global_step / best_acc / early-stop 计数
* **逐 epoch 指标落盘**: 每轮都会写 `output_dir/epochs/epoch_XXXX.json`，并维护 `output_dir/metrics_history.json`
* **wandb 自动探测**: `report_to=auto` 时, 若已安装 `wandb` 且环境变量未禁用, 启动训练时会自动接入

## wandb

安装:

```bash
pip install -e .[wandb]
```

说明:
* 默认 `report_to=auto`, 若已安装 `wandb` 会自动启用; 显式设为 `none` 可关闭
* `--report-to wandb` 开启后, 训练会通过 `accelerate` tracker 自动只在主进程写 wandb
* `--tracker-project-name` 控制 project 名称
* `wandb_run_name` 未设置时会按 `output_dir` 自动生成, 例如 `runs/8gpu_ddp/20260608_153000` -> `8gpu_ddp/20260608_153000`
* `--wandb-run-name` / `--wandb-entity` / `--wandb-tags` 用于附加 run 元数据
* `SHMTU_DISABLE_WANDB=1`、`SHMTU_WANDB_DISABLED=1`、`WANDB_DISABLED=true` 或 `WANDB_MODE=disabled` 都会禁用自动接入
* 若未安装 `wandb` 但显式开启了 `--report-to wandb`, 训练会直接报清晰错误

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
输出: 3 个 logits `(B, 10)`, `(B, 3)`, `(B, 10)`

## 依赖

```bash
pip install -e .
```

`accelerate>=0.27` 是 DDP 入口, 第一次跑前执行 `accelerate config` 选择本机环境。
