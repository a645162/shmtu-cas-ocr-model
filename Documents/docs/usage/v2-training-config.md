# V2 训练配置

V2 训练系统基于 `FullConfig` 数据类，支持 YAML、TOML 配置文件和 CLI 参数三种方式，且可互相覆盖。

## 配置加载优先级

CLI 参数 > 配置文件 > 默认值

## 配置结构

`FullConfig` 由以下子配置组成：

| 子配置 | 说明 |
|---|---|
| `DataConfig` | 数据集路径、图像尺寸、二值化参数、增强开关 |
| `ModelConfig` | 模型版本、backbone 选择、dropout、slot 维度 |
| `TrainConfig` | 训练超参、优化器、调度器、early stop |
| `LossConfig` | 损失权重、label smoothing、focal、槽位约束 |
| `AugmentationConfig` | 增强策略（平移、仿射、形态学、噪点、二值化扰动） |

## 完整配置示例 (YAML)

```yaml
data:
  data_root: /path/to/dataset
  image_size_h: 64           # 输入高度, 可尝试 48 / 64
  image_size_w: 192           # 输入宽度, 可尝试 128 / 160 / 192
  threshold: 200              # fixed 模式阈值
  binarize_mode: min_channel_otsu  # 推荐默认模式
  adaptive_block_size: 25     # adaptive 模式邻域大小 (奇数)
  adaptive_c: 15              # adaptive 模式阈值偏移
  train_ratio: 0.9            # 训练/验证划分比例
  num_workers: 4              # DataLoader worker 数
  pin_memory: true            # CUDA 推荐 true

model:
  version: "2.0"              # 当前模型版本
  backbone: mobilenet_v3_small # backbone 选择
  pretrained: true            # ImageNet 预训练
  dropout: 0.2                # head dropout
  slot_hidden_dim: 256        # TriSlot 隐层维度
  slot_attention_heads: 4     # 注意力头数

train:
  output_dir: ./runs/8gpu_ddp
  seed: 42
  epochs: 500
  early_stop_patience: -1    # -1 = 自动 (总 epoch 的 20%)
  per_device_batch_size: 512
  learning_rate: 8.0e-3
  weight_decay: 1.0e-4
  warmup_ratio: 0.05
  grad_clip: 1.0
  mixed_precision: fp16
  report_to: auto
  use_rich_progress: true

loss:
  weight_digit_left: 1.0
  weight_operator: 1.0
  weight_digit_right: 1.0
  label_smoothing: 0.05
  focal_gamma: 0.0
  weight_slot_order: 0.1
  weight_slot_overlap: 0.05
  slot_margin: 0.10
  enable_slot_right_boundary: true
  weight_slot_right_boundary: 0.01
  slot_right_boundary_max: 0.68
  enable_slot_attention_variance: true
  weight_slot_attention_variance: 0.005
  slot_attention_max_variance: 0.035
  enable_operator_class_balance: true
  operator_class_weights: [1.0, 1.05, 1.1]
```

## Backbone 选项

| Backbone | 说明 | 适用场景 |
|---|---|---|
| `mobilenet_v3_small` | 轻量首选，推荐 | 生产部署、移动端 |
| `mobilenet_v3_large` | 精度更高 | 服务端推理 |
| `resnet18` | 默认基线 | 对比实验 |
| `resnet34` | 更大更慢 | 精度优先 |
| `r50` / `resnet101` | 大模型 | 研究用途 |
| `mobilenetv4_conv_small` / `mobilenetv4_hybrid_medium` / ... | timm 中的 MobileNetV4 快捷别名 | 轻量 CNN 实验 |
| `repvgg_a0` / `repvgg_b1` / ... | timm 中的 RepVGG 快捷别名 | 轻量 CNN 实验 |
| `timm/<model_name>` | 任意 timm backbone | 灵活实验 |

## 二值化模式

| 模式 | 说明 |
|---|---|
| `min_channel_otsu` | 最小通道 Otsu，推荐默认 |
| `gray_otsu` | 灰度 Otsu |
| `adaptive` | 自适应阈值 |
| `fixed` | 固定阈值 (配合 `threshold` 参数) |

## 资产命名规则

训练产出的 checkpoint 和导出文件遵循统一命名：

| 类型 | 命名格式 | 示例 |
|---|---|---|
| Checkpoint | `{backbone}.trislot_decoder.v{version}.pt` | `mobilenet_v3_small.trislot_decoder.v2_0.pt` |
| ONNX (fp32) | `{backbone}.trislot_decoder.v{version}.fp32.onnx` | `mobilenet_v3_small.trislot_decoder.v2_0.fp32.onnx` |
| ONNX (fp16) | `{backbone}.trislot_decoder.v{version}.fp16.onnx` | `mobilenet_v3_small.trislot_decoder.v2_0.fp16.onnx` |
| NCNN param | `{backbone}.trislot_decoder.v{version}.fp16.param` | `mobilenet_v3_small.trislot_decoder.v2_0.fp16.param` |
| NCNN bin | `{backbone}.trislot_decoder.v{version}.fp16.bin` | `mobilenet_v3_small.trislot_decoder.v2_0.fp16.bin` |
