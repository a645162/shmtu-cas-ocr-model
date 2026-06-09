# V2 评估与推理

## 评估

### 命令行评估

```bash
python -m cas_ocr_model.trainer.eval \
    --checkpoint ./runs/exp1/best.pt \
    --data-root ./dataset
```

输出指标：

| 指标 | 含义 |
|---|---|
| `acc_digit_left` | 左数字分类准确率 |
| `acc_operator` | 运算符分类准确率 |
| `acc_digit_right` | 右数字分类准确率 |
| `acc_expression` | 整体表达式准确率 (三个头同时正确) |

### 多卡评估

```bash
torchrun --nproc_per_node=4 \
    -m cas_ocr_model.trainer.eval \
    --checkpoint ./runs/exp1/best.pt \
    --data-root ./dataset
```

## 推理

### CLI 推理

```bash
# PyTorch 后端
python -m cas_ocr_model.inference \
    --checkpoint ./runs/exp1/best.pt \
    --image ./test.jpg

# ONNX 后端
python -m cas_ocr_model.inference \
    --backend onnx \
    --onnx-path ./runs/exp1/model.onnx \
    --image ./test.jpg

# NCNN 后端
python -m cas_ocr_model.inference \
    --backend ncnn \
    --ncnn-param ./runs/exp1/model.param \
    --ncnn-bin ./runs/exp1/model.bin \
    --image ./test.jpg
```

### Python API

```python
from cas_ocr_model.model import CaptchaTriSlotDecoderCNN, predict_triple, load_checkpoint

# 加载模型
model = CaptchaTriSlotDecoderCNN(backbone="mobilenet_v3_small", pretrained=False)
model = load_checkpoint(model, "./runs/exp1/best.pt", device="cuda")
model.eval()

# 推理
import torch
from cas_ocr_model.inference.preprocess import preprocess_image

image_tensor = preprocess_image("./test.jpg", image_size_h=64, image_size_w=192)
image_tensor = image_tensor.unsqueeze(0).to("cuda")  # 添加 batch 维度

results = predict_triple(
    model, image_tensor,
    operator_labels=["+", "-", "*"],
    digit_labels=[str(i) for i in range(10)],
)
print(results[0]["expression"])  # 例: "9-2"
```

## 性能基准

```bash
# 单 GPU 基准
bash scripts/benchmark/bench_single.sh

# 多 GPU 基准
bash scripts/benchmark/bench_multi.sh
```

支持的后端及典型性能对比：

| 后端 | 延迟 (ms/img) | 适用场景 |
|---|---|---|
| PyTorch (GPU) | ~2 | 研发调试 |
| ONNX Runtime (GPU) | ~1 | 服务端推理 |
| NCNN (CPU) | ~5 | 边缘部署 |
| NCNN (Vulkan) | ~3 | 移动端 GPU |
