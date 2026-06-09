# 推理子包

对外提供 3-head 验证码模型的 PyTorch / ONNX / ncnn 推理能力。

## 目录

```
inference/
├── __init__.py                       # 公共 API (懒加载后端)
├── inference.py                      # CaptchaInferencer 主类
├── preprocess.py                     # 灰度+二值化+resize (与 trainer 一致)
├── backends/
│   ├── __init__.py
│   ├── pytorch_backend.py            # 本地 PyTorch 推理
│   ├── onnx_backend.py               # 本地 ONNX Runtime 推理
│   └── ncnn_backend.py               # 本地 ncnn 推理
├── cli.py                            # 命令行入口
└── README.md
```

## 公共 API

```python
from cas_ocr_model.inference import CaptchaInferencer, InferencerConfig
from cas_ocr_model.inference import PyTorchBackend, OnnxBackend, NcnnBackend

# PyTorch
backend = PyTorchBackend("runs/exp1/best.pt", device="cuda")
inferencer = CaptchaInferencer(backend)
result = inferencer.predict_one("00000007.jpg")
print(result.expression, result.result, result.confidence)

# ONNX
backend = OnnxBackend("runs/exp1/export/onnx/best.fp32.onnx")
inferencer = CaptchaInferencer(backend)

# ncnn
backend = NcnnBackend(
    "runs/exp1/export/ncnn/best.fp32.param",
    "runs/exp1/export/ncnn/best.fp32.bin",
)
inferencer = CaptchaInferencer(backend)

# 批量目录
for name, r in inferencer.predict_dir("./dataset", limit=100):
    print(name, r.expression)
```

## CLI

```bash
# 单图
python -m cas_ocr_model.inference.cli \
    --backend pytorch \
    --checkpoint runs/exp1/best.pt \
    --image dataset/00000007.jpg

# ONNX
python -m cas_ocr_model.inference.cli \
    --backend onnx \
    --onnx runs/exp1/export/onnx/best.fp32.onnx \
    --image dataset/00000007.jpg

# ncnn
python -m cas_ocr_model.inference.cli \
    --backend ncnn \
    --ncnn-param runs/exp1/export/ncnn/best.fp32.param \
    --ncnn-bin runs/exp1/export/ncnn/best.fp32.bin \
    --image dataset/00000007.jpg

# 批量目录
python -m cas_ocr_model.inference.cli \
    --backend pytorch \
    --checkpoint runs/exp1/best.pt \
    --dir dataset --limit 50 --output preds.json
```

## backend 可用性

- `PyTorchBackend` 依赖 `torch`
- `OnnxBackend` 依赖 `onnxruntime`
- `NcnnBackend` 依赖 `ncnn`

导入 `cas_ocr_model.inference` 或 `cas_ocr_model.inference.backends` 时，不会因为缺少某个可选依赖而整体失败；只有在真正实例化对应 backend 时，才会提示该 backend 不可用。

## 结果结构

`InferenceResult` 字段:

| 字段 | 类型 | 说明 |
|---|---|---|
| `digit_left`  | `str`  | 第一个数字字符 ("0"-"9") |
| `operator`    | `str`  | 运算符 ("+", "-", "*") |
| `digit_right` | `str`  | 第二个数字字符 |
| `expression`  | `str`  | 拼接后表达式, 如 "1+2" |
| `result`      | `int`  | 算式求值结果, 不可计算时为 None |
| `confidence`  | `float`| 3 个 head 各自的 argmax softmax 的最小值 (越接近 1 越确定) |
| `softmax`     | `dict` | 每 head 的 10/3 维 softmax 概率分布 |

## 预处理一致性

`preprocess.py` 与 `trainer/data.py` 共享同样的二值化阈值 (`200`) 和 resize (默认 `64x192`), 保证训练/推理分布一致。如果训练时改过 `--threshold` / `--image-size-*`, 推理时需传同样参数。
