# V2 模型导出

训练完成后，可将模型导出为 ONNX、NCNN、TorchScript 格式，用于不同部署场景。

## ONNX 导出

```bash
python -m cas_ocr_model.trainer.export \
    --checkpoint ./runs/exp1/best.pt \
    --output ./runs/exp1/model.onnx \
    --image-size-h 64 --image-size-w 192 \
    --dynamic-batch
```

导出规格：

| 项目 | 值 |
|---|---|
| 输入 | `(B, 1, 64, 192)` float32 ∈ [0, 1] |
| 输出 | `digit_left_logits` (B, 10), `operator_logits` (B, 3), `digit_right_logits` (B, 10) |
| Opset | 17 |
| 动态轴 | batch 维度可变 (启用 `--dynamic-batch` 时) |

## NCNN 导出

```bash
# 安装 ncnn 工具
bash scripts/export/install_ncnn_tools.sh

# 导出 ONNX → NCNN
bash scripts/export/export_ncnn.sh
```

或手动操作：

```bash
# 先导出 ONNX
python -m cas_ocr_model.trainer.export \
    --checkpoint ./runs/exp1/best.pt \
    --output ./runs/exp1/model.onnx

# onnx2ncnn 转换
onnx2ncnn ./runs/exp1/model.onnx ./runs/exp1/model.param ./runs/exp1/model.bin
```

## TorchScript 导出

```bash
python -m cas_ocr_model.trainer.export_torchscript \
    --checkpoint ./runs/exp1/best.pt \
    --output ./runs/exp1/model.pt
```

## 导出验证

```bash
# 验证 ONNX 与 PyTorch 输出一致性
python scripts/export/verify_onnx_against_pytorch.py \
    --checkpoint ./runs/exp1/best.pt \
    --onnx-path ./runs/exp1/model.onnx

# 验证 NCNN 与 PyTorch 输出一致性
python scripts/export/verify_ncnn_against_pytorch.py \
    --checkpoint ./runs/exp1/best.pt \
    --ncnn-param ./runs/exp1/model.param \
    --ncnn-bin ./runs/exp1/model.bin
```

## 一键导出所有格式

```bash
bash scripts/export/export_all.sh
```

## Release 自动发布

项目配置了 GitHub Actions workflow (`.github/workflows/release-model-assets.yml`)，当创建 GitHub Release 时自动：

1. 并行导出 ONNX (fp16 + fp32) 与 NCNN (param + bin)
2. 从 PyTorch checkpoint (`.pt`) 中还原 `pip list --format=json`，生成对应的 `*.pip-list.json`
3. 所有 release 资产完成后生成根目录 `SHA256SUMS.txt`
4. 上传为 Release Assets

生成的 `model-assets.json` 支持多模型发布：顶层 `modellist` 列出所有模型，`models` 中按模型聚合各自的 `pytorch/onnx/ncnn` 资产，`artifacts` 保留平铺结构用于兼容旧脚本。
