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

1. 并行导出 ONNX (fp16 + fp32) 与 NCNN (param + bin, fp16 + fp32)
2. 所有 release 资产完成后生成 `model-assets.json` 与 `SHA256SUMS.txt`
3. 上传为 Release Assets

### `model-assets.json` 结构

以 v2.0.2 release 为例，`model-assets.json` 包含以下顶层字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `schema_version` | int | 元数据 schema 版本，当前为 `1` |
| `generated_at_utc` | string | 生成时间 (UTC ISO 8601) |
| `models` | array | 模型级信息，每个模型一条 |
| `artifacts` | array | 资产平铺列表，按 engine × precision 展开 |
| `digests` | array | 校验文件摘要 (如 `SHA256SUMS.txt`) |

`models` 条目字段：

| 字段 | 说明 | 示例 |
|---|---|---|
| `version` | 模型版本 | `"2.0"` |
| `family` | 模型族 | `"trislot_decoder"` |
| `display_name` | 显示名称 | `"CAS OCR TriSlot Decoder"` |
| `backbone` | 当前 backbone | `"mobilenet_v3_small"` |
| `asset_stem` | 资产文件名前缀 | `"mobilenet_v3_small.trislot_decoder.v2_0"` |
| `supported_backbones` | 支持的 backbone 列表 | `["mobilenet_v3_large", "mobilenet_v3_small", ...]` |

`artifacts` 条目在 `models` 基础上增加：

| 字段 | 说明 | 示例 |
|---|---|---|
| `engine` | 推理引擎 | `"pytorch"` / `"onnx"` / `"ncnn"` |
| `precision` | 精度 | `"fp32"` / `"fp16"` |
| `format` | 文件格式 | `"checkpoint"` / `"onnx"` / `"ncnn"` |
| `files` | 文件列表 | `[{"path": ..., "sha256": ..., "release_asset_name": ...}]` |

### v2.0.2 Release 资产一览

| 引擎 | 精度 | 文件 | 大小 |
|---|---|---|---|
| PyTorch | fp32 | `mobilenet_v3_small.trislot_decoder.v2_0.pt` | 17.2 MB |
| ONNX | fp16 | `mobilenet_v3_small.trislot_decoder.v2_0.fp16.onnx` | 2.9 MB |
| ONNX | fp32 | `mobilenet_v3_small.trislot_decoder.v2_0.fp32.onnx` | 5.7 MB |
| NCNN | fp16 | `.fp16.param` + `.fp16.bin` | 13 KB + 2.8 MB |
| NCNN | fp32 | `.fp32.param` + `.fp32.bin` | 13 KB + 5.6 MB |
