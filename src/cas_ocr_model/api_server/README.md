# PyTorch API Server

这个子包直接复用当前仓库里的 PyTorch 推理实现，对外提供 OCR API 服务。

支持模型：

- `v2`: 新版 TriSlot Decoder `best.pt`
- `v1`: 旧版三模型 `.pth`

支持协议：

- HTTP:
  - `GET /api/health`
  - `GET /api/status`
  - `POST /api/ocr`
  - `POST /api/ocr/upload`
- TCP:
  - 图片二进制 + `<END>`
  - 返回表达式字符串

## 启动

```bash
python -m cas_ocr_model.api_server --model-kind v2 --checkpoint runs/exp1/best.pt
python -m cas_ocr_model.api_server --model-kind v1 --v1-model-dir workdir/Models
python scripts/api/run_api_server.py --model-kind v2 --checkpoint runs/exp1/best.pt
```

## 默认权重发现

- `v2` 默认在 `runs/`、`workdir/`、项目根目录中查找 `best.pt`
- `v1` 默认使用 `workdir/Models`

## 响应兼容说明

- `v1` 返回原始 `equalSymbol` 与 6 类 `operator`
- `v2` 只预测标准化后的 `+ / - / *`
  - `operator` 映射为兼容整数：`+ -> 0`、`- -> 2`、`* -> 4`
  - `equalSymbol` 固定返回 `-1`
