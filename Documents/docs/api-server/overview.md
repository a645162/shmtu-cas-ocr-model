# API 服务器概览

项目内置 PyTorch 推理 API 服务器，支持 V1 和 V2 模型，提供 HTTP 和 TCP 双协议访问。

## 支持模型

| 模型版本 | 说明 | 权重文件 |
|---|---|---|
| `v2` | TriSlot Decoder 单模型 | `best.pt` |
| `v1` | 三模型分离 | `workdir/Models/` 目录下 |

## 快速启动

```bash
# V2 模型 (推荐)
python -m cas_ocr_model.api_server --model-kind v2 --checkpoint runs/exp1/best.pt

# V1 模型
python -m cas_ocr_model.api_server --model-kind v1 --v1-model-dir workdir/Models

# 使用启动脚本
python scripts/api/run_api_server.py --model-kind v2 --checkpoint runs/exp1/best.pt
```

## 默认端口

| 协议 | 默认端口 |
|---|---|
| HTTP | 21600 |
| TCP | 21601 |

## 配置参数

```bash
python -m cas_ocr_model.api_server \
    --model-kind v2 \
    --checkpoint runs/exp1/best.pt \
    --host 0.0.0.0 \
    --http-port 21600 \
    --tcp-port 21601 \
    --device cuda \
    --workers 4 \
    --queue-size 100
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model-kind` | `v2` | 模型版本: `v1` / `v2` |
| `--checkpoint` | 自动搜索 | V2 模型路径 |
| `--v1-model-dir` | `workdir/Models` | V1 模型目录 |
| `--host` | `0.0.0.0` | 监听地址 |
| `--http-port` | `21600` | HTTP 端口 |
| `--tcp-port` | `21601` | TCP 端口 |
| `--device` | `cuda` | 推理设备: `cuda` / `cpu` |
| `--workers` | `4` | 推理线程数 |
| `--queue-size` | `100` | 请求队列大小 |

## 默认权重发现

- `v2` 默认在 `runs/`、`workdir/`、项目根目录中查找 `best.pt`
- `v1` 默认使用 `workdir/Models`

## V1/V2 响应兼容说明

- `v1` 返回原始 `equalSymbol` 与 6 类 `operator`
- `v2` 只预测标准化后的 `+` / `-` / `*`
  - `operator` 映射为兼容整数：`+ → 0`、`- → 2`、`* → 4`
  - `equalSymbol` 固定返回 `-1`

## 架构

```
客户端请求 (HTTP/TCP)
    │
    ▼
┌──────────────┐
│  API Server   │  app.py — HTTP + TCP 双协议监听
└──────────────┘
    │
    ▼
┌──────────────┐
│  OCR Service  │  service.py — 线程池队列调度
└──────────────┘
    │
    ▼
┌──────────────┐
│ Model Runner  │  model_runner.py — 模型加载与推理
└──────────────┘
    │
    ├── V2: CaptchaTriSlotDecoderCNN (单模型)
    └── V1: 3 × ResNet (三模型)
```
