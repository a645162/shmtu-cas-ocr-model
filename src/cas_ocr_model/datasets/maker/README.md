# datasets/maker — 验证码图片采集器

多进程 + 多会话 CAS 验证码图片 + json 标签采集器。

## 调用

```bash
# RESTful OCR (默认, 对齐 shmtu-ocr-server HTTP 21600)
python -m cas_ocr_model.datasets.maker \
    --backend restful --ocr-url http://127.0.0.1:21600 \
    --output ./dataset --count 5000 --processes 4 --per-process 8

# TCP OCR (对齐 shmtu-ocr-server TCP 21601)
python -m cas_ocr_model.datasets.maker \
    --backend tcp --ocr-host 127.0.0.1 --ocr-port 21601 \
    --output ./dataset --count 5000 --processes 4 --per-process 8

# 本地 PyTorch
python -m cas_ocr_model.datasets.maker \
    --backend pytorch --weights-dir ./weights \
    --output ./dataset --count 5000 --processes 2 --per-process 4
```

## 模块拆分

| 文件 | 职责 |
|---|---|
| `config.py`     | argparse + GitHub Release URL / USER_AGENT / 权重下载 |
| `ocr_backends.py` | Restful / Tcp / Pytorch 三种 OCR 后端 (async 协议) |
| `cas_client.py` | EpayAuth 三阶段 (probe→challenge→submit) + 落盘原子写 |
| `worker.py`     | 单 worker 入口 (asyncio 循环, 信号量并发控制) |
| `pool.py`       | 多进程 spawn Pool + 进度监控 + 断点续采 |
| `cli.py`        | 命令行入口 |

## 旧路径兼容

`python -m cas_ocr_model.datasets.dataset_collector` 改为 shim, 转发到 `maker.cli`, 旧命令行参数不变。

## 数据保存路径

| 项 | 路径 | 说明 |
|---|---|---|
| 图片 + json | `--output` 目录 (默认 `./dataset/`) | 8 位编号配对, 已 gitignore |
| PyTorch 权重 | `--weights-dir` (默认 `./weights/`) | 自动从 GitHub Release 下载, 已 gitignore |
| 临时文件 | `.{NNNNNNNN}.{jpg,json}.tmp` | 原子重命名, 失败不污染 |

## 断点续采

启动时扫描 `--output` 已有 8 位 jpg, 从 max+1 继续; 满足 `--count` 自动停止。

## 后续步骤

采集完后, 用 `cas_ocr_model.datasets.split` 切 train/val/test, 写 manifest.json:

```bash
python -m cas_ocr_model.datasets.split \
    --dataset-root ./dataset --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
```
