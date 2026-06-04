# cas_ocr_model scripts

一键式 shell 脚本, 串通采集 → 分割 → 训练 → 导出 → 评估 → benchmark 全流程。

## 公共环境

所有脚本通过 `source scripts/env.sh` 加载公共变量, 可在外部 export 覆盖:

| 变量 | 默认值 | 说明 |
|---|---|---|
| `CAS_OCR_BACKEND` | `restful` | 采集阶段 OCR 后端 (restful/tcp/pytorch) |
| `CAS_OCR_RESTFUL_URL` | `http://127.0.0.1:21600` | RESTful OCR base url (对齐 shmtu-ocr-server HTTP 21600) |
| `CAS_OCR_TCP_HOST` / `CAS_OCR_TCP_PORT` | `127.0.0.1` / `21601` | TCP OCR (对齐 shmtu-ocr-server TCP 21601) |
| `CAS_OCR_DATASET_ROOT` | `$MODEL_ROOT/dataset` | 数据集根目录 (gitignore) |
| `CAS_OCR_RUN_DIR` | `$MODEL_ROOT/runs/8gpu_ddp` | 训练输出目录 (gitignore) |
| `CAS_OCR_WEIGHTS_DIR` | `$MODEL_ROOT/weights` | PyTorch 权重缓存 (gitignore) |
| `CAS_OCR_NUM_GPUS` | `8` | 训练 / 多卡 bench 用的 GPU 数 |
| `CAS_OCR_PYTHON` | `python3` | Python 解释器 |
| `PYTHONPATH` | 自动追加 `$SRC:$REPO/Lib/shmtu-cas-python/src` | 库路径 |

## 脚本索引

| 脚本 | 职责 | 典型用法 |
|---|---|---|
| `env.sh`                            | 公共环境变量 | `source scripts/env.sh` |
| `cas_ocr_download_weights.sh`       | 仅下载 PyTorch 权重 (不采集) | `bash scripts/cas_ocr_download_weights.sh` |
| `cas_ocr_collect.sh`                | 启动 maker 采集 jpg+json | `bash scripts/cas_ocr_collect.sh` |
| `cas_ocr_split.sh`                  | 物理分割 train/val/test + 写 manifest | `bash scripts/cas_ocr_split.sh` |
| `cas_ocr_train.sh`                  | 8 卡 DDP 训练 (accelerate launch + fp16) | `bash scripts/cas_ocr_train.sh` |
| `cas_ocr_export.sh`                 | best.pt → model.onnx | `bash scripts/cas_ocr_export.sh` |
| `cas_ocr_evaluate.sh`               | 单卡 evaluate (test 集) | `bash scripts/cas_ocr_evaluate.sh` |
| `cas_ocr_bench_multi.sh`            | 多卡 DDP 精度 benchmark | `bash scripts/cas_ocr_bench_multi.sh` |
| `cas_ocr_bench_single.sh`           | 单卡速度 benchmark | `bash scripts/cas_ocr_bench_single.sh` |

## 一键式工作流

```bash
# 1) 准备 (只需跑一次)
cd /home/konghaomin/Prj/SHMTU/shmtu-terminal/Model/shmtu-cas-ocr-model
bash scripts/cas_ocr_download_weights.sh            # 预热权重缓存
bash scripts/cas_ocr_collect.sh                      # 采集数据集 (默认 5000 张)
bash scripts/cas_ocr_split.sh                        # 切 train/val/test

# 2) 训练
bash scripts/cas_ocr_train.sh                        # 8 卡 DDP, fp16, 30 epoch
bash scripts/cas_ocr_export.sh                       # 导出 ONNX

# 3) 评估 + benchmark
bash scripts/cas_ocr_evaluate.sh                     # 算 acc / ECE / 混淆矩阵
bash scripts/cas_ocr_bench_multi.sh                  # 多卡 DDP 精度
bash scripts/cas_ocr_bench_single.sh                 # 单卡速度 (QPS)
```

## 覆盖默认参数

```bash
# 例: 采集 10000 张, 用 8 进程 × 4 协程, TCP 后端
CAS_OCR_BACKEND=tcp COUNT=10000 PROCESSES=8 PER_PROCESS=4 \
    bash scripts/cas_ocr_collect.sh

# 例: 训练 4 卡 (单节点), 改输出目录
CAS_OCR_NUM_GPUS=4 CAS_OCR_RUN_DIR=./runs/exp_4gpu \
    bash scripts/cas_ocr_train.sh

# 例: 单卡速度 bench 用 CPU
DEVICE=cpu NUM_SAMPLES=200 \
    bash scripts/cas_ocr_bench_single.sh
```

## 前提

```bash
# 仓库根一次性安装
pip install -e ./Lib/shmtu-cas-python
pip install -e ./Model/shmtu-cas-ocr-model
# 或在 Model 目录:
pip install -r src/cas_ocr_model/trainer/requirements.txt
```

## 数据保存路径 (全部 gitignored)

| 阶段 | 路径 |
|---|---|
| 采集 | `$CAS_OCR_DATASET_ROOT/` (默认 `./dataset/`) |
| PyTorch 权重 | `$CAS_OCR_WEIGHTS_DIR/` (默认 `./weights/`) |
| 训练输出 | `$CAS_OCR_RUN_DIR/` (默认 `./runs/8gpu_ddp/`) |

详见 `Model/shmtu-cas-ocr-model/.gitignore`。
