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
| `SHMTU_RUNS_ROOT` | `$MODEL_ROOT/runs` | runs 根目录 (gitignore) |
| `SHMTU_PROFILE_NAME` | `8gpu_ddp` | 当前实验 profile 名称 |
| `SHMTU_RUN_DIR` | 空 | 显式指定某个具体 run 目录; 不设时默认解析 profile 下 `latest` |
| `CAS_OCR_WEIGHTS_DIR` | `$MODEL_ROOT/weights` | PyTorch 权重缓存 (gitignore) |
| `CAS_OCR_NUM_GPUS` | `8` | 训练 / 多卡 bench 用的 GPU 数 |
| `CAS_OCR_PYTHON` | `python3` | Python 解释器 |
| `PYTHONPATH` | 自动追加 `$SRC:$REPO/Lib/shmtu-cas-python/src` | 库路径 |

## 脚本索引

| 脚本 | 职责 | 典型用法 |
|---|---|---|
| `env.sh`                        | 公共环境变量 | `source scripts/env.sh` |
| `run_path.sh`                   | 创建 / 解析 `runs/{profile}/{date_time}` 与 `latest` | `bash scripts/run_path.sh resolve` |
| `download_weights.sh`           | 仅下载 PyTorch 权重 (不采集) | `bash scripts/download_weights.sh` |
| `collect.sh`                    | 启动 maker 采集 jpg+json | `bash scripts/collect.sh` |
| `split.sh`                      | 物理分割 train/val/test + 写 manifest | `bash scripts/split.sh` |
| `train.sh`                      | 8 卡 DDP 训练 (accelerate launch + fp16) | `bash scripts/train.sh` |
| `export.sh`                     | best.pt → model.onnx (仅导出脚本使用 ONNX) | `bash scripts/export.sh` |
| `output/install_ncnn_tools.sh`  | 下载 ncnn 预编译工具并准备 `pnnx` / `ncnnoptimize` | `bash scripts/output/install_ncnn_tools.sh` |
| `output/export_onnx.sh`         | best.pt → model.onnx (稳定模式默认走 legacy exporter) | `bash scripts/output/export_onnx.sh` |
| `output/export_torchscript.sh`  | best.pt → traced TorchScript | `bash scripts/output/export_torchscript.sh` |
| `output/export_ncnn.sh`         | best.pt → TorchScript → pnnx → ncnn | `bash scripts/output/export_ncnn.sh` |
| `output/export_all.sh`          | 一次导出 ONNX + ncnn | `bash scripts/output/export_all.sh` |
| `evaluate.sh`                   | 单卡 evaluate (test 集) | `bash scripts/evaluate.sh` |
| `bench_multi.sh`                | 多卡 DDP 精度 benchmark | `bash scripts/bench_multi.sh` |
| `bench_single.sh`               | 单卡速度 benchmark | `bash scripts/bench_single.sh` |
| `vis.sh`                        | 随机抽样 test 集并导出预测图 | `bash scripts/vis.sh` |
| `visualize_test_predictions.py` | 可视化实现脚本 | `python scripts/visualize_test_predictions.py --config src/cas_ocr_model/trainer/configs/8gpu_ddp.yaml` |

## 一键式工作流

```bash
# 1) 准备 (只需跑一次)
cd /home/konghaomin/Prj/SHMTU/shmtu-terminal/Model/shmtu-cas-ocr-model
bash scripts/download_weights.sh            # 预热权重缓存
bash scripts/collect.sh                     # 采集数据集 (默认 5000 张)
bash scripts/split.sh                       # 切 train/val/test

# 2) 训练
bash scripts/train.sh                       # 输出到 runs/{profile}/{YYYYMMDD_HHMMSS}, 并刷新 latest
bash scripts/export.sh                      # 导出 ONNX
bash scripts/output/install_ncnn_tools.sh   # 下载 pnnx / ncnnoptimize
bash scripts/output/export_all.sh           # 导出 ONNX + ncnn (pnnx)

# 3) 评估 + benchmark
bash scripts/evaluate.sh                    # 算 acc / ECE / 混淆矩阵
bash scripts/bench_multi.sh                 # 多卡 DDP 精度
bash scripts/bench_single.sh                # 单卡速度 (QPS)
bash scripts/vis.sh                         # 可视化 test 集预测
```

## 覆盖默认参数

```bash
# 例: 采集 10000 张, 用 8 进程 × 4 协程, TCP 后端
CAS_OCR_BACKEND=tcp COUNT=10000 PROCESSES=8 PER_PROCESS=4 \
    bash scripts/collect.sh

# 例: 训练 4 卡 (单节点), 改 profile 名称
SHMTU_NUM_GPUS=4 SHMTU_PROFILE_NAME=exp_4gpu \
    bash scripts/train.sh

# 例: 导出指定 profile 的 latest
SHMTU_PROFILE_NAME=exp_4gpu \
    bash scripts/output/export_all.sh

# 例: 显式指定某个具体 run 目录
SHMTU_RUN_DIR=./runs/exp_4gpu/20260608_153000 \
    bash scripts/evaluate.sh

# 例: 单卡速度 bench 用 CPU
DEVICE=cpu NUM_SAMPLES=200 \
    bash scripts/bench_single.sh
```

## 前提

```bash
# 仓库根一次性安装
pip install -e ./Lib/shmtu-cas-python
pip install -e ./Model/shmtu-cas-ocr-model
```

## 可视化脚本

```bash
bash scripts/vis.sh

CONFIG=src/cas_ocr_model/trainer/configs/8gpu_ddp.yaml \
SHMTU_RUN_DIR=./runs/8gpu_ddp/20260608_153000 \
OUTPUT_DIR=./runs/8gpu_ddp/20260608_153000/outputs \
N=20 \
DEVICE=cuda \
    bash scripts/vis.sh
```

输出目录下会生成一个子目录，里面包含：
- 按预测结果命名的采样图片，例如 `9-2=7.jpg`
- `predictions.json`
- `contact_sheet.jpg`

## 数据保存路径 (全部 gitignored)

| 阶段 | 路径 |
|---|---|
| 采集 | `$CAS_OCR_DATASET_ROOT/` (默认 `./dataset/`) |
| PyTorch 权重 | `$CAS_OCR_WEIGHTS_DIR/` (默认 `./weights/`) |
| 训练输出 | `$SHMTU_RUNS_ROOT/$SHMTU_PROFILE_NAME/$date_time/` |
| latest 指针 | `$SHMTU_RUNS_ROOT/$SHMTU_PROFILE_NAME/latest` (文件内容为相对路径) |

详见 `Model/shmtu-cas-ocr-model/.gitignore`。
