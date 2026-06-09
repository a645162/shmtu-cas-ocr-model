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
| `SHMTU_RESUME` | `0` | 设为 `1` 时从当前 profile/latest 对应 run 的 `last.pt` 续训 |
| `SHMTU_RESUME_FROM` | 空 | 显式指定训练续训 checkpoint, 优先级高于 `SHMTU_RESUME=1` |
| `SHMTU_AUTO_VIS` | `1` | 训练结束后自动运行 `vis.sh`，默认与手动执行一致，优先使用 `release/pytorch` 中的发布权重，否则回退到 `best.pt` |
| `VIS_CHECKPOINT` | `<unset>` | 可显式覆盖自动可视化使用的 checkpoint 文件名，例如 `best.pt` 或 `last.pt` |
| `SHMTU_DISABLE_WANDB` | `0` | 设为 `1/true/yes/on` 时禁用训练自动接入 wandb |
| `SHMTU_DYNAMO_BACKEND` | `inductor` | accelerate 的 dynamo backend; 可设为 `no` 回退到非 compile 路径 |
| `SHMTU_MIXED_PRECISION` | `bf16` | accelerate 的 mixed precision; 可设为 `fp16` / `bf16` / `no` |
| `MAX_FILES` | 空 | `split.sh` 最多使用多少个已配对样本; 会先随机选取再分割 |
| `CAS_OCR_WEIGHTS_DIR` | `$MODEL_ROOT/weights` | PyTorch 权重缓存 (gitignore) |
| `CAS_OCR_NUM_GPUS` | `8` | 训练 / 多卡 bench 用的 GPU 数 |
| `CAS_OCR_PYTHON` | `python3` | Python 解释器 |
| `PYTHONPATH` | 自动追加 `$SRC:$REPO/Lib/shmtu-cas-python/src` | 库路径 |

## 脚本索引

| 脚本 | 职责 | 典型用法 |
|---|---|---|
| `env.sh`                        | 公共环境变量 | `source scripts/env.sh` |
| `common/run_path.sh`            | 创建 / 解析 `runs/{profile}/{date_time}` 与 `latest` | `bash scripts/common/run_path.sh resolve` |
| `collection/download_weights.sh` | 仅下载 PyTorch 权重 (不采集) | `bash scripts/collection/download_weights.sh` |
| `collection/collect.sh`         | 启动 maker 采集 jpg+json | `bash scripts/collection/collect.sh` |
| `collection/collect_local_8gpu.sh` | 8 卡本地模型采集 | `bash scripts/collection/collect_local_8gpu.sh` |
| `data/split.sh`                 | 物理分割 train/val/test + 写 manifest | `bash scripts/data/split.sh` |
| `data/split_and_zip_dataset.sh` | 分割并打包数据集 | `bash scripts/data/split_and_zip_dataset.sh` |
| `training/train.sh`             | 8 卡 DDP 训练 (accelerate launch + fp16) | `bash scripts/training/train.sh` |
| `training/download_timm_backbone_weight.sh` | 预拉取 timm backbone 的 Hugging Face 预训练权重 | `bash scripts/training/download_timm_backbone_weight.sh` |
| `training/train_resume.sh`      | 自动续训当前 profile 的最后一个 run; 已完成则退出 | `bash scripts/training/train_resume.sh` |
| `training/train_new_or_resume.sh` | 无 latest 时新建训练, 未完成则续训, 已完成则退出 | `bash scripts/training/train_new_or_resume.sh` |
| `export/install_ncnn_tools.sh`  | 下载 ncnn 预编译工具并准备 `pnnx` / `ncnnoptimize` | `bash scripts/export/install_ncnn_tools.sh` |
| `export/export_onnx.sh`         | 默认同时导出 `fp16` + `fp32` ONNX | `bash scripts/export/export_onnx.sh` |
| `export/export_torchscript.sh`  | best.pt → traced TorchScript | `bash scripts/export/export_torchscript.sh` |
| `export/export_ncnn.sh`         | 默认同时导出 `fp16` + `fp32` ncnn | `bash scripts/export/export_ncnn.sh` |
| `export/export_all.sh`          | 一次导出 ONNX + ncnn；各子目录分别生成 SHA256SUMS | `bash scripts/export/export_all.sh` |
| `src/cas_ocr_model/export/release_bundle.py` | 批量生成 release 用的 PyTorch/ONNX/ncnn + `model-assets.json` | `python -m cas_ocr_model.export.release_bundle ...` |
| `export/verify_onnx_against_pytorch.py` | 校验 ONNX 与 PyTorch logits 是否一致 | `python scripts/export/verify_onnx_against_pytorch.py ...` |
| `export/verify_ncnn_against_pytorch.py` | 校验 ncnn 与 PyTorch logits 是否一致 | `python scripts/export/verify_ncnn_against_pytorch.py ...` |
| `inference/predict_pytorch.sh`  | 单图/目录预测，PyTorch | `bash scripts/inference/predict_pytorch.sh` |
| `inference/predict_onnx.sh`     | 单图/目录预测，ONNX | `bash scripts/inference/predict_onnx.sh` |
| `inference/predict_ncnn.sh`     | 单图/目录预测，ncnn | `bash scripts/inference/predict_ncnn.sh` |
| `evaluation/eval_accuracy.sh`   | maker 本地/服务端识别准确率评估 | `bash scripts/evaluation/eval_accuracy.sh` |
| `evaluation/evaluate_pytorch.sh` | 单卡 evaluate，PyTorch | `bash scripts/evaluation/evaluate_pytorch.sh` |
| `evaluation/evaluate_onnx.sh`   | 单卡 evaluate，ONNX | `bash scripts/evaluation/evaluate_onnx.sh` |
| `evaluation/evaluate_ncnn.sh`   | 单卡 evaluate，ncnn | `bash scripts/evaluation/evaluate_ncnn.sh` |
| `benchmark/bench_single_pytorch.sh` | 单卡速度 benchmark，PyTorch | `bash scripts/benchmark/bench_single_pytorch.sh` |
| `benchmark/bench_single_onnx.sh` | 单卡速度 benchmark，ONNX | `bash scripts/benchmark/bench_single_onnx.sh` |
| `benchmark/bench_single_ncnn.sh` | 单卡速度 benchmark，ncnn | `bash scripts/benchmark/bench_single_ncnn.sh` |
| `benchmark/bench_multi.sh`      | 多卡 DDP 分片推理 test 集并汇总指标，可选保存预测图片 | `bash scripts/benchmark/bench_multi.sh` |
| `visualization/vis.sh`          | 随机抽样 test 集并导出预测图 | `bash scripts/visualization/vis.sh` |
| `visualization/visualize_test_predictions.py` | 可视化实现脚本 | `python scripts/visualization/visualize_test_predictions.py --config src/cas_ocr_model/trainer/configs/8gpu_ddp.yaml` |
| `api/run_api_server.py`         | 启动 API server | `python scripts/api/run_api_server.py` |

## 一键式工作流

```bash
# 1) 准备 (只需跑一次)
cd /home/konghaomin/Prj/SHMTU/shmtu-terminal/Model/shmtu-cas-ocr-model
bash scripts/collection/download_weights.sh # 预热权重缓存
bash scripts/collection/collect.sh          # 采集数据集 (默认 5000 张)
bash scripts/data/split.sh                  # 切 train/val/test

# 2) 训练
bash scripts/training/train.sh              # 输出到 runs/{profile}/{YYYYMMDD_HHMMSS}, 并刷新 latest
SHMTU_RESUME=1 bash scripts/training/train.sh # 从当前 profile/latest/last.pt 续训, 继续写回原 run
bash scripts/training/train_resume.sh       # 自动检查 latest/last.pt; 未完成则续训, 已完成则退出
bash scripts/training/train_new_or_resume.sh # 无 latest 则新建训练, 否则自动判断续训/退出
bash scripts/export/export_onnx.sh          # 导出 ONNX
bash scripts/export/install_ncnn_tools.sh   # 下载 pnnx / ncnnoptimize
bash scripts/export/export_all.sh           # 基于 runs/.../release/pytorch 的 pt 导出到 release/onnx 和 release/ncnn

# 3) 评估 + benchmark
bash scripts/evaluation/evaluate_pytorch.sh  # 算 acc / ECE / 混淆矩阵
bash scripts/benchmark/bench_multi.sh        # 多卡 DDP 精度
bash scripts/benchmark/bench_single_pytorch.sh # 单卡速度 (QPS)
bash scripts/visualization/vis.sh            # 可视化 test 集预测
```

## 覆盖默认参数

```bash
# 例: 采集 10000 张, 用 8 进程 × 4 协程, TCP 后端
CAS_OCR_BACKEND=tcp COUNT=10000 PROCESSES=8 PER_PROCESS=4 \
    bash scripts/collection/collect.sh

# 例: 仅随机抽 5000 个已配对样本参与 train/val/test 分割
MAX_FILES=5000 SEED=42 \
    bash scripts/data/split.sh

# 例: 训练 4 卡 (单节点), 改 profile 名称
SHMTU_NUM_GPUS=4 SHMTU_PROFILE_NAME=exp_4gpu \
    bash scripts/training/train.sh

# 例: 训练后不自动生成测试集可视化
SHMTU_AUTO_VIS=0 \
    bash scripts/training/train.sh

# 例: 若 compile 路径不稳定, 临时关闭 dynamo
SHMTU_DYNAMO_BACKEND=no \
    bash scripts/training/train.sh

# 例: 若 fp16 梯度不稳定, 临时切到 bf16
SHMTU_MIXED_PRECISION=bf16 \
    bash scripts/training/train.sh

# 例: 导出指定 profile 的 latest
SHMTU_PROFILE_NAME=exp_4gpu \
    bash scripts/export/export_all.sh

# 例: 显式指定某个具体 run 目录
SHMTU_RUN_DIR=./runs/exp_4gpu/20260608_153000 \
    bash scripts/evaluation/evaluate_pytorch.sh

# 例: 从指定 checkpoint 续训, 继续写回该 checkpoint 所在 run 目录
SHMTU_RESUME_FROM=./runs/exp_4gpu/20260608_153000/last.pt \
    bash scripts/training/train.sh

# 例: 自动续训当前 profile 的最后一个 run
SHMTU_PROFILE_NAME=exp_4gpu \
    bash scripts/training/train_resume.sh

# 例: 自动选择新建或续训
SHMTU_PROFILE_NAME=exp_4gpu \
    bash scripts/training/train_new_or_resume.sh

# 例: 单卡速度 bench 用 CPU
DEVICE=cpu NUM_SAMPLES=200 \
    bash scripts/benchmark/bench_single_pytorch.sh
```

## 前提

```bash
# 仓库根一次性安装
pip install -e ./Lib/shmtu-cas-python
pip install -e ./Model/shmtu-cas-ocr-model
```

## 可视化脚本

```bash
bash scripts/visualization/vis.sh

CONFIG=src/cas_ocr_model/trainer/configs/8gpu_ddp.yaml \
SHMTU_RUN_DIR=./runs/8gpu_ddp/20260608_153000 \
OUTPUT_DIR=./runs/8gpu_ddp/20260608_153000/outputs \
N=20 \
DEVICE=cuda \
    bash scripts/visualization/vis.sh
```

输出目录下会生成一个子目录，里面包含：
- 按预测结果命名的采样图片，例如 `9-2=7.jpg`
- `predictions.json`
- `contact_sheet.jpg`，每张图下方展示 `GT` / `Pred` 公式，并用绿色 `CORRECT`、红色 `WRONG` 标注正误

## 数据保存路径 (全部 gitignored)

| 阶段 | 路径 |
|---|---|
| 采集 | `$CAS_OCR_DATASET_ROOT/` (默认 `./dataset/`) |
| PyTorch 权重 | `$CAS_OCR_WEIGHTS_DIR/` (默认 `./weights/`) |
| 训练输出 | `$SHMTU_RUNS_ROOT/$SHMTU_PROFILE_NAME/$date_time/` |
| latest 指针 | `$SHMTU_RUNS_ROOT/$SHMTU_PROFILE_NAME/latest` (文件内容为相对路径) |

训练 run 目录内会额外保存:
- `last.pt` / `best.pt`
- `results.csv`
- `metrics_history.json`
- `epochs/epoch_0001.json`, `epochs/epoch_0002.json`, ...

详见 `Model/shmtu-cas-ocr-model/.gitignore`。
