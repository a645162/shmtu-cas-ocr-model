# export scripts

用于模型导出与格式转换:

- `install_ncnn_tools.sh`: 优先下载官方预编译 `shared` 版 `ncnn` 工具，并为 `pnnx` / `ncnnoptimize` 加执行位
- `export_onnx.sh`: 默认同时导出 `fp16` 和 `fp32` 两份 `.onnx`
- `export_torchscript.sh`: `best.pt -> traced .ts.pt`
- `export_ncnn.sh`: 默认同时导出 `fp16` 和 `fp32` 两份 `.param/.bin`，也支持回退到 `TorchScript -> pnnx`
- `export_ncnn_python.py`: `best.pt -> pnnx.export(...) -> .pt/.param/.bin`
- `verify_onnx_against_pytorch.py`: 对比 ONNX 与 PyTorch 直接推理的 logits
- `verify_ncnn_against_pytorch.py`: 对比 ncnn 与 PyTorch 直接推理的 logits
- `generate_release_digest.sh`: 生成 GitHub Release 用的 `SHA256SUMS.txt`
- `export_all.sh`: 同时导出 ONNX 和 ncnn, 并刷新 `SHA256SUMS.txt`

## 常用环境变量

```bash
SHMTU_PROFILE_NAME=8gpu_ddp
SHMTU_RUN_DIR=./runs/8gpu_ddp/20260608_153000
CHECKPOINT=./runs/8gpu_ddp/20260608_153000/best.pt
EXPORT_ROOT=./runs/8gpu_ddp/20260608_153000/export
MODEL_NAME=best
EXPORT_PRECISIONS="fp16 fp32"
EXPORT_DEVICE=auto
IMAGE_SIZE_H=64
IMAGE_SIZE_W=192
OPSET=17
LEGACY_EXPORTER=1
PNNX=/abs/path/to/pnnx
NCNNOPTIMIZE=/abs/path/to/ncnnoptimize
EXPORT_NCNN_MODE=python
RUN_NCNNOPTIMIZE=0
```

默认目录结构:

```text
runs/.../export/
  SHA256SUMS.txt
  onnx/
    best.fp16.onnx
    best.fp32.onnx
  ncnn/
    best.fp16.pt
    best.fp16.param
    best.fp16.bin
    best.fp16.opt.param
    best.fp16.opt.bin
    best.fp32.pt
    best.fp32.param
    best.fp32.bin
    best.fp32.opt.param
    best.fp32.opt.bin
  torchscript/
    best.fp16.ts.pt   # 仅独立运行 export_torchscript.sh 时默认放这里
```

## 示例

```bash
bash scripts/export/install_ncnn_tools.sh

# 首次安装会优先选 shared 发行包；如果预编译包不含 pnnx，会自动下载 full-source 并编译 pnnx

# 默认解析 runs/{profile}/latest
bash scripts/export/export_onnx.sh

# 或显式指定某个 run
SHMTU_RUN_DIR=./runs/8gpu_ddp/20260608_153000 \
    bash scripts/export/export_ncnn.sh

# 只导出单个精度
SHMTU_RUN_DIR=./runs/8gpu_ddp/20260608_153000 \
EXPORT_PRECISIONS=fp32 \
    bash scripts/export/export_all.sh

# 如需回退到旧的 TorchScript + pnnx 命令行路径
EXPORT_NCNN_MODE=cli \
    bash scripts/export/export_ncnn.sh

# 如需额外执行 ncnnoptimize
RUN_NCNNOPTIMIZE=1 \
    bash scripts/export/export_ncnn.sh

# 直接用 pnnx Python API
python scripts/export/export_ncnn_python.py \
    --checkpoint ./runs/8gpu_ddp/20260608_153000/best.pt \
    --output ./runs/8gpu_ddp/20260608_153000/export/ncnn/best.fp16.pt \
    --image-size-h 64 --image-size-w 192

bash scripts/export/export_all.sh

# 校验 ONNX / ncnn 是否与 PyTorch 直接推理一致
python scripts/export/verify_onnx_against_pytorch.py \
    --checkpoint ./runs/8gpu_ddp/20260608_153000/best.pt \
    --onnx ./runs/8gpu_ddp/20260608_153000/export/onnx/best.fp32.onnx

python scripts/export/verify_ncnn_against_pytorch.py \
    --checkpoint ./runs/8gpu_ddp/20260608_153000/best.pt \
    --param ./runs/8gpu_ddp/20260608_153000/export/ncnn/best.fp16.param \
    --bin ./runs/8gpu_ddp/20260608_153000/export/ncnn/best.fp16.bin
```
