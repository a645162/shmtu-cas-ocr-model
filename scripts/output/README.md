# output scripts

用于模型导出与格式转换:

- `install_ncnn_tools.sh`: 下载官方预编译 `ncnn` 工具并为 `pnnx` / `ncnnoptimize` 加执行位
- `export_onnx.sh`: `best.pt -> .onnx`
- `export_torchscript.sh`: `best.pt -> traced .ts.pt`
- `export_ncnn.sh`: `best.pt -> TorchScript -> pnnx -> .ncnn.param/.bin`
- `export_all.sh`: 同时导出 ONNX 和 ncnn

## 常用环境变量

```bash
CHECKPOINT=./runs/8gpu_ddp/best.pt
EXPORT_DIR=./runs/8gpu_ddp/export
MODEL_NAME=best
IMAGE_SIZE_H=64
IMAGE_SIZE_W=192
OPSET=17
LEGACY_EXPORTER=1
PNNX=/abs/path/to/pnnx
NCNNOPTIMIZE=/abs/path/to/ncnnoptimize
```

## 示例

```bash
bash scripts/output/install_ncnn_tools.sh

# 首次安装如果预编译包不含 pnnx，会自动下载 full-source 并编译 pnnx

bash scripts/output/export_onnx.sh

bash scripts/output/export_ncnn.sh

bash scripts/output/export_all.sh
```
