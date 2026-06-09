# inference scripts

推理预测入口统一放在这里。

- `common.sh`: 共享 backend / run 路径解析
- `predict.sh`: 通用预测入口，配合 `BACKEND=pytorch|onnx|ncnn`
- `predict_pytorch.sh` / `predict_onnx.sh` / `predict_ncnn.sh`: 直接可运行的预测入口

默认路径约定：

- `PyTorch`: `runs/.../best.pt`
- `ONNX`: `runs/.../export/onnx/best.fp32.onnx`
- `ncnn`: `runs/.../export/ncnn/best.fp32.param` + `best.fp32.bin`
