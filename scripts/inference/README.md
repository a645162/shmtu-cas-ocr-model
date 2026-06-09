# inference scripts

推理预测入口统一放在这里。

- `common.sh`: 共享 backend / run 路径解析
- `predict.sh`: 通用预测入口，配合 `BACKEND=pytorch|onnx|ncnn`
- `predict_pytorch.sh` / `predict_onnx.sh` / `predict_ncnn.sh`: 直接可运行的预测入口

默认路径约定：

- `PyTorch`: `runs/.../best.pt` 或规范命名的 `runs/.../{backbone}.trislot_decoder.v{version}.pt`
- `ONNX`: `runs/.../export/onnx/{backbone}.trislot_decoder.v{version}.fp32.onnx`
- `ncnn`: `runs/.../export/ncnn/{backbone}.trislot_decoder.v{version}.fp32.param` + `.bin`
