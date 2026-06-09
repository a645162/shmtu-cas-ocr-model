# benchmark scripts

- `bench_single.sh`: 通用单机 benchmark 入口，配合 `BACKEND=pytorch|onnx|ncnn`
- `bench_single_pytorch.sh`: 单机 benchmark，PyTorch
- `bench_single_onnx.sh`: 单机 benchmark，ONNX
- `bench_single_ncnn.sh`: 单机 benchmark，ncnn
- `bench_multi.sh`: 多卡 DDP 精度 benchmark，仅 PyTorch
