# visualization scripts

- `vis.sh`: 测试集可视化入口，支持 `pytorch | onnx | ncnn`
- `visualize_test_predictions.py`: 可视化实现

示例:

```bash
# PyTorch
DEVICE=cpu N=20 bash scripts/visualization/vis.sh

# ONNX
BACKEND=onnx N=20 bash scripts/visualization/vis.sh

# ncnn
BACKEND=ncnn N=20 bash scripts/visualization/vis.sh
```
