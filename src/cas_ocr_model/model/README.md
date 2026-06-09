# 模型子包

存放所有验证码识别模型。现在已经引入**模型版本注册表**，配置中的 `model.version` 会绑定到具体实现。当前版本列表只有 `2.0`，后续可继续扩展而不破坏训练/推理/导出入口。

## 文件职责

| 文件 | 职责 |
|---|---|
| `backbones.py`        | backbone 工厂: `build_resnet_backbone(name, pretrained)`. 支持 torchvision 旧名、`r50`/`resnet101`、多种 MobileNetV3、`mobilenetv4_*` 与 `repvgg_*` 快捷别名，以及 `timm/<model_name>` 动态 backbone，统一接收 1 通道灰度图 |
| `heads.py`            | `TriSlotDecoder` 容器: 共享特征向量 → 3 个槽位输出 (digit_left / operator / digit_right) |
| `captcha_trislot_decoder_cnn.py` | 当前主模型: backbone + TriSlotDecoder, 一次前向输出 3 个 logits |
| `registry.py`         | 模型版本注册表、checkpoint 元信息、release 资产命名 |
| `cli.py`              | 打印版本列表 / checkpoint 元信息 |
| `versions/v2_0.py`    | 当前 `2.0` 版本的实现入口 |
| `__init__.py`         | 公共 API re-export |

## 公共 API

```python
from cas_ocr_model.model import (
    CaptchaTriSlotDecoderCNN,
    build_model_from_config,
    build_model_metadata,
    build_release_asset_stem,
    inspect_checkpoint,
    list_model_versions,
    load_checkpoint,
    predict_triple,
    build_resnet_backbone,
    list_available_backbones,
)
```

## 未来扩展点

新增版本时，优先在 `versions/` 下添加新实现，并在 `registry.py` 中注册；训练配置只需要切换 `model.version` 即可。

当前 release 资产命名规则:

- checkpoint: `{backbone}.trislot_decoder.v{version}.pt`
- onnx: `{backbone}.trislot_decoder.v{version}.fp16.onnx` / `.fp32.onnx`
- ncnn: `{backbone}.trislot_decoder.v{version}.fp16.param` / `.bin`

例如:

- `mobilenet_v3_small.trislot_decoder.v2_0.pt`
- `mobilenet_v3_small.trislot_decoder.v2_0.fp32.onnx`
- `mobilenet_v3_small.trislot_decoder.v2_0.fp16.param`

未来扩展时，把更复杂的模型作为新版本加入注册表即可，例如:

- `crnn_with_attention.py`     — CRNN + attention 头
- `transformer_triple.py`     — ViT 共享特征 + 多 head attention
- `multiscale_fpn.py`         — FPN 多尺度融合
- `capsule_captcha.py`        — 胶囊网络

模型只要实现 `forward(x) -> dict[str, Tensor]`, 与 `losses.TriSlotDecoderLoss` 对接, 就能在 `trainer/train.py` 即插即用 (只要构造时传对 head 维度)。

## 不变量

任何新模型必须满足:
1. 接收 `x: (B, 1, H, W) float32 ∈ [0, 1]`
2. `forward` 返回 dict, **至少** 含:
   - `digit_left_logits`:  (B, 10)
   - `operator_logits`:   (B, 3)
   - `digit_right_logits`: (B, 10)
3. `load_checkpoint(model, path)` 可以加载 (兼容 DDP `module.` 前缀)
