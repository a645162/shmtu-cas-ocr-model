# 模型子包

存放所有验证码识别模型。当前是简单版 (`CaptchaTripleHeadCNN` = ResNet-18/34 + 3 独立 head), 未来会加入更复杂的实现而不影响训练入口。

## 文件职责

| 文件 | 职责 |
|---|---|
| `backbones.py`        | backbone 工厂: `build_resnet_backbone(name, pretrained)`. 当前支持 `resnet18` / `resnet34`, 都已改成接收 1 通道灰度图 |
| `heads.py`            | `TripleHead` 容器: 共享特征向量 → 3 个独立分类头 (digit_left / operator / digit_right) |
| `captcha_triple_head_cnn.py` | 当前主模型: backbone + TripleHead, 一次前向输出 3 个 logits |
| `__init__.py`         | 公共 API re-export |

## 公共 API

```python
from cas_ocr_model.model import (
    CaptchaTripleHeadCNN,
    load_checkpoint,
    predict_triple,
    build_resnet_backbone,
    list_available_backbones,
)
```

## 未来扩展点

把更复杂的模型作为新文件加到本目录, 然后在 `__init__.py` re-export 即可:

- `crnn_with_attention.py`     — CRNN + attention 头
- `transformer_triple.py`     — ViT 共享特征 + 多 head attention
- `multiscale_fpn.py`         — FPN 多尺度融合
- `capsule_captcha.py`        — 胶囊网络

模型只要实现 `forward(x) -> dict[str, Tensor]`, 与 `losses.TripleHeadLoss` 对接, 就能在 `trainer/train.py` 即插即用 (只要构造时传对 head 维度)。

## 不变量

任何新模型必须满足:
1. 接收 `x: (B, 1, H, W) float32 ∈ [0, 1]`
2. `forward` 返回 dict, **至少** 含:
   - `digit_left_logits`:  (B, 10)
   - `operator_logits`:   (B, 4)
   - `digit_right_logits`: (B, 10)
3. `load_checkpoint(model, path)` 可以加载 (兼容 DDP `module.` 前缀)
