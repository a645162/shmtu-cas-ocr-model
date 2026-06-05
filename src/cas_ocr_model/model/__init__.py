"""模型子包.

存放所有的验证码识别 CNN 模型, 当前实现是简单版 (``captcha_triple_head_cnn``),
未来可加入更复杂的 (注意力 / Transformer / 多尺度融合) 而不影响训练入口.

模块:
    backbones                         - backbone 工厂 (ResNet-18/34 1-通道)
    heads                             - 3-head 分类器容器
    captcha_triple_head_cnn           - 当前模型: backbone + 3-head, 一次前向 3 logits

公共 API:
    CaptchaTripleHeadCNN              - 模型类
    load_checkpoint                   - 加载权重 (兼容 DDP 'module.' 前缀)
    predict_triple                    - 批量 argmax -> 字符串表达式
"""
from .backbones import build_resnet_backbone, list_available_backbones
from .captcha_triple_head_cnn import (
    CaptchaTripleHeadCNN,
    build_model_from_checkpoint,
    load_checkpoint,
    predict_triple,
)
from .heads import TripleHead

__all__ = [
    # 当前模型
    "CaptchaTripleHeadCNN",
    "build_model_from_checkpoint",
    "load_checkpoint",
    "predict_triple",
    # backbone / head 工具
    "build_resnet_backbone",
    "list_available_backbones",
    "TripleHead",
]
