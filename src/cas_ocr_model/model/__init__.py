"""模型子包.

存放所有的验证码识别 CNN 模型, 当前实现是简单版 (``captcha_trislot_decoder_cnn``),
未来可加入更复杂的 (注意力 / Transformer / 多尺度融合) 而不影响训练入口.

模块:
    backbones                         - backbone 工厂 (torchvision + timm, 统一 1-通道输入)
    heads                             - TriSlot Decoder 容器
    captcha_trislot_decoder_cnn       - 当前模型: backbone + TriSlot Decoder, 一次前向 3 logits

公共 API:
    CaptchaTriSlotDecoderCNN          - 模型类
    load_checkpoint                   - 加载权重 (兼容 DDP 'module.' 前缀)
    predict_triple                    - 批量 argmax -> 字符串表达式
"""
from .backbones import build_resnet_backbone, list_available_backbones
from .captcha_trislot_decoder_cnn import (
    CaptchaTriSlotDecoderCNN,
    build_model_from_checkpoint,
    load_checkpoint,
    predict_triple,
)
from .registry import (
    build_model_from_config,
    build_model_metadata,
    build_release_asset_stem,
    find_release_checkpoint,
    get_model_spec,
    inspect_checkpoint,
    list_model_versions,
    normalize_model_version,
)
from .heads import TriSlotDecoder
from .stats import (
    ModelStats,
    collect_model_stats,
    format_flops,
    format_model_stats,
    format_params_m,
)

__all__ = [
    # 当前模型
    "CaptchaTriSlotDecoderCNN",
    "build_model_from_checkpoint",
    "build_model_from_config",
    "build_model_metadata",
    "build_release_asset_stem",
    "find_release_checkpoint",
    "load_checkpoint",
    "predict_triple",
    "get_model_spec",
    "inspect_checkpoint",
    "list_model_versions",
    "normalize_model_version",
    # backbone / head 工具
    "build_resnet_backbone",
    "list_available_backbones",
    "TriSlotDecoder",
    "ModelStats",
    "collect_model_stats",
    "format_flops",
    "format_model_stats",
    "format_params_m",
]
