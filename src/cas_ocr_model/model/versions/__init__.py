"""模型版本实现入口."""

from .v2_0 import (
    MODEL_FAMILY,
    MODEL_VERSION,
)
from .v2_0 import (
    build_model as build_v2_0_model,
)

__all__ = [
    "MODEL_FAMILY",
    "MODEL_VERSION",
    "build_v2_0_model",
]
