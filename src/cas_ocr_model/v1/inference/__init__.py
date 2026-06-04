"""推理层（懒加载 torch/cv2）。"""

__all__ = [
    "load_models",
    "predict_cv_image",
    "predict_validate_code",
    "get_default_transform",
    "start_server",
]

_LAZY = {}


def __getattr__(name):
    if name in ("load_models", "predict_cv_image", "predict_validate_code", "get_default_transform"):
        if "predictor" not in _LAZY:
            from . import predictor
            _LAZY["predictor"] = predictor
        return {
            "load_models": _LAZY["predictor"].load_models,
            "predict_cv_image": _LAZY["predictor"].predict_cv_image,
            "predict_validate_code": _LAZY["predictor"].predict_validate_code,
            "get_default_transform": _LAZY["predictor"].get_default_transform,
        }[name]
    if name == "start_server":
        if "server" not in _LAZY:
            from . import server
            _LAZY["server"] = server
        return _LAZY["server"].start_server
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
