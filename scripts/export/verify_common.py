#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from cas_ocr_model.common.preprocess import decode_color_image, preprocess_captcha_to_tensor
from cas_ocr_model.trainer.model import build_model_from_checkpoint


HEAD_NAMES = ("digit_left_logits", "operator_logits", "digit_right_logits")
OPERATOR_LABELS = ("+", "-", "*")


def add_common_args(parser) -> None:
    parser.add_argument("--checkpoint", required=True, help="PyTorch checkpoint, 例如 best.pt")
    parser.add_argument("--image-size-h", type=int, default=64)
    parser.add_argument("--image-size-w", type=int, default=192)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--input-npy", default=None, help="输入 .npy, shape 支持 (1,H,W) 或 (B,1,H,W)")
    parser.add_argument("--image", default=None, help="输入验证码图片路径, 会走训练同款预处理")
    parser.add_argument("--batch-size", type=int, default=1, help="随机输入模式下使用")
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("指定了 --device cuda, 但当前 CUDA 不可用")
    return torch.device(device_name)


def load_input_array(args) -> np.ndarray:
    if args.input_npy and args.image:
        raise SystemExit("--input-npy 和 --image 只能二选一")

    if args.input_npy:
        array = np.load(args.input_npy)
    elif args.image:
        image = decode_color_image(args.image)
        tensor = preprocess_captcha_to_tensor(
            image,
            image_size=(args.image_size_h, args.image_size_w),
        )
        array = tensor.unsqueeze(0).numpy()
    else:
        rng = np.random.default_rng(args.seed)
        array = rng.random(
            (args.batch_size, 1, args.image_size_h, args.image_size_w),
            dtype=np.float32,
        )

    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 3:
        array = array[None, ...]
    if array.ndim != 4 or array.shape[1] != 1:
        raise SystemExit(
            f"输入 shape 必须是 (B,1,H,W) 或 (1,H,W), 当前是 {array.shape}"
        )
    return np.ascontiguousarray(array)


def run_pytorch(checkpoint: str, input_array: np.ndarray, device_name: str) -> dict[str, np.ndarray]:
    device = resolve_device(device_name)
    model = build_model_from_checkpoint(checkpoint, device=device)
    model.eval()
    with torch.inference_mode():
        outputs = model(torch.from_numpy(input_array).to(device))
    return {
        name: outputs[name].detach().cpu().numpy().astype(np.float32, copy=False)
        for name in HEAD_NAMES
    }


def describe_predictions(outputs: dict[str, np.ndarray]) -> list[str]:
    digit_left = outputs["digit_left_logits"].argmax(axis=-1)
    operator = outputs["operator_logits"].argmax(axis=-1)
    digit_right = outputs["digit_right_logits"].argmax(axis=-1)

    expressions: list[str] = []
    for dl, op, dr in zip(digit_left.tolist(), operator.tolist(), digit_right.tolist()):
        expressions.append(f"{dl}{OPERATOR_LABELS[op]}{dr}")
    return expressions


def compare_outputs(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    atol: float,
    rtol: float,
) -> tuple[bool, list[str]]:
    lines: list[str] = []
    passed = True
    for name in HEAD_NAMES:
        ref = reference[name].astype(np.float32, copy=False)
        cur = candidate[name].astype(np.float32, copy=False)
        if ref.shape != cur.shape:
            passed = False
            lines.append(f"[compare] {name}: shape mismatch {ref.shape} vs {cur.shape}")
            continue
        diff = np.abs(ref - cur)
        max_abs = float(diff.max(initial=0.0))
        mean_abs = float(diff.mean()) if diff.size else 0.0
        same = bool(np.allclose(ref, cur, atol=atol, rtol=rtol))
        passed = passed and same
        lines.append(
            f"[compare] {name}: allclose={same} max_abs={max_abs:.6g} mean_abs={mean_abs:.6g}"
        )
    return passed, lines
