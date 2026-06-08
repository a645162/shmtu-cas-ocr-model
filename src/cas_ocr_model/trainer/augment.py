from __future__ import annotations

import math
import random

import cv2
import numpy as np

from .config import AugmentationConfig


def sample_binarize_params(
    cfg: AugmentationConfig,
    base_mode: str,
    base_threshold: int,
    base_adaptive_c: int,
    rng: random.Random,
) -> tuple[str, int, int]:
    mode = base_mode
    threshold = base_threshold
    adaptive_c = base_adaptive_c

    if cfg.binarize_jitter_enabled and cfg.binarize_jitter_prob > 0 and rng.random() < cfg.binarize_jitter_prob:
        if cfg.alt_binarize_modes:
            mode = rng.choice(cfg.alt_binarize_modes)
        if cfg.threshold_jitter > 0:
            threshold += rng.randint(-cfg.threshold_jitter, cfg.threshold_jitter)
        if cfg.adaptive_c_jitter > 0:
            adaptive_c += rng.randint(-cfg.adaptive_c_jitter, cfg.adaptive_c_jitter)

    threshold = max(0, min(255, int(threshold)))
    adaptive_c = int(adaptive_c)
    return mode, threshold, adaptive_c


def augment_binary_image(binary: np.ndarray, cfg: AugmentationConfig, rng: random.Random) -> np.ndarray:
    out = binary.copy()

    if cfg.translate_enabled or cfg.affine_enabled:
        out = _apply_affine(out, cfg, rng)

    if cfg.morphology_enabled and cfg.morphology_prob > 0 and rng.random() < cfg.morphology_prob:
        out = _apply_morphology(out, cfg, rng)

    if cfg.noise_enabled and cfg.noise_prob > 0 and rng.random() < cfg.noise_prob:
        out = _apply_sparse_noise(out, cfg, rng)

    if cfg.rethreshold_after_aug:
        out = np.where(out > 127, 255, 0).astype(np.uint8)

    return out


def _apply_affine(binary: np.ndarray, cfg: AugmentationConfig, rng: random.Random) -> np.ndarray:
    h, w = binary.shape[:2]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    tx = 0.0
    ty = 0.0
    if cfg.translate_enabled and cfg.translate_prob > 0 and rng.random() < cfg.translate_prob:
        tx = rng.uniform(-max(0, cfg.translate_x_px), max(0, cfg.translate_x_px))
        ty = rng.uniform(-max(0, cfg.translate_y_px), max(0, cfg.translate_y_px))

    angle = 0.0
    shear = 0.0
    scale = 1.0
    if cfg.affine_enabled and cfg.affine_prob > 0 and rng.random() < cfg.affine_prob:
        angle = rng.uniform(-max(0.0, cfg.rotate_deg), max(0.0, cfg.rotate_deg))
        shear = rng.uniform(-max(0.0, cfg.shear_deg), max(0.0, cfg.shear_deg))
        scale_min = min(cfg.scale_min, cfg.scale_max)
        scale_max = max(cfg.scale_min, cfg.scale_max)
        scale = rng.uniform(scale_min, scale_max)

    if tx == 0.0 and ty == 0.0 and angle == 0.0 and shear == 0.0 and scale == 1.0:
        return binary

    affine = (
        _translation(tx, ty)
        @ _translation(cx, cy)
        @ _shear_x(math.radians(shear))
        @ _rotation(math.radians(angle))
        @ _scale(scale, scale)
        @ _translation(-cx, -cy)
    )
    matrix = affine[:2, :]
    return cv2.warpAffine(
        binary,
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _apply_morphology(binary: np.ndarray, cfg: AugmentationConfig, rng: random.Random) -> np.ndarray:
    k = max(1, int(cfg.morphology_kernel_size))
    kernel = np.ones((k, k), dtype=np.uint8)
    if rng.random() < 0.5:
        return cv2.erode(binary, kernel, iterations=1)
    return cv2.dilate(binary, kernel, iterations=1)


def _apply_sparse_noise(binary: np.ndarray, cfg: AugmentationConfig, rng: random.Random) -> np.ndarray:
    density = max(0.0, float(cfg.noise_density))
    if density <= 0:
        return binary

    out = binary.copy()
    num_pixels = out.size
    n = int(num_pixels * density)
    if n <= 0:
        return out

    flat = out.reshape(-1)
    idx = rng.sample(range(num_pixels), k=min(n, num_pixels))
    split = len(idx) // 2
    flat[idx[:split]] = 255
    flat[idx[split:]] = 0
    return out


def _translation(tx: float, ty: float) -> np.ndarray:
    return np.array(
        [[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _rotation(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _scale(sx: float, sy: float) -> np.ndarray:
    return np.array(
        [[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _shear_x(theta: float) -> np.ndarray:
    return np.array(
        [[1.0, math.tan(theta), 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
