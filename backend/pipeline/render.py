from __future__ import annotations

import cv2
import numpy as np

from .warp import WarpResult


def composite_with_realism(mannequin_bgr: np.ndarray, warped: WarpResult) -> np.ndarray:
    canvas = mannequin_bgr.copy().astype(np.float32)
    garment = warped.garment_rgba_on_canvas.astype(np.float32)
    alpha = (warped.alpha_on_canvas.astype(np.float32) / 255.0)[:, :, None]

    shadow = cv2.GaussianBlur(warped.alpha_on_canvas, (0, 0), 11)
    shadow = cv2.cvtColor(shadow, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    canvas = canvas * (1.0 - shadow * 0.18)

    canvas = garment[:, :, :3] * alpha + canvas * (1.0 - alpha)
    canvas = np.clip(canvas * np.array([0.995, 0.995, 1.0], dtype=np.float32), 0, 255)

    noise = np.random.default_rng(7).normal(0, 2.2, size=canvas.shape).astype(np.float32)
    canvas = np.clip(canvas + noise, 0, 255)
    return canvas.astype(np.uint8)
