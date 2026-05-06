from __future__ import annotations

import cv2
import numpy as np

from .warp import WarpResult


def _harmonize_luminance(garment_bgr: np.ndarray, scene_bgr: np.ndarray, garment_mask: np.ndarray) -> np.ndarray:
    if not garment_mask.any():
        return garment_bgr
    garment_lab = cv2.cvtColor(garment_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    scene_lab = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    scene_l = float(scene_lab[:, :, 0].mean())
    gar_l = float(garment_lab[:, :, 0][garment_mask].mean())
    delta = float(np.clip((scene_l - gar_l) * 0.18, -8.0, 8.0))
    if abs(delta) < 0.5:
        return garment_bgr
    garment_lab[:, :, 0] = np.clip(garment_lab[:, :, 0] + delta, 0.0, 255.0)
    return cv2.cvtColor(garment_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _drop_shadow(alpha_8u: np.ndarray, offset: tuple[int, int] = (10, 16), sigma: float = 22.0) -> np.ndarray:
    shadow = cv2.GaussianBlur(alpha_8u, (0, 0), sigma)
    M = np.float32([[1.0, 0.0, float(offset[0])], [0.0, 1.0, float(offset[1])]])
    h, w = shadow.shape[:2]
    shadow = cv2.warpAffine(shadow, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return shadow.astype(np.float32) / 255.0


def composite_with_realism(mannequin_bgr: np.ndarray, warped: WarpResult) -> np.ndarray:
    canvas = mannequin_bgr.astype(np.float32)
    alpha_8u = warped.alpha_on_canvas
    alpha = alpha_8u.astype(np.float32) / 255.0

    garment_bgr = warped.garment_rgba_on_canvas[:, :, :3]
    garment_mask = alpha > 0.25
    garment_bgr = _harmonize_luminance(garment_bgr, mannequin_bgr, garment_mask)

    shadow = _drop_shadow(alpha_8u, offset=(10, 18), sigma=24.0)
    shadow3 = shadow[:, :, None]
    canvas = canvas * (1.0 - shadow3 * 0.24)

    contact = cv2.GaussianBlur(alpha_8u, (0, 0), 4.0).astype(np.float32) / 255.0
    contact = np.maximum(contact - alpha, 0.0)[:, :, None]
    canvas = canvas * (1.0 - contact * 0.35)

    a3 = alpha[:, :, None]
    canvas = garment_bgr.astype(np.float32) * a3 + canvas * (1.0 - a3)

    canvas = np.clip(canvas, 0.0, 255.0)
    return canvas.astype(np.uint8)
