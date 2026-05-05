from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class WarpResult:
    cutout_rgba: np.ndarray
    garment_rgba_on_canvas: np.ndarray
    alpha_on_canvas: np.ndarray
    warp_mode: str


def _anchor_points(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 10)
    if len(xs) == 0:
        h, w = mask.shape
        return np.float32([[w * 0.5, h * 0.1], [w * 0.2, h * 0.25], [w * 0.8, h * 0.25], [w * 0.5, h * 0.9]])
    top = np.argmin(ys)
    bottom = np.argmax(ys)
    top_center = np.array([xs[top], ys[top]], dtype=np.float32)
    bottom_center = np.array([xs[bottom], ys[bottom]], dtype=np.float32)
    shoulder_y = int(np.percentile(ys, 22))
    shoulder_band = np.where(np.abs(ys - shoulder_y) < 8)[0]
    left = np.array([np.min(xs[shoulder_band]), shoulder_y], dtype=np.float32)
    right = np.array([np.max(xs[shoulder_band]), shoulder_y], dtype=np.float32)
    return np.float32([top_center, left, right, bottom_center])


def warp_garment_to_mannequin(image_bgr: np.ndarray, alpha_mask: np.ndarray, mannequin_bgr: np.ndarray) -> WarpResult:
    h, w = mannequin_bgr.shape[:2]
    source = _anchor_points(alpha_mask)
    dest = np.float32([
        [w * 0.5, h * 0.18],
        [w * 0.31, h * 0.3],
        [w * 0.69, h * 0.3],
        [w * 0.5, h * 0.78],
    ])

    M = cv2.getAffineTransform(source[:3], dest[:3])
    warped_img = cv2.warpAffine(image_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    warped_alpha = cv2.warpAffine(alpha_mask, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    warp_mode = "affine"

    rgba_cutout = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
    rgba_cutout[:, :, 3] = alpha_mask
    garment_rgba = cv2.cvtColor(warped_img, cv2.COLOR_BGR2BGRA)
    garment_rgba[:, :, 3] = warped_alpha

    return WarpResult(
        cutout_rgba=rgba_cutout,
        garment_rgba_on_canvas=garment_rgba,
        alpha_on_canvas=warped_alpha,
        warp_mode=warp_mode,
    )
