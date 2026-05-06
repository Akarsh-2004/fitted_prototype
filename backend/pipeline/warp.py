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
    bbox: tuple[int, int, int, int]


def _mask_bbox(alpha_mask: np.ndarray) -> tuple[int, int, int, int] | None:
    binary = (alpha_mask > 10).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(contour)
    if bw < 8 or bh < 8:
        return None
    return (int(x), int(y), int(bw), int(bh))


def _tight_cutout(image_bgr: np.ndarray, alpha_mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    src_h, src_w = image_bgr.shape[:2]
    x, y, bw, bh = bbox
    pad = int(0.04 * max(bw, bh))
    cx0 = max(0, x - pad)
    cy0 = max(0, y - pad)
    cx1 = min(src_w, x + bw + pad)
    cy1 = min(src_h, y + bh + pad)
    rgb = image_bgr[cy0:cy1, cx0:cx1]
    alpha = alpha_mask[cy0:cy1, cx0:cx1]
    rgba = cv2.cvtColor(rgb, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    return rgba


def warp_garment_to_mannequin(
    image_bgr: np.ndarray,
    alpha_mask: np.ndarray,
    mannequin_bgr: np.ndarray,
) -> WarpResult:
    canvas_h, canvas_w = mannequin_bgr.shape[:2]
    src_h, src_w = image_bgr.shape[:2]

    bbox = _mask_bbox(alpha_mask) or (0, 0, src_w, src_h)
    x, y, bw, bh = bbox

    target_cx = canvas_w * 0.5
    target_cy = canvas_h * 0.46
    target_max_w = canvas_w * 0.62
    target_max_h = canvas_h * 0.64

    scale = float(min(target_max_w / max(bw, 1), target_max_h / max(bh, 1)))

    src_cx = x + bw * 0.5
    src_cy = y + bh * 0.5
    tx = target_cx - src_cx * scale
    ty = target_cy - src_cy * scale
    affine = np.float32([[scale, 0.0, tx], [0.0, scale, ty]])

    warped_img = cv2.warpAffine(
        image_bgr,
        affine,
        (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    warped_alpha = cv2.warpAffine(
        alpha_mask,
        affine,
        (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    cutout_rgba = _tight_cutout(image_bgr, alpha_mask, bbox)
    garment_rgba = cv2.cvtColor(warped_img, cv2.COLOR_BGR2BGRA)
    garment_rgba[:, :, 3] = warped_alpha

    return WarpResult(
        cutout_rgba=cutout_rgba,
        garment_rgba_on_canvas=garment_rgba,
        alpha_on_canvas=warped_alpha,
        warp_mode="similarity",
        bbox=bbox,
    )
