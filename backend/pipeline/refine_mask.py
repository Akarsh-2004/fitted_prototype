from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

try:
    from cv2 import ximgproc as _ximgproc

    _HAS_GUIDED = hasattr(_ximgproc, "guidedFilter")
except Exception:
    _ximgproc = None
    _HAS_GUIDED = False

# Guided filter is O(pixels * radius^2); skip on large images for responsive CPU runs.
_GUIDED_MAX_PIXELS = 480_000


@dataclass
class RefineParams:
    edge_feather: int = 3
    morph_kernel: int = 3
    smoothness: float = 0.5


def _largest_contour_mask(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    largest = max(contours, key=cv2.contourArea)
    out = np.zeros_like(mask)
    cv2.drawContours(out, [largest], contourIdx=-1, color=255, thickness=cv2.FILLED)
    return out


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    flood = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, seedPoint=(0, 0), newVal=255)
    inv = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, inv)


def _build_trimap(binary: np.ndarray, band: int) -> np.ndarray:
    band = max(2, int(band))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band * 2 + 1, band * 2 + 1))
    sure_fg = cv2.erode(binary, kernel, iterations=1)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    trimap = np.full_like(binary, 128)
    trimap[sure_fg == 255] = 255
    trimap[dilated == 0] = 0
    return trimap


def _alpha_from_trimap(trimap: np.ndarray) -> np.ndarray:
    sure_fg = (trimap == 255).astype(np.uint8)
    sure_bg = (trimap == 0).astype(np.uint8)
    unknown = trimap == 128
    if not unknown.any():
        return sure_fg.astype(np.float32)
    d_fg = cv2.distanceTransform(1 - sure_fg, cv2.DIST_L2, 3)
    d_bg = cv2.distanceTransform(1 - sure_bg, cv2.DIST_L2, 3)
    alpha = sure_fg.astype(np.float32)
    ramp = d_bg / (d_fg + d_bg + 1e-6)
    alpha[unknown] = ramp[unknown]
    return np.clip(alpha, 0.0, 1.0)


def _edge_aware_smooth(
    alpha: np.ndarray,
    image_bgr: Optional[np.ndarray],
    feather: int,
) -> np.ndarray:
    if image_bgr is not None and _HAS_GUIDED:
        h, w = image_bgr.shape[:2]
        if h * w <= _GUIDED_MAX_PIXELS:
            guide = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            radius = max(2, feather * 2 + 1)
            return np.clip(
                _ximgproc.guidedFilter(guide, alpha.astype(np.float32), radius=radius, eps=1e-3),
                0.0,
                1.0,
            )
    sigma = max(0.6, float(feather) * 0.6)
    return np.clip(cv2.GaussianBlur(alpha, (0, 0), sigma), 0.0, 1.0)


def refine_mask(
    raw_mask: np.ndarray,
    params: RefineParams,
    image_bgr: Optional[np.ndarray] = None,
) -> np.ndarray:
    _, binary = cv2.threshold(raw_mask, 127, 255, cv2.THRESH_BINARY)
    ksize = max(3, int(params.morph_kernel) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    largest = _largest_contour_mask(opened)
    filled = _fill_holes(largest)

    feather = int(np.clip(params.edge_feather, 1, 8))
    band = max(2, feather + 1)
    trimap = _build_trimap(filled, band=band)
    alpha = _alpha_from_trimap(trimap)
    alpha = _edge_aware_smooth(alpha, image_bgr, feather=feather)

    contraction = 0.05 * float(np.clip(params.smoothness, 0.0, 1.0))
    if contraction > 0.0:
        alpha = np.clip((alpha - contraction) / max(1.0 - contraction, 1e-6), 0.0, 1.0)
    return (alpha * 255.0).astype(np.uint8)


def ensure_nonempty_mask(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    area = float((mask > 10).mean())
    if area >= 0.02:
        return mask

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    pad_h, pad_w = max(8, h // 14), max(8, w // 14)
    corners = np.concatenate(
        [
            lab[:pad_h, :pad_w].reshape(-1, 3),
            lab[:pad_h, w - pad_w:].reshape(-1, 3),
            lab[h - pad_h:, :pad_w].reshape(-1, 3),
            lab[h - pad_h:, w - pad_w:].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = np.median(corners, axis=0)
    dist = np.linalg.norm(lab - bg, axis=2)
    fallback = (dist > max(20.0, np.percentile(dist, 60))).astype(np.uint8) * 255
    fallback = cv2.morphologyEx(
        fallback, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1
    )
    fallback = _largest_contour_mask(fallback)
    fallback = _fill_holes(fallback)

    if float((fallback > 10).mean()) < 0.02:
        safe = np.zeros_like(mask)
        cv2.rectangle(
            safe,
            (int(w * 0.12), int(h * 0.08)),
            (int(w * 0.88), int(h * 0.94)),
            color=255,
            thickness=-1,
        )
        return safe
    return fallback


def compute_mask_quality(mask: np.ndarray) -> float:
    fg = (mask > 10).astype(np.uint8)
    area_ratio = float(fg.mean())
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True) + 1e-5
    compactness = float(4.0 * np.pi * cv2.contourArea(contour) / (perimeter * perimeter))
    border_ratio = float(
        np.concatenate((fg[0, :], fg[-1, :], fg[:, 0], fg[:, -1])).mean()
    )
    border_penalty = float(np.clip(1.0 - border_ratio * 2.0, 0.0, 1.0))
    soft_pixels = float(((mask > 10) & (mask < 245)).mean())
    edge_quality = float(np.clip(1.0 - abs(soft_pixels - 0.04) * 6.0, 0.0, 1.0))
    score = 0.35 * area_ratio + 0.25 * compactness + 0.2 * border_penalty + 0.2 * edge_quality
    return float(np.clip(score, 0.0, 1.0))
