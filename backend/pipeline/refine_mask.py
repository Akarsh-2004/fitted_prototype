from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


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
    flood = mask.copy()
    h, w = mask.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, seedPoint=(0, 0), newVal=255)
    inv = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, inv)


def refine_mask(raw_mask: np.ndarray, params: RefineParams) -> np.ndarray:
    _, binary = cv2.threshold(raw_mask, 127, 255, cv2.THRESH_BINARY)
    ksize = max(3, int(params.morph_kernel) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    largest = _largest_contour_mask(opened)
    filled = _fill_holes(largest)

    edge = cv2.Canny(filled, 80, 180)
    feather = max(1, min(4, int(params.edge_feather)))
    edge_blur = cv2.GaussianBlur(edge, (0, 0), sigmaX=feather)
    alpha = filled.astype(np.float32)
    alpha = np.clip(alpha - edge_blur * (0.2 + params.smoothness * 0.4), 0, 255)
    return alpha.astype(np.uint8)


def ensure_nonempty_mask(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    area = float((mask > 10).mean())
    if area >= 0.02:
        return mask

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    fallback = np.where((sat > 28) | (val < 238), 255, 0).astype(np.uint8)
    fallback = cv2.morphologyEx(
        fallback, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    fallback = _largest_contour_mask(fallback)
    fallback = _fill_holes(fallback)

    fallback_area = float((fallback > 10).mean())
    if fallback_area < 0.02:
        # Last-resort clamp: keep visible center crop instead of empty cutout.
        h, w = mask.shape[:2]
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
    area_ratio = float((mask > 10).mean())
    contours, _ = cv2.findContours((mask > 10).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    c = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(c, True) + 1e-5
    compactness = float(4 * np.pi * cv2.contourArea(c) / (perimeter * perimeter))
    lap = cv2.Laplacian(mask, cv2.CV_32F).var()
    edge_smoothness = float(np.clip(1.0 - (lap / 4000.0), 0.0, 1.0))
    score = 0.4 * area_ratio + 0.3 * compactness + 0.3 * edge_smoothness
    return float(np.clip(score, 0.0, 1.0))
