from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class SegmentConfig:
    cache_dir: Path
    request_key: str
    sam_timeout_s: float = 8.0
    min_confidence: float = 0.35


@dataclass
class SegmentResult:
    mask: np.ndarray
    confidence: float
    used_fallback: bool
    elapsed_s: float


def _sam_like_segment(image: np.ndarray) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    confidence = float(np.clip(mask.mean() / 255.0, 0.0, 1.0))
    return mask, confidence


def _u2net_like_segment(image: np.ndarray) -> tuple[np.ndarray, float]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = np.where((sat > 35) | (val < 220), 255, 0).astype(np.uint8)
    confidence = float(np.clip(mask.mean() / 255.0, 0.0, 1.0))
    return mask, confidence


def segment_with_fallback(image: np.ndarray, config: SegmentConfig) -> SegmentResult:
    cache = config.cache_dir / f"{config.request_key}_seg.png"
    if cache.exists():
        mask = cv2.imread(str(cache), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            return SegmentResult(mask=mask, confidence=float(np.clip(mask.mean() / 255.0, 0.0, 1.0)), used_fallback=False, elapsed_s=0.0)

    t0 = time.perf_counter()
    sam_mask, sam_conf = _sam_like_segment(image)
    elapsed = time.perf_counter() - t0
    use_fallback = elapsed > config.sam_timeout_s or sam_conf < config.min_confidence

    if use_fallback:
        fb_t0 = time.perf_counter()
        mask, conf = _u2net_like_segment(image)
        elapsed = time.perf_counter() - fb_t0
        cv2.imwrite(str(cache), mask)
        return SegmentResult(mask=mask, confidence=conf, used_fallback=True, elapsed_s=elapsed)

    cv2.imwrite(str(cache), sam_mask)
    return SegmentResult(mask=sam_mask, confidence=sam_conf, used_fallback=False, elapsed_s=elapsed)
