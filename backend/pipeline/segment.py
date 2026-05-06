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
    """GrabCut iterations; 2 is usually enough on a downscaled image."""
    grabcut_iters: int = 2
    min_confidence: float = 0.22
    """Max long edge for segmentation (GrabCut cost grows ~with pixels). Full-res mask is restored after."""
    max_seg_long_edge: int = 640


@dataclass
class SegmentResult:
    mask: np.ndarray
    confidence: float
    method: str
    elapsed_s: float
    used_fallback: bool


def _corner_patches(image_lab: np.ndarray) -> np.ndarray:
    h, w = image_lab.shape[:2]
    pad_h = max(8, h // 14)
    pad_w = max(8, w // 14)
    patches = [
        image_lab[:pad_h, :pad_w],
        image_lab[:pad_h, w - pad_w:],
        image_lab[h - pad_h:, :pad_w],
        image_lab[h - pad_h:, w - pad_w:],
    ]
    return np.concatenate([p.reshape(-1, 3) for p in patches], axis=0).astype(np.float32)


def _estimate_background_lab(image_lab: np.ndarray) -> tuple[np.ndarray, float]:
    """Return median LAB color of corners and a uniformity score (0..1)."""
    samples = _corner_patches(image_lab)
    median = np.median(samples, axis=0)
    spread = float(np.linalg.norm(np.std(samples, axis=0)))
    uniformity = float(np.clip(1.0 - spread / 60.0, 0.0, 1.0))
    return median, uniformity


def _color_distance_saliency(image_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-pixel LAB distance from estimated background; returns saliency + bg uniformity."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg, uniformity = _estimate_background_lab(lab)
    dist = np.linalg.norm(lab - bg, axis=2)
    saliency = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Gaussian is ~10x faster than bilateralFilter on CPU with similar smoothing here.
    saliency = cv2.GaussianBlur(saliency, (5, 5), 0)
    return saliency, uniformity


def _resize_for_segmentation(image_bgr: np.ndarray, max_long_edge: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Return (possibly scaled image, scale relative to original, (orig_w, orig_h))."""
    orig_h, orig_w = image_bgr.shape[:2]
    long_edge = max(orig_h, orig_w)
    if long_edge <= max_long_edge:
        return image_bgr, 1.0, (orig_w, orig_h)
    scale = max_long_edge / float(long_edge)
    nw = max(1, int(round(orig_w * scale)))
    nh = max(1, int(round(orig_h * scale)))
    small = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    return small, scale, (orig_w, orig_h)


def _bbox_from_saliency(saliency: np.ndarray) -> tuple[int, int, int, int] | None:
    h, w = saliency.shape
    _, hot = cv2.threshold(saliency, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    hot = cv2.morphologyEx(hot, cv2.MORPH_OPEN, open_kernel, iterations=1)
    hot = cv2.morphologyEx(
        hot, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1
    )
    contours, _ = cv2.findContours(hot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(contour)
    if bw < int(0.06 * w) or bh < int(0.06 * h):
        return None
    pad_x = max(2, int(bw * 0.06))
    pad_y = max(2, int(bh * 0.06))
    x0 = max(1, x - pad_x)
    y0 = max(1, y - pad_y)
    x1 = min(w - 2, x + bw + pad_x)
    y1 = min(h - 2, y + bh + pad_y)
    if x1 - x0 < 10 or y1 - y0 < 10:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def _grabcut_segment(
    image_bgr: np.ndarray,
    iters: int,
    max_seg_long_edge: int,
) -> tuple[np.ndarray, float]:
    orig_h, orig_w = image_bgr.shape[:2]
    work, _, _ = _resize_for_segmentation(image_bgr, max_seg_long_edge)

    saliency, _ = _color_distance_saliency(work)
    rect = _bbox_from_saliency(saliency)
    if rect is None:
        return np.zeros((orig_h, orig_w), dtype=np.uint8), 0.0

    h, w = work.shape[:2]
    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    x, y, bw, bh = rect
    gc_mask[y : y + bh, x : x + bw] = cv2.GC_PR_FGD
    inner_thr = max(70, int(np.percentile(saliency[y : y + bh, x : x + bw], 80)))
    roi_sal = saliency[y : y + bh, x : x + bw]
    sub = gc_mask[y : y + bh, x : x + bw]
    sub[roi_sal >= inner_thr] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(work, gc_mask, rect, bgd_model, fgd_model, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return np.zeros((orig_h, orig_w), dtype=np.uint8), 0.0

    mask_small = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    if mask_small.shape[0] == orig_h and mask_small.shape[1] == orig_w:
        mask = mask_small
    else:
        mask = cv2.resize(mask_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    area = float((mask > 0).mean())
    border = float(
        np.concatenate((mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1])).mean() / 255.0
    )
    confidence = float(np.clip(area * 1.6 - border * 0.8, 0.0, 1.0))
    return mask, confidence


def _color_distance_segment(image_bgr: np.ndarray, max_seg_long_edge: int) -> tuple[np.ndarray, float]:
    orig_h, orig_w = image_bgr.shape[:2]
    work, _, _ = _resize_for_segmentation(image_bgr, max_seg_long_edge)
    saliency, uniformity = _color_distance_saliency(work)
    threshold = max(40, int(np.percentile(saliency, 55)))
    _, mask_small = cv2.threshold(saliency, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_OPEN, kernel, iterations=1)
    if mask_small.shape[0] == orig_h and mask_small.shape[1] == orig_w:
        mask = mask_small
    else:
        mask = cv2.resize(mask_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    confidence = float(np.clip((mask.mean() / 255.0) * (0.5 + 0.5 * uniformity), 0.0, 1.0))
    return mask, confidence


def segment_with_fallback(image: np.ndarray, config: SegmentConfig) -> SegmentResult:
    cache = config.cache_dir / f"{config.request_key}_seg.png"
    if cache.exists():
        cached = cv2.imread(str(cache), cv2.IMREAD_GRAYSCALE)
        if cached is not None:
            return SegmentResult(
                mask=cached,
                confidence=float(np.clip(cached.mean() / 255.0, 0.0, 1.0)),
                method="cached",
                elapsed_s=0.0,
                used_fallback=False,
            )

    t0 = time.perf_counter()
    mask, conf = _grabcut_segment(
        image,
        config.grabcut_iters,
        max_seg_long_edge=config.max_seg_long_edge,
    )
    used_fallback = False
    method = "grabcut"
    if conf < config.min_confidence or (mask > 0).mean() < 0.02:
        fb_mask, fb_conf = _color_distance_segment(image, config.max_seg_long_edge)
        if fb_conf > conf:
            mask, conf = fb_mask, fb_conf
            used_fallback = True
            method = "color_distance"
    elapsed = time.perf_counter() - t0
    cv2.imwrite(str(cache), mask)
    return SegmentResult(
        mask=mask,
        confidence=conf,
        method=method,
        elapsed_s=elapsed,
        used_fallback=used_fallback,
    )
