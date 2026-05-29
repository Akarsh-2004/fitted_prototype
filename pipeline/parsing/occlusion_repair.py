from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from pipeline.config import settings


@dataclass
class RepairResult:
    mask: np.ndarray
    metadata: Dict[str, Any]


@dataclass
class CompletionResult:
    mask: np.ndarray
    missing_mask: np.ndarray
    metadata: Dict[str, Any]


def polygon_to_mask(shape: Tuple[int, int], polygon: List[List[float]]) -> np.ndarray:
    """Rasterize a polygon into a binary mask."""
    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    if len(pts) >= 3:
        cv2.fillPoly(mask, [pts], 255)
    return mask


def build_blocked_mask(
    shape: Tuple[int, int],
    cached_people: List[Dict[str, Any]],
    selected_index: int,
    dilate_px: int = 3,
) -> np.ndarray:
    """Build a hard negative mask from non-selected people in a group photo."""
    blocked = np.zeros(shape, dtype=np.uint8)
    for idx, person in enumerate(cached_people):
        if idx == selected_index:
            continue
        polygon = person.get("polygon") or []
        blocked = cv2.bitwise_or(blocked, polygon_to_mask(shape, polygon))

    if dilate_px > 0 and np.any(blocked > 0):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px | 1, dilate_px | 1))
        blocked = cv2.dilate(blocked, kernel, iterations=1)
    return blocked


def crop_mask(mask: Optional[np.ndarray], x1: int, y1: int, x2: int, y2: int) -> Optional[np.ndarray]:
    if mask is None:
        return None
    return mask[y1:y2, x1:x2].copy()


def constrain_mask(mask: np.ndarray, allowed_mask: Optional[np.ndarray], blocked_mask: Optional[np.ndarray]) -> np.ndarray:
    constrained = (mask > 0).astype(np.uint8) * 255
    if allowed_mask is not None:
        constrained = cv2.bitwise_and(constrained, (allowed_mask > 0).astype(np.uint8) * 255)
    if blocked_mask is not None:
        constrained[blocked_mask > 0] = 0
    return constrained


def component_filter(mask: np.ndarray, min_area: int, max_components: int = 1) -> Tuple[np.ndarray, int]:
    """Keep the largest N components above min_area."""
    if not np.any(mask > 0):
        return np.zeros_like(mask), 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components = []
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area >= min_area:
            components.append((area, label_idx))

    components.sort(reverse=True)
    kept = components[:max_components]
    filtered = np.zeros_like(mask)
    for _, label_idx in kept:
        filtered[labels == label_idx] = 255
    return filtered, len(kept)


class OcclusionRepair:
    """Inference-time mask repair inspired by amodal gating and occlusion-aware VTON."""

    MAX_COMPONENTS = {
        "footwear": 2,
        "left_shoe": 1,
        "right_shoe": 1,
        "top_garment": 1,
        "outerwear": 1,
        "bottom_garment": 1,
        "hat": 1,
        "bag": 1,
        "accessory": 2,
    }

    CONF_THRESHOLDS = {
        "top_garment": 0.28,
        "outerwear": 0.28,
        "bottom_garment": 0.28,
        "footwear": 0.24,
        "hat": 0.24,
        "bag": 0.22,
        "accessory": 0.20,
    }

    NO_HOLE_FILL = {"bag", "accessory", "hat", "footwear", "left_shoe", "right_shoe"}
    AMODAL_FILL_LABELS = {"top_garment", "outerwear", "bottom_garment"}

    def repair(
        self,
        label: str,
        raw_mask: np.ndarray,
        *,
        allowed_mask: Optional[np.ndarray] = None,
        blocked_mask: Optional[np.ndarray] = None,
        confidence_map: Optional[np.ndarray] = None,
    ) -> RepairResult:
        before_pixels = int(np.sum(raw_mask > 0))
        blocked_overlap = int(np.sum((raw_mask > 0) & (blocked_mask > 0))) if blocked_mask is not None else 0

        mask = constrain_mask(raw_mask, allowed_mask, blocked_mask)
        mask = self._apply_label_band(label, mask)

        if confidence_map is not None and np.any(mask > 0):
            threshold = self.CONF_THRESHOLDS.get(label, 0.25)
            confident = (confidence_map >= threshold).astype(np.uint8) * 255
            # Preserve the eroded core even when edge confidence is weak.
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            core = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.bitwise_or(cv2.bitwise_and(mask, confident), core)

        mask = self._morphology(label, mask)
        min_area = max(50, int(mask.shape[0] * mask.shape[1] * 0.002))
        max_components = self.MAX_COMPONENTS.get(label, 1)
        mask, component_count = component_filter(mask, min_area, max_components=max_components)
        mask = constrain_mask(mask, allowed_mask, blocked_mask)

        after_pixels = int(np.sum(mask > 0))
        metadata = {
            "repair_applied": True,
            "raw_pixel_area": before_pixels,
            "repaired_pixel_area": after_pixels,
            "blocked_pixels_removed": blocked_overlap,
            "component_count": component_count,
        }
        return RepairResult(mask=mask, metadata=metadata)

    def complete_occluded_garment(
        self,
        label: str,
        visible_mask: np.ndarray,
        *,
        allowed_mask: Optional[np.ndarray] = None,
        blocked_mask: Optional[np.ndarray] = None,
    ) -> CompletionResult:
        """Fill likely hidden garment regions using deterministic amodal shape cues.

        This borrows the paper idea at inference time: visible pixels remain the
        reliable anchor, while only internal/near-boundary occluded gaps are
        completed. Known neighbouring people remain hard background.
        """
        base = constrain_mask(visible_mask, allowed_mask, blocked_mask)
        empty_missing = np.zeros_like(base)
        if label not in self.AMODAL_FILL_LABELS or not np.any(base > 0):
            return CompletionResult(base, empty_missing, {
                "amodal_fill_applied": False,
                "amodal_missing_pixels": 0,
            })

        completed = self._close_shape(label, base)
        if label in {"top_garment", "outerwear"}:
            completed = cv2.bitwise_or(completed, self._row_span_fill(base, min_row_pixels_ratio=0.08))
            completed = cv2.bitwise_or(completed, self._column_span_fill(base, min_col_pixels_ratio=0.10))
        elif label == "bottom_garment":
            completed = cv2.bitwise_or(completed, self._row_span_fill(base, min_row_pixels_ratio=0.10, max_span_ratio=0.72))

        completed = constrain_mask(completed, allowed_mask, blocked_mask)
        completed = self._limit_completion_growth(base, completed, max_growth_ratio=0.85 if label != "bottom_garment" else 0.55)
        completed = constrain_mask(completed, allowed_mask, blocked_mask)

        missing = cv2.subtract(completed, base)
        missing_pixels = int(np.sum(missing > 0))
        return CompletionResult(completed, missing, {
            "amodal_fill_applied": missing_pixels > 0,
            "amodal_missing_pixels": missing_pixels,
        })

    def inpaint_missing_texture(
        self,
        crop_bgr: np.ndarray,
        visible_mask: np.ndarray,
        missing_mask: np.ndarray,
    ) -> np.ndarray:
        if not np.any(missing_mask > 0):
            return crop_bgr

        # Neutralize non-garment pixels so OpenCV samples texture from the
        # visible garment boundary instead of skin/arms/background.
        source = crop_bgr.copy()
        source[visible_mask == 0] = self._mean_visible_color(crop_bgr, visible_mask)
        inpaint_mask = (missing_mask > 0).astype(np.uint8) * 255
        return cv2.inpaint(source, inpaint_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    def _morphology(self, label: str, mask: np.ndarray) -> np.ndarray:
        if not np.any(mask > 0):
            return mask

        kernel_size = 3 if label in {"hat", "bag", "accessory", "footwear"} else 5
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        repaired = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        repaired = cv2.morphologyEx(repaired, cv2.MORPH_CLOSE, kernel, iterations=1)

        if label not in self.NO_HOLE_FILL:
            repaired = self._fill_small_holes(repaired, max_hole_area=max(120, int(mask.size * 0.01)))
        return repaired

    def _fill_small_holes(self, mask: np.ndarray, max_hole_area: int) -> np.ndarray:
        filled = mask.copy()
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return filled

        hierarchy = hierarchy[0]
        for idx, contour in enumerate(contours):
            parent = hierarchy[idx][3]
            if parent >= 0 and cv2.contourArea(contour) <= max_hole_area:
                cv2.drawContours(filled, [contour], -1, 255, -1)
        return filled

    def _close_shape(self, label: str, mask: np.ndarray) -> np.ndarray:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return mask

        bbox_w = int(xs.max() - xs.min() + 1)
        bbox_h = int(ys.max() - ys.min() + 1)
        if label in {"top_garment", "outerwear"}:
            k_w = max(9, int(bbox_w * 0.22) | 1)
            k_h = max(7, int(bbox_h * 0.12) | 1)
        else:
            k_w = max(7, int(bbox_w * 0.14) | 1)
            k_h = max(9, int(bbox_h * 0.16) | 1)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_w, k_h))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    def _row_span_fill(
        self,
        mask: np.ndarray,
        min_row_pixels_ratio: float,
        max_span_ratio: float = 0.88,
    ) -> np.ndarray:
        h, w = mask.shape[:2]
        filled = np.zeros_like(mask)
        min_pixels = max(4, int(w * min_row_pixels_ratio))

        for y in range(h):
            xs = np.where(mask[y] > 0)[0]
            if len(xs) < min_pixels:
                continue
            left = int(xs.min())
            right = int(xs.max())
            if (right - left + 1) > int(w * max_span_ratio):
                continue
            filled[y, left:right + 1] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        return cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel, iterations=1)

    def _column_span_fill(
        self,
        mask: np.ndarray,
        min_col_pixels_ratio: float,
        max_span_ratio: float = 0.78,
    ) -> np.ndarray:
        h, w = mask.shape[:2]
        filled = np.zeros_like(mask)
        min_pixels = max(4, int(h * min_col_pixels_ratio))

        for x in range(w):
            ys = np.where(mask[:, x] > 0)[0]
            if len(ys) < min_pixels:
                continue
            top = int(ys.min())
            bottom = int(ys.max())
            if (bottom - top + 1) > int(h * max_span_ratio):
                continue
            filled[top:bottom + 1, x] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        return cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel, iterations=1)

    def _limit_completion_growth(self, base: np.ndarray, completed: np.ndarray, max_growth_ratio: float) -> np.ndarray:
        base_area = int(np.sum(base > 0))
        if base_area <= 0:
            return base

        missing = cv2.subtract(completed, base)
        missing_area = int(np.sum(missing > 0))
        if missing_area <= base_area * max_growth_ratio:
            return completed

        # Keep nearest completed pixels first; this prevents shape priors from
        # exploding into large background regions.
        dist = cv2.distanceTransform((base == 0).astype(np.uint8), cv2.DIST_L2, 3)
        candidate_ys, candidate_xs = np.where(missing > 0)
        order = np.argsort(dist[candidate_ys, candidate_xs])
        keep_count = int(base_area * max_growth_ratio)
        limited = base.copy()
        if keep_count > 0:
            keep_ys = candidate_ys[order[:keep_count]]
            keep_xs = candidate_xs[order[:keep_count]]
            limited[keep_ys, keep_xs] = 255
        return limited

    def _mean_visible_color(self, crop_bgr: np.ndarray, visible_mask: np.ndarray) -> np.ndarray:
        pixels = crop_bgr[visible_mask > 0]
        if pixels.size == 0:
            return np.array([128, 128, 128], dtype=np.uint8)
        return np.median(pixels, axis=0).astype(np.uint8)

    def _apply_label_band(self, label: str, mask: np.ndarray) -> np.ndarray:
        if not np.any(mask > 0):
            return mask

        h, w = mask.shape[:2]
        band = np.zeros((h, w), dtype=np.uint8)
        if label in {"top_garment", "outerwear"}:
            band[: int(h * 0.78), :] = 255
        elif label == "bottom_garment":
            band[int(h * 0.28):, :] = 255
        elif label in {"footwear", "left_shoe", "right_shoe"}:
            band[int(h * 0.60):, :] = 255
        elif label == "hat":
            band[: int(h * 0.35), :] = 255
        else:
            return mask

        # Widen the band to avoid cutting valid garments near boundaries.
        expand = max(5, int(h * 0.04))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, expand | 1))
        band = cv2.dilate(band, kernel, iterations=1)
        return cv2.bitwise_and(mask, band)


occlusion_repair = OcclusionRepair()
