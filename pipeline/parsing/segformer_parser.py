import logging
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from pipeline.config import settings
from pipeline.parsing.garment_refiner import garment_refiner
from pipeline.parsing.mask_cleanup import MaskCleanup
from pipeline.parsing.occlusion_repair import crop_mask, occlusion_repair
from pipeline.services.model_registry import model_registry
from pipeline.services.storage_service import storage_service

logger = logging.getLogger("VestirAI.SegFormerParser")


def _normalize_label(label: str) -> str:
    return label.strip().lower().replace("_", "-").replace(" ", "-")


class SegFormerParser:
    """
    True SegFormer clothes parser for selected group-photo people.

    The model emits clothing semantic labels. This helper maps those labels into
    the app's stable parsed-part names so the existing UI, wardrobe store, and
    composer alignment can continue unchanged.
    """

    LABEL_GROUPS = {
        "top_garment": {"upper-clothes", "upper-clothing", "shirt", "t-shirt", "top", "dress"},
        "bottom_garment": {"pants", "trousers", "jeans", "skirt", "shorts"},
        "footwear": {"left-shoe", "right-shoe", "shoe", "shoes", "footwear"},
        "hat": {"hat", "cap", "beanie"},
        "bag": {"bag", "backpack", "handbag", "purse"},
        "accessory": {"scarf", "glove", "sunglasses", "belt"},
    }

    INGEST_LABELS = {"top_garment", "bottom_garment", "footwear", "hat"}

    def parse_clothing_layers(
        self,
        img_bgr: np.ndarray,
        person_mask: np.ndarray,
        person_box: List[float],
        job_id: str,
        person_index: int,
        blocked_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        segformer_bundle = model_registry.get_segformer()
        if segformer_bundle is None:
            raise RuntimeError("SegFormer model is not available. Check model download/cache status.")

        processor, model = segformer_bundle

        h_img, w_img = img_bgr.shape[:2]
        px1, py1, px2, py2 = [int(v) for v in person_box]
        pad_x = int(max(8, (px2 - px1) * 0.04))
        pad_y = int(max(8, (py2 - py1) * 0.04))
        crop_x1 = max(0, px1 - pad_x)
        crop_y1 = max(0, py1 - pad_y)
        crop_x2 = min(w_img, px2 + pad_x)
        crop_y2 = min(h_img, py2 + pad_y)

        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1
        if crop_w <= 10 or crop_h <= 10:
            raise ValueError(f"Selected person crop is too small: {crop_w}x{crop_h}")

        person_crop_bgr = img_bgr[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        person_mask_crop = person_mask[crop_y1:crop_y2, crop_x1:crop_x2]
        blocked_mask_crop = crop_mask(blocked_mask, crop_x1, crop_y1, crop_x2, crop_y2)
        visible_support_mask = person_mask_crop.copy()
        if blocked_mask_crop is not None:
            visible_support_mask[blocked_mask_crop > 0] = 0

        neutral_crop = person_crop_bgr.copy()
        neutral_crop[visible_support_mask == 0] = 128
        crop_rgb = cv2.cvtColor(neutral_crop, cv2.COLOR_BGR2RGB)

        device = next(model.parameters()).device
        inputs = processor(images=crop_rgb, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        upsampled_logits = F.interpolate(
            logits,
            size=(crop_h, crop_w),
            mode="bilinear",
            align_corners=False,
        )
        probs = torch.softmax(upsampled_logits, dim=1)
        conf_tensor, pred_tensor = probs.max(dim=1)
        seg_map = pred_tensor.squeeze(0).cpu().numpy().astype(np.uint8)
        confidence_map = conf_tensor.squeeze(0).cpu().numpy().astype(np.float32)

        id_to_label = {
            int(idx): _normalize_label(str(label))
            for idx, label in getattr(model.config, "id2label", {}).items()
        }
        raw_masks = self._build_raw_masks(seg_map, id_to_label, visible_support_mask)

        cutout_result = self._save_selected_person_cutout(
            person_crop_bgr,
            visible_support_mask,
            job_id,
            person_index,
            crop_x1,
            crop_y1,
            crop_w,
            crop_h,
        )
        parts = self._save_parts(
            person_crop_bgr,
            raw_masks,
            confidence_map,
            job_id,
            person_index,
            crop_x1,
            crop_y1,
            visible_support_mask,
            blocked_mask_crop,
        )

        return {
            "person_id": f"person_{job_id}_{person_index}",
            "cutout": cutout_result,
            "parts": parts,
            "parser_used": "segformer",
        }

    def _build_raw_masks(
        self,
        seg_map: np.ndarray,
        id_to_label: Dict[int, str],
        person_mask_crop: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        h, w = seg_map.shape[:2]
        raw_masks: Dict[str, np.ndarray] = {}

        for app_label, model_labels in self.LABEL_GROUPS.items():
            combined = np.zeros((h, w), dtype=np.uint8)
            for class_id, model_label in id_to_label.items():
                if model_label in model_labels:
                    combined = cv2.bitwise_or(combined, (seg_map == class_id).astype(np.uint8) * 255)

            if np.any(combined > 0):
                combined = cv2.bitwise_and(combined, person_mask_crop)
                raw_masks[app_label] = combined

        return raw_masks

    def _save_selected_person_cutout(
        self,
        person_crop_bgr: np.ndarray,
        person_mask_crop: np.ndarray,
        job_id: str,
        person_index: int,
        crop_x1: int,
        crop_y1: int,
        crop_w: int,
        crop_h: int,
    ) -> Dict[str, Any]:
        cutout_rgba = cv2.cvtColor(person_crop_bgr, cv2.COLOR_BGR2RGBA)
        cutout_rgba[:, :, 3] = person_mask_crop
        item_id = f"person_{job_id}_{person_index}_segformer"
        crop_path = storage_service.save_crop_rgba(cutout_rgba, item_id)

        mask_path = settings.storage_dir / "crops" / f"{item_id}_mask.png"
        cv2.imwrite(str(mask_path), person_mask_crop)

        coverage_ratio = float(np.sum(person_mask_crop > 0) / max(1, crop_w * crop_h))
        return {
            "rgba_crop_path": storage_service.get_relative_path(crop_path),
            "mask_path": storage_service.get_relative_path(mask_path),
            "coverage_ratio": round(coverage_ratio, 4),
            "method_used": "segformer",
            "bbox": [crop_x1, crop_y1, crop_w, crop_h],
            "contour_polygon": [],
        }

    def _save_parts(
        self,
        person_crop_bgr: np.ndarray,
        raw_masks: Dict[str, np.ndarray],
        confidence_map: np.ndarray,
        job_id: str,
        person_index: int,
        crop_x1: int,
        crop_y1: int,
        allowed_mask: np.ndarray,
        blocked_mask: Optional[np.ndarray],
    ) -> List[Dict[str, Any]]:
        crop_h, crop_w = person_crop_bgr.shape[:2]
        min_area = max(80, int(crop_h * crop_w * 0.003))
        parts: List[Dict[str, Any]] = []

        for label, raw_mask in raw_masks.items():
            repair_result = occlusion_repair.repair(
                label,
                raw_mask,
                allowed_mask=allowed_mask,
                blocked_mask=blocked_mask,
                confidence_map=confidence_map,
            )
            preserve_components = 3 if label in {"top_garment", "outerwear"} else 2 if label in {"footwear", "accessory"} else 1
            fill_holes = label not in {"hat", "bag", "accessory", "footwear", "left_shoe", "right_shoe"}
            cleaned_mask, _, local_bbox = MaskCleanup.clean_mask(
                repair_result.mask,
                fill_holes=fill_holes,
                allowed_mask=allowed_mask,
                blocked_mask=blocked_mask,
                preserve_components=preserve_components,
                min_area=min_area,
            )
            pixel_area = int(np.sum(cleaned_mask > 0))
            if pixel_area < min_area:
                continue

            bx1, by1, bx2, by2 = [int(v) for v in local_bbox]
            if bx2 <= bx1 or by2 <= by1:
                continue

            grabcut_refined = False
            if label in {"top_garment", "bottom_garment", "footwear"}:
                refined_mask = garment_refiner.refine_garment_mask(
                    person_crop_bgr,
                    cleaned_mask,
                    local_bbox,
                    iterations=2,
                    allowed_mask=allowed_mask,
                    blocked_mask=blocked_mask,
                )
                cleaned_mask, _, local_bbox = MaskCleanup.clean_mask(
                    refined_mask,
                    fill_holes=fill_holes,
                    allowed_mask=allowed_mask,
                    blocked_mask=blocked_mask,
                    preserve_components=preserve_components,
                    min_area=min_area,
                )
                pixel_area = int(np.sum(cleaned_mask > 0))
                if pixel_area < min_area:
                    continue
                bx1, by1, bx2, by2 = [int(v) for v in local_bbox]
                if bx2 <= bx1 or by2 <= by1:
                    continue
                grabcut_refined = True

            active_conf = confidence_map[cleaned_mask > 0]
            confidence = float(active_conf.mean()) if active_conf.size else 0.0
            if confidence < 0.18:
                continue

            skin_ratio = self._skin_like_ratio(person_crop_bgr, cleaned_mask)
            if self._should_suppress_part(label, cleaned_mask, local_bbox, crop_h, crop_w, skin_ratio):
                continue

            completion_result = occlusion_repair.complete_occluded_garment(
                label,
                cleaned_mask,
                allowed_mask=allowed_mask,
                blocked_mask=blocked_mask,
            )
            final_mask = completion_result.mask
            completed_bgr = occlusion_repair.inpaint_missing_texture(
                person_crop_bgr,
                cleaned_mask,
                completion_result.missing_mask,
            )

            final_pixel_area = int(np.sum(final_mask > 0))
            if final_pixel_area < min_area:
                continue

            bx1, by1, bw, bh = cv2.boundingRect((final_mask > 0).astype(np.uint8))
            bx2 = bx1 + bw
            by2 = by1 + bh

            rgba_crop = cv2.cvtColor(completed_bgr, cv2.COLOR_BGR2RGBA)
            rgba_crop[:, :, 3] = final_mask
            rgba_crop = rgba_crop[by1:by2, bx1:bx2]

            part_id = f"{job_id}_person_{person_index}_segformer_{label}"
            rgba_crop_path = storage_service.save_crop_rgba(rgba_crop, part_id)

            mask_path = settings.storage_dir / "crops" / f"{part_id}_mask.png"
            cv2.imwrite(str(mask_path), final_mask)

            repair_meta = repair_result.metadata
            parts.append({
                "label": label,
                "rgba_crop_path": storage_service.get_relative_path(rgba_crop_path),
                "mask_path": storage_service.get_relative_path(mask_path),
                "bbox": [bx1 + crop_x1, by1 + crop_y1, bx2 - bx1, by2 - by1],
                "pixel_area": final_pixel_area,
                "grabcut_refined": grabcut_refined,
                "confidence": round(confidence, 4),
                "ingest": label in self.INGEST_LABELS,
                "parser_used": "segformer",
                "skin_like_ratio": round(skin_ratio, 4),
                **repair_meta,
                **completion_result.metadata,
            })

        parts.sort(key=lambda part: part["pixel_area"], reverse=True)
        return parts

    def _skin_like_ratio(self, crop_bgr: np.ndarray, mask: np.ndarray) -> float:
        if not np.any(mask > 0):
            return 0.0

        ycrcb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YCrCb)
        _, cr, cb = cv2.split(ycrcb)
        skin = (cr >= 133) & (cr <= 180) & (cb >= 77) & (cb <= 135)
        active = mask > 0
        return float(np.sum(skin & active) / max(1, np.sum(active)))

    def _should_suppress_part(
        self,
        label: str,
        mask: np.ndarray,
        bbox: List[float],
        crop_h: int,
        crop_w: int,
        skin_ratio: float,
    ) -> bool:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        area_ratio = float(np.sum(mask > 0) / max(1, crop_h * crop_w))
        aspect = width / float(height)
        center_y = (y1 + y2) / 2.0 / max(1, crop_h)

        if label in {"bag", "accessory"}:
            # Hands are commonly mislabeled as purse/bag/accessory by clothes parsers.
            return skin_ratio > 0.30 or area_ratio < 0.025

        if label == "bottom_garment":
            if center_y < 0.45 or height < crop_h * 0.16:
                return True
            if aspect > 3.2 and area_ratio < 0.12:
                return True

        if label == "top_garment":
            if height < crop_h * 0.18 or area_ratio < 0.035:
                return True

        if label == "footwear":
            if center_y < 0.62 or area_ratio < 0.008:
                return True

        return False


segformer_parser = SegFormerParser()
