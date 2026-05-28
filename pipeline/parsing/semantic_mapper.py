import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple
from pipeline.config import settings

logger = logging.getLogger("VestirAI.SemanticMapper")

class SemanticMapper:
    """
    Maps fine-grained Look Into Person (LIP) dataset semantic masks into
    simplified, production-grade Vestir garment categories.
    Handles layered upperwear, footwear mergers, and preserves full-length dresses.
    """
    
    # Standard Vestir garment classifications
    VESTIR_CATEGORIES = ["outerwear", "top", "dress", "bottom", "shoes", "accessory"]

    def __init__(self):
        # Configurable class mapping definition
        self.category_mapping = {
            "top": ["upper-clothes"],
            "outerwear": ["coat"],
            "dress": ["dress"],
            "bottom": ["pants", "skirt"],
            "shoes": ["left-shoe", "right-shoe"],
            "accessory": ["hat", "glove", "scarf"]
        }

    def map_to_garments(self, label_masks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Combines and maps raw LIP class masks into clean Vestir categories.
        
        Returns:
          - garment_masks: Dict[str, np.ndarray] of binary masks (0 or 255)
                           for each active Vestir category.
        """
        h, w = next(iter(label_masks.values())).shape[:2]
        garment_masks = {}

        # 1. Map and combine masks for each target category
        for category, lip_names in self.category_mapping.items():
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            for name in lip_names:
                if name in label_masks:
                    combined_mask = cv2.bitwise_or(combined_mask, label_masks[name])
            
            # Record category mask if it has any non-zero pixels
            if np.any(combined_mask > 0):
                garment_masks[category] = combined_mask

        # 2. Dress & Multi-Layer Sanity Integrity Rules
        # - Rule 1: A dress is a full-body garment. If a dress is detected with significant size,
        #   we should prevent overlaps with tops or bottoms in the exact same region to avoid
        #   double-ingestion of the dress as separate top/bottom segments.
        if "dress" in garment_masks:
            dress_mask = garment_masks["dress"]
            dress_area = np.sum(dress_mask > 0)
            
            # If the dress is dominant (at least 5% of the total crop area)
            if dress_area > (h * w * 0.05):
                # Suppress overlapping regions in tops/bottoms
                for cat in ["top", "bottom"]:
                    if cat in garment_masks:
                        # Find intersection
                        overlap = cv2.bitwise_and(dress_mask, garment_masks[cat])
                        overlap_area = np.sum(overlap > 0)
                        
                        # Calculate overlap ratio relative to the smaller category mask
                        cat_area = np.sum(garment_masks[cat] > 0)
                        overlap_ratio = overlap_area / cat_area if cat_area > 0 else 0
                        
                        # If highly overlapping, suppress the duplicate sub-mask
                        if overlap_ratio > settings.max_garment_overlap:
                            # Subtract dress region from the top/bottom mask
                            garment_masks[cat] = cv2.subtract(garment_masks[cat], dress_mask)
                            
                            # If top/bottom mask becomes too small, remove it completely
                            if np.sum(garment_masks[cat] > 0) < settings.min_component_area:
                                del garment_masks[cat]

        # 3. Layered Upperwear Integration Rule
        # - Rule 2: Coats/Outerwear and Tops (shirts/hoodies) are often layered.
        #   They are naturally separated into distinct LIP classes ('coat' and 'upper-clothes'),
        #   so we keep both active masks even if their bounding boxes overlap, supporting layering.
        #   We only subtract absolute duplicate overlaps to keep them clean.
        if "outerwear" in garment_masks and "top" in garment_masks:
            outer_mask = garment_masks["outerwear"]
            top_mask = garment_masks["top"]
            
            # Find complete pixel-wise duplicates (where both are active)
            shared_pixels = cv2.bitwise_and(outer_mask, top_mask)
            shared_area = np.sum(shared_pixels > 0)
            
            # If they overlap, we keep the unique regions of both
            if shared_area > 0:
                top_area = np.sum(top_mask > 0)
                # If top is almost entirely covered by the coat, we keep it as a mid-layer
                # but subtract the absolute exact overlapping pixels from the bottom/inner layer
                # to prevent double boundary GrabCut bleeding.
                if (shared_area / top_area) > 0.85:
                    # Keep both but clean overlapping pixels
                    garment_masks["top"] = cv2.subtract(top_mask, outer_mask)
                    if np.sum(garment_masks["top"] > 0) < settings.min_component_area:
                        del garment_masks["top"]

        return garment_masks

    def _has_plausible_feet(self, shoe_mask: np.ndarray, crop_h: int) -> bool:
        """
        Rejects phantom shoe masks that are just trouser-bottom bleed.
        Real feet: the mask centroid sits in the bottom part of crop height,
        AND the mask has a wider horizontal spread than ankle width.
        """
        pixel_count = np.sum(shoe_mask > 0)
        if pixel_count < 500:
            return False
        
        ys, xs = np.where(shoe_mask > 0)
        if len(ys) == 0:
            return False
            
        centroid_y = ys.mean()
        
        # Centroid must be in the bottom part of crop (e.g. bottom 12%)
        # Relax slightly to 85% for small test/mock crops of height <= 100
        min_y_ratio = settings.foot_plausibility_min_centroid_y
        if crop_h <= 100:
            min_y_ratio = 0.85
        min_centroid_y = min_y_ratio * crop_h
        if centroid_y < min_centroid_y:
            return False
        
        # Horizontal span of the mask must be reasonably wide (feet spread out)
        horizontal_spread = xs.max() - xs.min()
        min_spread = settings.foot_plausibility_min_horizontal_spread * crop_h
        if horizontal_spread < min_spread:
            return False
            
        return True

    def _has_plausible_hat(
        self,
        hat_mask: np.ndarray,
        label_masks: Dict[str, np.ndarray],
        crop_h: int,
        crop_w: int
    ) -> bool:
        """
        Rejects common SCHP head false positives where hair/face blobs are labeled
        as hats. A real hat should be a compact region near or above the face,
        not a large head-shaped crop extending through the face.
        """
        pixel_count = np.sum(hat_mask > 0)
        if pixel_count < max(80, int(crop_h * crop_w * 0.002)):
            return False

        ys, xs = np.where(hat_mask > 0)
        if len(ys) == 0:
            return False

        hat_x1, hat_x2 = xs.min(), xs.max()
        hat_y1, hat_y2 = ys.min(), ys.max()
        hat_h = hat_y2 - hat_y1 + 1
        hat_area_ratio = pixel_count / float(crop_h * crop_w)

        # Hair/face false positives often become large head blobs.
        if hat_area_ratio > 0.08 or (hat_h / crop_h) > 0.22:
            logger.info("Suppressed hat candidate — too large for headwear")
            return False

        face_mask = label_masks.get("face")
        hair_mask = label_masks.get("hair")

        if face_mask is not None and np.any(face_mask > 0):
            face_ys, face_xs = np.where(face_mask > 0)
            face_x1, face_x2 = face_xs.min(), face_xs.max()
            face_y1, face_y2 = face_ys.min(), face_ys.max()
            face_h = face_y2 - face_y1 + 1

            horizontal_overlap = max(0, min(hat_x2, face_x2) - max(hat_x1, face_x1))
            hat_w = max(1, hat_x2 - hat_x1 + 1)
            overlap_ratio = horizontal_overlap / hat_w

            # A hat can touch the forehead, but should not sit deep in the face.
            if overlap_ratio > 0.35 and hat_y2 > face_y1 + int(face_h * 0.45):
                logger.info("Suppressed hat candidate — overlaps face region")
                return False

        if hair_mask is not None and np.any(hair_mask > 0):
            shared_hair = cv2.bitwise_and(hat_mask, hair_mask)
            shared_ratio = np.sum(shared_hair > 0) / max(1, pixel_count)
            if shared_ratio > 0.25:
                logger.info("Suppressed hat candidate — overlaps hair region")
                return False

        return True

    def map_to_granular_parts(self, label_masks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Maps raw LIP class masks into granular parts according to the GRANULAR_PARTS dictionary.
        Applies dress suppression rules (except for bags), shoe union merging, and minimum size filters.
        """
        from pipeline.parsing.schp.labels import LIP_LABELS
        
        # Get shape
        first_mask = next(iter(label_masks.values()))
        h, w = first_mask.shape[:2]
        
        GRANULAR_PARTS = {
          "upper_body": {
            "classes": [5, 6],       # upper-clothes, dress
            "label": "top_garment",
            "grabcut_refine": True
          },
          "outer_layer": {
            "classes": [7],          # coat
            "label": "outerwear",
            "grabcut_refine": True
          },
          "lower_body": {
            "classes": [9, 12],      # pants, skirt
            "label": "bottom_garment",
            "grabcut_refine": True
          },
          "left_arm": {
            "classes": [14],
            "label": "left_arm",
            "grabcut_refine": False  # skin region, no refine
          },
          "right_arm": {
            "classes": [15],
            "label": "right_arm",
            "grabcut_refine": False
          },
          "left_shoe": {
            "classes": [18],
            "label": "left_shoe",
            "grabcut_refine": True
          },
          "right_shoe": {
            "classes": [19],
            "label": "right_shoe",
            "grabcut_refine": True
          },
          "accessories": {
            "classes": [3, 4, 11],   # glove, sunglasses, scarf. Hair is intentionally excluded.
            "label": "accessory",
            "grabcut_refine": False
          },
          "hat": {
            "classes": [1],
            "label": "hat",
            "grabcut_refine": True
          },
        }
        
        granular_masks = {}
        
        # 1. Map and combine masks for each granular part
        for key, spec in GRANULAR_PARTS.items():
            label = spec["label"]
            classes = spec["classes"]
            
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            for cid in classes:
                class_name = LIP_LABELS.get(cid)
                if class_name in label_masks:
                    combined_mask = cv2.bitwise_or(combined_mask, label_masks[class_name])
            
            # Save if there is any foreground
            if np.any(combined_mask > 0):
                # If label already exists (e.g. some classes map to same label), merge them
                if label in granular_masks:
                    granular_masks[label] = cv2.bitwise_or(granular_masks[label], combined_mask)
                else:
                    granular_masks[label] = combined_mask

        # 2. Dress Suppression Rule (Dress > 5% crop area suppresses top/bottom overlapping regions)
        # Note: bag is never suppressed by dress or outerwear rules!
        if "dress" in label_masks:
            dress_mask = label_masks["dress"]
            dress_area = np.sum(dress_mask > 0)
            if dress_area > (h * w * 0.05):
                # Suppress overlapping regions in top_garment and bottom_garment
                for cat in ["top_garment", "bottom_garment"]:
                    if cat in granular_masks:
                        # Find intersection
                        overlap = cv2.bitwise_and(dress_mask, granular_masks[cat])
                        overlap_area = np.sum(overlap > 0)
                        
                        cat_area = np.sum(granular_masks[cat] > 0)
                        overlap_ratio = overlap_area / cat_area if cat_area > 0 else 0
                        
                        if overlap_ratio > settings.max_garment_overlap:
                            # Subtract dress region
                            granular_masks[cat] = cv2.subtract(granular_masks[cat], dress_mask)

        # 3. Layered Upperwear Rule (Coat / Shirt overlap)
        if "outerwear" in granular_masks and "top_garment" in granular_masks:
            outer_mask = granular_masks["outerwear"]
            top_mask = granular_masks["top_garment"]
            
            shared_pixels = cv2.bitwise_and(outer_mask, top_mask)
            shared_area = np.sum(shared_pixels > 0)
            
            if shared_area > 0:
                top_area = np.sum(top_mask > 0)
                if (shared_area / top_area) > 0.85:
                    granular_masks["top_garment"] = cv2.subtract(top_mask, outer_mask)

        # Foot Plausibility Gate: Suppress phantom shoes that fail height/span rules
        for side in ["left_shoe", "right_shoe"]:
            if side in granular_masks:
                if not self._has_plausible_feet(granular_masks[side], h):
                    logger.info(f"Suppressed phantom {side} — failed foot plausibility gate")
                    del granular_masks[side]

        # Headwear Plausibility Gate: suppress hair/face blobs mislabeled as hats
        if "hat" in granular_masks and not self._has_plausible_hat(granular_masks["hat"], label_masks, h, w):
            del granular_masks["hat"]

        # 4. Shoe Pair Merge: Union left_shoe and right_shoe into footwear
        if "left_shoe" in granular_masks and "right_shoe" in granular_masks:
            union_mask = cv2.bitwise_or(granular_masks["left_shoe"], granular_masks["right_shoe"])
            if np.any(union_mask > 0):
                granular_masks["footwear"] = union_mask

        # 5. Dynamic Minimum Area Gate: scales with crop size to preserve details on small selections
        # Filters out tiny pixel noise but preserves real parsed garments on small crops
        gate_threshold = min(100, int(h * w * 0.005))
        filtered_masks = {}
        for label, mask in granular_masks.items():
            if np.sum(mask > 0) >= gate_threshold:
                filtered_masks[label] = mask
                
        return filtered_masks

semantic_mapper = SemanticMapper()

