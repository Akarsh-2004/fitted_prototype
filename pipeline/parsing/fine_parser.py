import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
from pipeline.config import settings
from pipeline.detectors.schp_parser import human_clothing_parser
from pipeline.detectors.sam2_segmenter import sam2_segmenter
from pipeline.parsing.schp.inference import run_schp_inference
from pipeline.parsing.semantic_mapper import semantic_mapper
from pipeline.parsing.mask_cleanup import MaskCleanup
from pipeline.parsing.garment_refiner import garment_refiner
from pipeline.services.storage_service import storage_service
from pipeline.services.model_registry import model_registry

logger = logging.getLogger("VestirAI.FineParser")

class FineParser:
    """
    Orchestrates advanced Stage 2: Granular Human Parsing from scratch.
    Combines:
      - Stage 1: Isolation & Preprocessing
      - Stage 2: Topological Anatomy / Pose Prior Estimation
      - Stage 3: Semantic Parsing (Transformer/SCHP)
      - Stage 4: Garment Instance Separation & Same-Color Splitter
      - Stage 5: Boundary Refinement (SAM2/GrabCut)
      - Stage 6: Alpha Matting (Guided Image Filtering)
      - Stage 7: Post-Processing Rules
      - Stage 8: Confidence Fusion
    """

    def parse_granular_clothing_layers(
        self,
        img_bgr: np.ndarray,
        person_mask: np.ndarray,
        person_box: List[float],
        job_id: str,
        person_index: int
    ) -> Dict[str, Any]:
        """
        Main entry point for Stage 2 parsing. Structures execution through the 8 stages.
        """
        h_img, w_img = img_bgr.shape[:2]
        px1, py1, px2, py2 = [int(v) for v in person_box]
        person_id = f"person_{job_id}_{person_index}"
        
        dress_rule_applied = False
        bag_detected = False
        parser_used = "ce2p"
        was_split = False

        parts_list = []

        # -------------------------------------------------------------
        # STAGE 1: HUMAN ISOLATION & PREPROCESSING
        # -------------------------------------------------------------
        pad = 8
        crop_x1 = max(0, px1 - pad)
        crop_y1 = max(0, py1 - pad)
        crop_x2 = min(w_img, px2 + pad)
        crop_y2 = min(h_img, py2 + pad)
        
        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1

        if crop_w <= 10 or crop_h <= 10:
            # Fall back directly if crop is invalid
            return self._execute_fallback(img_bgr, person_mask, person_box, job_id, person_index)

        try:
            logger.info("⚡ [Stage 1] Isolating and preprocessing human crop...")
            person_crop_bgr, person_crop_rgb, person_mask_crop = self._stage1_isolate_and_preprocess(
                img_bgr, person_mask, crop_x1, crop_y1, crop_x2, crop_y2
            )

            # -------------------------------------------------------------
            # STAGE 2: TOPOLOGICAL ANATOMY / POSE PRIOR ESTIMATOR
            # -------------------------------------------------------------
            logger.info("⚡ [Stage 2] Estimating topological pose and body structure priors...")
            landmarks, prior_masks = self._stage2_estimate_pose_priors(crop_h, crop_w, person_mask_crop)
            hip_row = landmarks["hip_row"]
            knee_row = landmarks["knee_row"]

            # -------------------------------------------------------------
            # STAGE 3: SEMANTIC HUMAN PARSING (SegFormer / SCHP)
            # -------------------------------------------------------------
            logger.info("⚡ [Stage 3] Executing semantic human parser inference...")
            parsing_result, granular_masks = self._stage3_semantic_parse(
                person_crop_rgb, crop_w, crop_h
            )

            if "dress" in parsing_result.label_masks:
                dress_area = np.sum(parsing_result.label_masks["dress"] > 0)
                if dress_area > (crop_h * crop_w * 0.05):
                    dress_rule_applied = True

            # -------------------------------------------------------------
            # STAGE 4: GARMENT INSTANCE SEPARATION & SAME-COLOR SPLITTER
            # -------------------------------------------------------------
            logger.info("⚡ [Stage 4] Performing instance separation and multi-cue same-color splitting...")
            granular_masks, was_split = self._stage4_garment_instance_split(
                person_crop_bgr, person_mask_crop, granular_masks, landmarks, dress_rule_applied
            )

            # -------------------------------------------------------------
            # STAGES 5 & 6 & 7 & 8: REFINEMENT, MATTING, RULES, AND FUSION PER GARMENT
            # -------------------------------------------------------------
            # GrabCut refine mappings
            GRABCUT_REFINE_MAP = {
                "top_garment": True,
                "outerwear": True,
                "bottom_garment": True,
                "left_shoe": True,
                "right_shoe": True,
                "footwear": True,
                "bag": True,
                "hat": False,
                "left_arm": False,
                "right_arm": False,
                "accessory": False
            }

            for label in list(granular_masks.keys()):
                raw_mask = granular_masks[label]

                # Pre-clean mask (connected components, morphology)
                cleaned_mask, local_poly, local_bbox = MaskCleanup.clean_mask(raw_mask)
                if not np.any(cleaned_mask > 0) or len(local_poly) < 3:
                    continue

                if label in ["top_garment", "outerwear", "bottom_garment", "hat", "bag", "left_shoe", "right_shoe"]:
                    cleaned_mask = self._largest_component_only(cleaned_mask)

                # STAGE 5: SAM2-STYLE BOUNDARY REFINEMENT
                do_grabcut = GRABCUT_REFINE_MAP.get(label, False)
                if was_split and label in ("top_garment", "bottom_garment"):
                    do_grabcut = False # Override waist splits from GrabCut bleeding

                refined_mask = cleaned_mask
                grabcut_refined = False

                if do_grabcut:
                    try:
                        refined_mask = self._stage5_sam2_refine(person_crop_bgr, cleaned_mask, local_bbox)
                        grabcut_refined = True
                    except Exception as e:
                        logger.warning(f"GrabCut refinement failed for {label}: {e}")
                        refined_mask = cleaned_mask

                # Clean post-refinement mask
                refined_mask, final_poly, final_bbox = MaskCleanup.clean_mask(refined_mask)
                if not np.any(refined_mask > 0) or len(final_poly) < 3:
                    continue

                # STAGE 6: GUIDED FILTER EDGE ALPHA MATTING
                logger.info(f"⚡ [Stage 6] Blending soft edges via Guided Filter Matting for {label}...")
                matted_crop_rgba = self._stage6_alpha_matting(person_crop_bgr, refined_mask, final_bbox)

                # STAGE 7: POST-PROCESSING RULES & TOPOLOGY ENGINE
                logger.info(f"⚡ [Stage 7] Applying anatomical and topology rules for {label}...")
                rule_passed = self._stage7_apply_topology_rules(label, refined_mask, crop_h, landmarks)
                if not rule_passed:
                    logger.info(f"Discarded {label} mask due to Stage 7 topology check violation.")
                    continue

                # STAGE 8: MULTI-CUE CONFIDENCE FUSION
                logger.info(f"⚡ [Stage 8] Computing fused confidence score for {label}...")
                confidence = self._stage8_fuse_confidence(
                    cleaned_mask, refined_mask, label, parsing_result.confidence_map, person_crop_bgr, landmarks
                )

                if confidence < 0.45:
                    logger.info(f"Discarded {label} mask due to low fused confidence ({confidence:.2f})")
                    continue

                # Save transparent RGBA crop and mask to global coordinate systems
                global_mask = np.zeros((h_img, w_img), dtype=np.uint8)
                global_mask[crop_y1:crop_y2, crop_x1:crop_x2] = refined_mask

                part_id = f"{job_id}_person_{person_index}_{label}"
                rgba_crop_path = storage_service.save_crop_rgba(matted_crop_rgba, part_id)

                cx1 = max(0, int(person_box[0]))
                cy1 = max(0, int(person_box[1]))
                cx2 = min(w_img, int(person_box[2]))
                cy2 = min(h_img, int(person_box[3]))
                cropped_mask = global_mask[cy1:cy2, cx1:cx2]

                mask_path = settings.storage_dir / "crops" / f"{part_id}_mask.png"
                cv2.imwrite(str(mask_path), cropped_mask)

                bx1, by1, bx2, by2 = [int(v) for v in final_bbox]
                # Shift local final bbox to global coordinates
                gbx1 = bx1 + crop_x1
                gby1 = by1 + crop_y1
                gbw = bx2 - bx1
                gbh = by2 - by1
                bbox_xywh = [gbx1, gby1, gbw, gbh]
                pixel_area = int(np.sum(refined_mask > 0))

                ingest = False if label in ["left_arm", "right_arm"] else True
                if label == "bag":
                    bag_detected = True

                parts_list.append({
                    "label": label,
                    "rgba_crop_path": storage_service.get_relative_path(rgba_crop_path),
                    "mask_path": storage_service.get_relative_path(mask_path),
                    "bbox": bbox_xywh,
                    "pixel_area": pixel_area,
                    "grabcut_refined": grabcut_refined,
                    "confidence": float(round(confidence, 4)),
                    "ingest": ingest,
                    "waist_force_split": was_split if label in ("top_garment", "bottom_garment") else False
                })

            # Detect accessories inside arm regions
            self._detect_wrist_accessories(img_bgr, parts_list, job_id, person_index, person_box)

            # Filter out skin arms
            parts_list = [p for p in parts_list if p["label"] not in ["left_arm", "right_arm"]]

        except Exception as e:
            logger.error(f"❌ [FineParser] Advanced 8-stage parsing failed: {e}. Routing to vertical fallback...")
            return self._execute_fallback(img_bgr, person_mask, person_box, job_id, person_index)

        return {
            "person_id": person_id,
            "parts": parts_list,
            "parser_used": parser_used,
            "dress_rule_applied": dress_rule_applied,
            "bag_detected": bag_detected
        }

    # -------------------------------------------------------------
    # PIPELINE STAGES METHODS
    # -------------------------------------------------------------

    def _stage1_isolate_and_preprocess(
        self, img_bgr: np.ndarray, person_mask: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Stage 1: Extracts person crop, neutralizes the background to neutral gray,
        and runs CLAHE contrast normalization.
        """
        person_crop_bgr = img_bgr[y1:y2, x1:x2].copy()
        person_mask_crop = person_mask[y1:y2, x1:x2]

        # Neutral Gray background neutralization
        person_crop_bgr[person_mask_crop == 0] = 128

        # Apply CLAHE on luminance channel (sharpens inner garments wrinkles/edges)
        lab = cv2.cvtColor(person_crop_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

        return enhanced_bgr, enhanced_rgb, person_mask_crop

    def _stage2_estimate_pose_priors(
        self, crop_h: int, crop_w: int, mask_crop: np.ndarray
    ) -> Tuple[Dict[str, int], Dict[str, np.ndarray]]:
        """
        Stage 2: Topology pose estimator. Uses anatomical proportions to identify
        body landmark vertical lines (shoulder, hip, knee) and prior zone masks.
        """
        # Vertical anatomical proportions
        shoulder_row = int(crop_h * 0.22)
        hip_row = int(crop_h * 0.55)
        knee_row = int(crop_h * 0.82)

        # Build prior masks
        torso_prior = np.zeros((crop_h, crop_w), dtype=np.uint8)
        torso_prior[shoulder_row:hip_row, :] = mask_crop[shoulder_row:hip_row, :]

        legs_prior = np.zeros((crop_h, crop_w), dtype=np.uint8)
        legs_prior[hip_row:knee_row, :] = mask_crop[hip_row:knee_row, :]

        feet_prior = np.zeros((crop_h, crop_w), dtype=np.uint8)
        feet_prior[knee_row:crop_h, :] = mask_crop[knee_row:crop_h, :]

        landmarks = {
            "shoulder_row": shoulder_row,
            "hip_row": hip_row,
            "knee_row": knee_row
        }
        prior_masks = {
            "torso_prior": torso_prior,
            "legs_prior": legs_prior,
            "feet_prior": feet_prior
        }

        return landmarks, prior_masks

    def _stage3_semantic_parse(
        self, crop_rgb: np.ndarray, crop_w: int, crop_h: int
    ) -> Tuple[Any, Dict[str, np.ndarray]]:
        """
        Stage 3: Runs SegFormer or SCHP semantic human parsing model.
        """
        # Prepare crop with short-side upscaling
        TARGET_SHORT_SIDE = 384
        scale = max(1.0, TARGET_SHORT_SIDE / min(crop_h, crop_w))
        
        if scale > 1.0:
            new_h, new_w = int(crop_h * scale), int(crop_w * scale)
            scaled_rgb = cv2.resize(crop_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        else:
            scaled_rgb = crop_rgb.copy()
            scale = 1.0

        # Execute CE2P SCHP Model Inference
        parsing_result = run_schp_inference(scaled_rgb)

        # Scale output maps back to original crop bounds
        if scale > 1.0:
            from pipeline.parsing.schp.inference import ParsingResult
            seg_map = cv2.resize(parsing_result.segmentation_map, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
            conf_map = cv2.resize(parsing_result.confidence_map, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
            label_masks = {}
            for name, mask in parsing_result.label_masks.items():
                label_masks[name] = cv2.resize(mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
            parsing_result = ParsingResult(seg_map, conf_map, label_masks)

        # Run Extended Semantic Mapper to map LIP to target categories
        granular_masks = semantic_mapper.map_to_granular_parts(parsing_result.label_masks)

        return parsing_result, granular_masks

    def _stage4_garment_instance_split(
        self, crop_bgr: np.ndarray, mask_crop: np.ndarray, granular_masks: Dict[str, np.ndarray],
        landmarks: Dict[str, int], dress_rule_applied: bool
    ) -> Tuple[Dict[str, np.ndarray], bool]:
        """
        Stage 4: Instance separation. Uses a vertical Sobel gradient search band
        around the hip row to split merged same-color garments (e.g. black-on-black).
        """
        if dress_rule_applied:
            return granular_masks, False

        top_mask = granular_masks.get("top_garment")
        bottom_mask = granular_masks.get("bottom_garment")
        
        if top_mask is None or bottom_mask is None:
            return granular_masks, False

        crop_h, crop_w = crop_bgr.shape[:2]
        
        # Determine color delta-E between upper and lower halves
        color_dist = self._upper_lower_color_distance(crop_bgr, mask_crop)
        force_split = color_dist < settings.same_color_delta_e_threshold

        t_bin = (top_mask > 0).astype(np.uint8)
        b_bin = (bottom_mask > 0).astype(np.uint8)

        intersection = np.logical_and(t_bin, b_bin).sum()
        union = np.logical_or(t_bin, b_bin).sum()
        iou = intersection / (union + 1e-6)

        trigger = (iou >= settings.waist_split_iou_trigger) or (force_split and (iou >= 0.1 or t_bin.sum() == 0 or b_bin.sum() == 0))

        if union < 500 or not trigger:
            return granular_masks, False

        logger.info(f"Multi-cue same-color splitting active. Color ΔE={color_dist:.1f}, overlap IoU={iou:.2f}")

        combined = np.logical_or(t_bin, b_bin)

        # Establish narrow waist search band around Stage 2 Hip Landmark
        hip_row = landmarks["hip_row"]
        waist_lo = max(1, hip_row - int(crop_h * 0.08))
        waist_hi = min(crop_h - 1, hip_row + int(crop_h * 0.08))

        best_row = hip_row
        max_gradient = -1.0

        # Convert crop to grayscale
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

        for row in range(waist_lo, waist_hi):
            row_pixels = combined[row] > 0
            if row_pixels.sum() < crop_w * 0.2:
                continue
            
            # Compute vertical difference (vertical gradient) to detect belt/waist line edges
            row_gradient = np.abs(
                gray[row].astype(float) - gray[row-1].astype(float)
            )[row_pixels].mean()
            
            if row_gradient > max_gradient:
                max_gradient = row_gradient
                best_row = row

        # Safe fallback: if edge contrast is too low, default strictly to anatomical hip prior
        if max_gradient < 5.0:
            best_row = hip_row

        # Split masks at the optimal gradient row boundary
        corrected_top = np.zeros((crop_h, crop_w), dtype=np.uint8)
        corrected_bottom = np.zeros((crop_h, crop_w), dtype=np.uint8)

        corrected_top[:best_row, :] = combined[:best_row, :]
        corrected_bottom[best_row:, :] = combined[best_row:, :]

        granular_masks["top_garment"] = (corrected_top * 255).astype(np.uint8)
        granular_masks["bottom_garment"] = (corrected_bottom * 255).astype(np.uint8)

        logger.info(f"Successful same-color waist split at row {best_row} (Anatomical hip ratio: {best_row/crop_h:.0%})")
        return granular_masks, True

    def _stage5_sam2_refine(self, img_bgr: np.ndarray, semantic_mask: np.ndarray, bbox: List[float]) -> np.ndarray:
        """
        Stage 5: Boundary Snapping. Uses GrabCut configured as a boundary snapper.
        """
        return garment_refiner.refine_garment_mask(img_bgr, semantic_mask, bbox, iterations=5)

    def _stage6_alpha_matting(self, crop_bgr: np.ndarray, mask: np.ndarray, bbox: List[float]) -> np.ndarray:
        """
        Stage 6: Guided Image Filter Matting Engine. Evaluates mathematical guided filter
        along the mask boundary to generate beautiful premium soft transparency alpha channels.
        """
        h, w = crop_bgr.shape[:2]
        bx1, by1, bx2, by2 = [int(v) for v in bbox]
        
        # 1. Generate grayscale guidance image normalized to [0, 1]
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        I = gray.astype(np.float32) / 255.0
        p = (mask > 0).astype(np.float32)

        # 2. Run high-performance Guided Filter
        q = self._guided_filter(I, p, r=4, eps=1e-4)

        # 3. Create a narrow boundary band (trimap) to apply matting only along edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(mask, kernel, iterations=2)
        eroded = cv2.erode(mask, kernel, iterations=2)
        boundary_band = (dilated > 0) & (eroded == 0)

        # 4. Synthesize final soft alpha channel
        alpha = (p * 255).astype(np.uint8)
        alpha[boundary_band] = (q[boundary_band] * 255).astype(np.uint8)

        # 5. Extract 4-channel BGRA/RGBA crop
        rgba_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGBA)
        rgba_crop[:, :, 3] = alpha
        
        return rgba_crop[by1:by2, bx1:bx2]

    def _guided_filter(self, I: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
        """Standard guided filter implementation."""
        mean_I = cv2.boxFilter(I, -1, (r, r))
        mean_p = cv2.boxFilter(p, -1, (r, r))
        mean_Ip = cv2.boxFilter(I * p, -1, (r, r))
        cov_Ip = mean_Ip - mean_I * mean_p
        
        mean_II = cv2.boxFilter(I * I, -1, (r, r))
        var_I = mean_II - mean_I * mean_I
        
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        
        mean_a = cv2.boxFilter(a, -1, (r, r))
        mean_b = cv2.boxFilter(b, -1, (r, r))
        
        q = mean_a * I + mean_b
        return np.clip(q, 0.0, 1.0)

    def _stage7_apply_topology_rules(
        self, label: str, mask: np.ndarray, crop_h: int, landmarks: Dict[str, int]
    ) -> bool:
        """
        Stage 7: Domain post-processing logic and anatomical topology rules.
        """
        # Centroid check first
        if not self._passes_centroid_check(mask, label, crop_h):
            return False

        ys, _ = np.where(mask > 0)
        if len(ys) == 0:
            return False
            
        min_y, max_y = ys.min(), ys.max()
        knee_row = landmarks["knee_row"]
        hip_row = landmarks["hip_row"]

        # Rule 1: Upper clothes cannot extend below knee unless dress or coat
        if label in ["top_garment"] and max_y > knee_row:
            # Clip upper garment mask strictly at knee level
            mask[knee_row:crop_h, :] = 0
            
        # Rule 2: Shoes must touch lower-leg/feet region (below knee)
        if label in ["left_shoe", "right_shoe", "footwear"] and min_y < hip_row:
            # Shoe centroid must lie in bottom portion of the crop
            centroid_y = ys.mean()
            if centroid_y < knee_row:
                return False

        return True

    def _stage8_fuse_confidence(
        self, raw_mask: np.ndarray, refined_mask: np.ndarray, label: str,
        conf_map: np.ndarray, crop_bgr: np.ndarray, landmarks: Dict[str, int]
    ) -> float:
        """
        Stage 8: Confidence Fusion. Fuses semantic parsing logits, pose prior
        conformity, and GrabCut refinement IoU.
        """
        crop_h = crop_bgr.shape[0]
        
        # 1. Semantic logit confidence
        raw_pixels = conf_map[raw_mask > 0]
        semantic_score = float(raw_pixels.mean()) if len(raw_pixels) > 0 else 0.5

        # 2. Pose conformity score
        ys, _ = np.where(refined_mask > 0)
        if len(ys) == 0:
            return 0.0
            
        knee_row = landmarks["knee_row"]
        hip_row = landmarks["hip_row"]

        if label in ["top_garment", "outerwear"]:
            # Upper garments should be mostly above knees
            above_knee = np.sum(ys < knee_row)
            pose_score = above_knee / len(ys)
        elif label in ["left_shoe", "right_shoe", "footwear"]:
            # Shoes must be below knee
            below_knee = np.sum(ys >= knee_row)
            pose_score = below_knee / len(ys)
        elif label == "bottom_garment":
            # Bottoms should mostly start below hips
            below_hips = np.sum(ys >= hip_row)
            pose_score = below_hips / len(ys)
        else:
            pose_score = 1.0

        # 3. Refinement IoU
        intersection = np.logical_and(raw_mask > 0, refined_mask > 0).sum()
        union = np.logical_or(raw_mask > 0, refined_mask > 0).sum()
        iou_score = intersection / (union + 1e-6)

        # Fused Confidence Fusion formula
        fused = 0.4 * semantic_score + 0.3 * pose_score + 0.3 * iou_score
        return float(np.clip(fused, 0.0, 1.0))

    # -------------------------------------------------------------
    # AUXILIARY UTILITIES (PRESERVING LEGACY COMPATIBILITY)
    # -------------------------------------------------------------

    def _execute_fallback(
        self, img_bgr: np.ndarray, person_mask: np.ndarray, person_box: List[float],
        job_id: str, person_index: int
    ) -> Dict[str, Any]:
        """Routing wrapper for standard anatomical vertical fallback."""
        h_img, w_img = img_bgr.shape[:2]
        logger.info("🔮 [FineParser] Triggering Legact Vertical High-Availability Fallback Parser...")
        
        fallback_layers = human_clothing_parser._parse_clothing_layers_geometric_fallback(
            img_bgr, 
            person_mask, 
            person_box, 
            granular=True
        )

        parts_list = []
        for fl in fallback_layers:
            label = fl["layer_type"]
            raw_mask = fl["mask"]

            clean_mask, local_poly, local_bbox = MaskCleanup.clean_mask(raw_mask)
            if not np.any(clean_mask > 0) or len(local_poly) < 3:
                continue

            crop_rgba = sam2_segmenter.extract_transparent_crop(img_bgr, clean_mask, local_bbox)
            part_id = f"{job_id}_person_{person_index}_{label}"
            rgba_crop_path = storage_service.save_crop_rgba(crop_rgba, part_id)

            cx1 = max(0, int(person_box[0]))
            cy1 = max(0, int(person_box[1]))
            cx2 = min(w_img, int(person_box[2]))
            cy2 = min(h_img, int(person_box[3]))
            cropped_mask = clean_mask[cy1:cy2, cx1:cx2]

            mask_path = settings.storage_dir / "crops" / f"{part_id}_mask.png"
            cv2.imwrite(str(mask_path), cropped_mask)

            bx1, by1, bx2, by2 = [int(v) for v in local_bbox]
            bbox_xywh = [bx1, by1, bx2 - bx1, by2 - by1]
            pixel_area = int(np.sum(clean_mask > 0))

            ingest = False if label in ["left_arm", "right_arm"] else True

            parts_list.append({
                "label": label,
                "rgba_crop_path": storage_service.get_relative_path(rgba_crop_path),
                "mask_path": storage_service.get_relative_path(mask_path),
                "bbox": bbox_xywh,
                "pixel_area": pixel_area,
                "grabcut_refined": False,
                "confidence": 1.0,
                "ingest": ingest,
                "waist_force_split": False
            })

        parts_list = [p for p in parts_list if p["label"] not in ["left_arm", "right_arm"]]

        return {
            "person_id": f"person_{job_id}_{person_index}",
            "parts": parts_list,
            "parser_used": "geometric_fallback",
            "dress_rule_applied": False,
            "bag_detected": False
        }

    def _detect_wrist_accessories(
        self,
        img_bgr: np.ndarray,
        parts_list: List[Dict[str, Any]],
        job_id: str,
        person_index: int,
        person_box: List[float]
    ):
        """Runs the YOLOv8 wrist accessory sub-detector on active arm regions."""
        h_img, w_img = img_bgr.shape[:2]
        
        # Load YOLOv8 accessories detector
        yolo_acc_model = model_registry.get_accessories_model()
        if yolo_acc_model is None:
            return

        # Find arm regions to crop and evaluate
        arm_parts = [p for p in parts_list if p["label"] in ["left_arm", "right_arm"]]

        for arm in arm_parts:
            abx, aby, abw, abh = arm["bbox"]
            
            # Crop each arm region expanded by 15px padding
            pad = 15
            ax1 = max(0, abx - pad)
            ay1 = max(0, aby - pad)
            ax2 = min(w_img, abx + abw + pad)
            ay2 = min(h_img, aby + abh + pad)
            
            arm_crop_w = ax2 - ax1
            arm_crop_h = ay2 - ay1
            
            if arm_crop_w <= 10 or arm_crop_h <= 10:
                continue

            arm_crop = img_bgr[ay1:ay2, ax1:ax2]

            try:
                results = yolo_acc_model.predict(arm_crop, conf=0.6, verbose=False)
                result = results[0]

                if result.boxes is not None:
                    for i, box in enumerate(result.boxes):
                        conf = float(box.conf[0])
                        lx1, ly1, lx2, ly2 = box.xyxy[0].cpu().numpy().tolist()

                        # Remap local box back to global image coordinates
                        gx1 = max(0, int(lx1 + ax1))
                        gy1 = max(0, int(ly1 + ay1))
                        gx2 = min(w_img, int(lx2 + ax1))
                        gy2 = min(h_img, int(ly2 + ay1))
                        
                        gbox = [gx1, gy1, gx2, gy2]
                        gw = gx2 - gx1
                        gh = gy2 - gy1
                        
                        if gw <= 5 or gh <= 5:
                            continue

                        refined_mask = sam2_segmenter.refine_mask_from_box(img_bgr, gbox, iterations=3)
                        
                        final_mask, final_poly, final_bbox = MaskCleanup.clean_mask(refined_mask)
                        if not np.any(final_mask > 0) or len(final_poly) < 3:
                            continue

                        crop_rgba = sam2_segmenter.extract_transparent_crop(img_bgr, final_mask, final_bbox)

                        part_id = f"{job_id}_person_{person_index}_wrist_acc_{i}"
                        rgba_crop_path = storage_service.save_crop_rgba(crop_rgba, part_id)
                        
                        cx1 = max(0, int(person_box[0]))
                        cy1 = max(0, int(person_box[1]))
                        cx2 = min(w_img, int(person_box[2]))
                        cy2 = min(h_img, int(person_box[3]))
                        cropped_mask = final_mask[cy1:cy2, cx1:cx2]

                        mask_path = settings.storage_dir / "crops" / f"{part_id}_mask.png"
                        cv2.imwrite(str(mask_path), cropped_mask)

                        bx1, by1, bx2, by2 = [int(v) for v in final_bbox]
                        bbox_xywh = [bx1, by1, bx2 - bx1, by2 - by1]
                        pixel_area = int(np.sum(final_mask > 0))

                        parts_list.append({
                            "label": "wrist_accessory",
                            "rgba_crop_path": storage_service.get_relative_path(rgba_crop_path),
                            "mask_path": storage_service.get_relative_path(mask_path),
                            "bbox": bbox_xywh,
                            "pixel_area": pixel_area,
                            "grabcut_refined": True,
                            "confidence": conf,
                            "ingest": True
                        })
            except Exception as e:
                logger.warning(f"Wrist accessory detection failed inside arm region: {e}")

    def _upper_lower_color_distance(self, person_bgr: np.ndarray, person_fg_mask: np.ndarray) -> float:
        """
        Returns CIELAB Delta-E distance between upper and lower halves of person.
        < 15.0 = same-color outfit -> waist enforcer should run regardless of CE2P IoU.
        """
        h = person_bgr.shape[0]
        mid = h // 2

        upper_pixels = person_bgr[:mid][person_fg_mask[:mid] > 0]
        lower_pixels = person_bgr[mid:][person_fg_mask[mid:] > 0]

        if len(upper_pixels) < 100 or len(lower_pixels) < 100:
            return 100.0   # can't determine, assume different

        upper_lab = cv2.cvtColor(
            upper_pixels.mean(axis=0).reshape(1,1,3).astype(np.uint8),
            cv2.COLOR_BGR2Lab
        )[0,0]
        lower_lab = cv2.cvtColor(
            lower_pixels.mean(axis=0).reshape(1,1,3).astype(np.uint8),
            cv2.COLOR_BGR2Lab
        )[0,0]

        delta_e = np.sqrt(((upper_lab.astype(float) - lower_lab.astype(float))**2).sum())
        return float(delta_e)

    def _largest_component_only(self, binary_mask: np.ndarray) -> np.ndarray:
        """Eliminates satellite fragments — keeps only the largest connected region."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_mask.astype(np.uint8), connectivity=8
        )
        
        if num_labels <= 1:
            return binary_mask
        
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        return ((labels == largest_label).astype(np.uint8) * 255)

    def _passes_centroid_check(self, mask: np.ndarray, label: str, crop_h: int) -> bool:
        """Ensures the garment's centroid is located in a structurally plausible region of the body."""
        PART_CENTROID_CONSTRAINTS = {
            "top_garment":    {"y_min": 0.05, "y_max": 0.65},
            "outerwear":      {"y_min": 0.05, "y_max": 0.70},
            "bottom_garment": {"y_min": 0.35, "y_max": 0.95},
            "left_shoe":      {"y_min": 0.80, "y_max": 1.00},
            "right_shoe":     {"y_min": 0.80, "y_max": 1.00},
            "hat":            {"y_min": 0.00, "y_max": 0.20},
            "bag":            {"y_min": 0.20, "y_max": 0.85},
        }
        constraint = PART_CENTROID_CONSTRAINTS.get(label)
        if constraint is None:
            return True

        ys, _ = np.where(mask > 0)
        if len(ys) == 0:
            return False

        centroid_y_norm = ys.mean() / crop_h
        passes = constraint["y_min"] <= centroid_y_norm <= constraint["y_max"]
        
        if not passes:
            logger.info(
                f"Centroid sanity FAIL: {label} centroid at "
                f"{centroid_y_norm:.0%} (expected {constraint['y_min']:.0%}–{constraint['y_max']:.0%})"
            )
        return passes

fine_parser = FineParser()
