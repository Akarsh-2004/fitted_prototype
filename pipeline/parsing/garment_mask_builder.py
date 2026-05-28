import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from pipeline.config import settings
from pipeline.parsing.schp.inference import run_schp_inference
from pipeline.parsing.semantic_mapper import semantic_mapper
from pipeline.parsing.mask_cleanup import MaskCleanup
from pipeline.parsing.garment_refiner import garment_refiner

class GarmentMaskBuilder:
    """
    Central orchestrator of the SCHP + SAM2 hybrid human parsing pipeline.
    Responsible for extracting tight crops, executing neural inference,
    semantic mapping, and applying SAM2/GrabCut boundary perfection.
    """

    def parse_garments(
        self,
        img: np.ndarray,
        person_mask: np.ndarray,
        person_box: List[float],
        job_id: str = "temp_job"
    ) -> List[Dict[str, Any]]:
        """
        Runs the full semantic human parsing pipeline on an isolated person region.
        
        Returns a list of parsed garment layers matching the downstream interface:
          - layer_type: 'upper' | 'lower' | 'shoes' | 'accessory'
          - box: [gx1, gy1, gx2, gy2] in global coordinates
          - polygon: List[List[float]] in global coordinates
          - mask: binary mask in global image size
        """
        h, w = img.shape[:2]
        
        # 1. Determine padding-matched crop offset boundaries in the original image
        px1, py1, px2, py2 = [int(v) for v in person_box]
        pad = 8
        crop_x1 = max(0, px1 - pad)
        crop_y1 = max(0, py1 - pad)
        crop_x2 = min(w, px2 + pad)
        crop_y2 = min(h, py2 + pad)
        
        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1
        
        if crop_w <= 10 or crop_h <= 10:
            return []

        # 2. Extract tight person crop (RGB format for SCHP inference)
        person_crop_bgr = img[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        
        # Apply the person binary mask inside the crop to ensure background is clean/zeroed
        person_mask_crop = person_mask[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # Zero-out pixels outside the person mask to eliminate background clutter
        person_crop_bgr[person_mask_crop == 0] = 0
        person_crop_rgb = cv2.cvtColor(person_crop_bgr, cv2.COLOR_BGR2RGB)

        # 3. Run SCHP Model Inference on the isolated crop
        parsing_result = run_schp_inference(person_crop_rgb)
        
        # 4. Map LIP classes into target Vestir categories
        garment_masks = semantic_mapper.map_to_garments(parsing_result.label_masks)
        
        # Optional: Save visualization debug exports
        if settings.debug_exports_enabled:
            self._save_debug_visualizations(
                person_crop_rgb, 
                parsing_result, 
                garment_masks, 
                job_id
            )

        layers = []

        # 5. Process and refine each category mask individually
        for category, raw_mask in garment_masks.items():
            # Clean up the semantic mask first (holes, morphology, min components)
            clean_mask, local_poly, local_bbox = MaskCleanup.clean_mask(raw_mask)
            
            # Skip if cleaned mask is empty or too tiny
            if not np.any(clean_mask > 0) or len(local_poly) < 3:
                continue
                
            # Run GrabCut (SAM2 fallback emulation) to refine boundaries to perfection
            refined_mask = garment_refiner.refine_garment_mask(
                person_crop_bgr, 
                clean_mask, 
                local_bbox
            )
            
            # Clean up GrabCut output and extract final polygon and bounding box
            final_refined_mask, final_poly, final_bbox = MaskCleanup.clean_mask(refined_mask)
            
            if not np.any(final_refined_mask > 0) or len(final_poly) < 3:
                continue

            # 6. Remap outputs back to global full-size coordinate systems
            # Global bounding box [gx1, gy1, gx2, gy2]
            gx1 = final_bbox[0] + crop_x1
            gy1 = final_bbox[1] + crop_y1
            gx2 = final_bbox[2] + crop_x1
            gy2 = final_bbox[3] + crop_y1
            
            global_bbox = [gx1, gy1, gx2, gy2]

            # Global polygon points
            global_poly = [[pt[0] + crop_x1, pt[1] + crop_y1] for pt in final_poly]

            # Global binary mask of full image size
            global_mask = np.zeros((h, w), dtype=np.uint8)
            global_mask[crop_y1:crop_y2, crop_x1:crop_x2] = final_refined_mask

            # Map category back to standard downstream layer hint
            # 'top', 'outerwear', 'dress' -> 'upper'
            # 'bottom' -> 'lower'
            # 'shoes' -> 'shoes'
            # 'accessory' -> 'accessory'
            if category in ["top", "outerwear", "dress"]:
                layer_type = "upper"
            elif category == "bottom":
                layer_type = "lower"
            else:
                layer_type = category  # 'shoes' or 'accessory'

            layers.append({
                "layer_type": layer_type,
                "box": global_bbox,
                "polygon": global_poly,
                "mask": global_mask,
                "category_tag": category  # extra metadata for tracking/tagging
            })

        return layers

    def _save_debug_visualizations(
        self,
        crop_rgb: np.ndarray,
        parsing_result: Any,
        garment_masks: Dict[str, np.ndarray],
        job_id: str
    ):
        """Saves optional debug visualization images into data/storage/debug/"""
        debug_dir = settings.storage_dir / "debug" / job_id
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)

        # 1. Save raw person crop
        cv2.imwrite(str(debug_dir / "crop_input.jpg"), crop_bgr)

        # 2. Save segmentation map colored overlay
        h, w = crop_rgb.shape[:2]
        seg_overlay = crop_bgr.copy()
        
        # Color palette for 20 classes
        np.random.seed(42)
        colors = np.random.randint(0, 255, (20, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0] # Background is black
        
        for i in range(20):
            mask = (parsing_result.segmentation_map == i)
            # Safe, high-performance NumPy blending (50% blend)
            seg_overlay[mask] = (seg_overlay[mask] * 0.5 + colors[i] * 0.5).astype(np.uint8)
            
        cv2.imwrite(str(debug_dir / "schp_segmentation.png"), seg_overlay)

        # 3. Save mapped garment category masks
        for cat, mask in garment_masks.items():
            cv2.imwrite(str(debug_dir / f"mask_raw_{cat}.png"), mask)

        # 4. Save confidence heatmap
        conf_heatmap = (parsing_result.confidence_map * 255).astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(conf_heatmap, cv2.COLORMAP_JET)
        cv2.imwrite(str(debug_dir / "confidence_heatmap.png"), colored_heatmap)

garment_mask_builder = GarmentMaskBuilder()
