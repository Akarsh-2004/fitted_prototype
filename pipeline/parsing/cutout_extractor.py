import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List
from pipeline.config import settings
from pipeline.services.storage_service import storage_service
from pipeline.parsing.segformer_cutout import segformer_cutout
from pipeline.parsing.grabcut_cutout import grabcut_cutout

logger = logging.getLogger("VestirAI.CutoutExtractor")

class CutoutExtractor:
    """
    Central orchestrator for extracting a high-fidelity transparent person cutout.
    Attempts primary SegFormer-B2 segmentation, automatically falling back
    to YOLO-seeded GrabCut on any failure or low mask coverage.
    """

    def extract_person_cutout(
        self,
        img_bgr: np.ndarray,
        bbox_xyxy: List[float],
        polygon: List[List[float]],
        job_id: str,
        person_index: int
    ) -> Dict[str, Any]:
        """
        Extracts a clean transparent RGBA person crop and simplifies the selection
        polygon to exactly 12 points.
        """
        # Ensure target storage folders exist
        cutout_dir = settings.storage_dir / "cutouts" / job_id
        cutout_dir.mkdir(parents=True, exist_ok=True)

        # File paths inside settings.storage_dir
        rgba_crop_filename = f"person_{person_index}.png"
        mask_filename = f"person_{person_index}_mask.png"
        
        rgba_crop_abs = cutout_dir / rgba_crop_filename
        mask_abs = cutout_dir / mask_filename

        # 1. Attempt Primary SegFormer path
        logger.info("🔮 [CutoutExtractor] Attempting primary SegFormer person cutout extraction...")
        segformer_result = segformer_cutout.extract_cutout(img_bgr, bbox_xyxy)
        
        if segformer_result is not None:
            rgba_crop, mask, coverage_ratio = segformer_result
            method_used = "segformer"
            logger.info("✅ [CutoutExtractor] SegFormer cutout successful.")
        else:
            # 2. Fall back to GrabCut path
            logger.warning("⚠️ [CutoutExtractor] SegFormer failed or disabled. Activating GrabCut fallback...")
            rgba_crop, mask, coverage_ratio = grabcut_cutout.extract_cutout(img_bgr, bbox_xyxy, polygon)
            method_used = "grabcut"
            logger.info("✅ [CutoutExtractor] Fallback GrabCut cutout completed successfully.")

        # Save transparent RGBA crop (CV expects BGRA for writing)
        rgba_bgra = cv2.cvtColor(rgba_crop, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(str(rgba_crop_abs), rgba_bgra)

        # Save binary mask for debug
        cv2.imwrite(str(mask_abs), mask)

        # 3. Simplify polygon to exactly 12 points for canvas selection dash-stroke overlays
        simplified_poly = self._simplify_to_12_points(polygon)

        # Calculate bounding box in [x, y, w, h] format
        px1, py1, px2, py2 = [int(v) for v in bbox_xyxy]
        bbox_xywh = [px1, py1, px2 - px1, py2 - py1]

        # Convert absolute paths to relative URLs
        rgba_crop_relative = storage_service.get_relative_path(rgba_crop_abs)
        mask_relative = storage_service.get_relative_path(mask_abs)

        return {
            "rgba_crop_path": rgba_crop_relative,
            "mask_path": mask_relative,
            "coverage_ratio": float(round(coverage_ratio, 4)),
            "method_used": method_used,
            "bbox": bbox_xywh,
            "contour_polygon": simplified_poly
        }

    def _simplify_to_12_points(self, polygon: List[List[float]]) -> List[List[float]]:
        """Downsamples a polygon to exactly 12 points evenly spaced."""
        if len(polygon) <= 12:
            # Pad with last element to reach exactly 12 points if too few
            padded = list(polygon)
            while len(padded) < 12 and len(padded) > 0:
                padded.append(padded[-1])
            return padded if padded else [[0.0, 0.0]] * 12
            
        indices = np.linspace(0, len(polygon) - 1, 12, dtype=int)
        return [polygon[i] for i in indices]

cutout_extractor = CutoutExtractor()
