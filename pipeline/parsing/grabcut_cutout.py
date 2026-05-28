import cv2
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger("VestirAI.GrabCutCutout")

class GrabCutCutout:
    """
    Fallback person cutout extractor.
    Uses OpenCV GrabCut initialized with the cached YOLO11 person polygon
    to segment and isolate the human.
    """

    def extract_cutout(self, img_bgr: np.ndarray, bbox_xyxy: list, polygon: list) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Executes OpenCV GrabCut utilizing the polygon and bounding box.
        
        Returns:
          - rgba_crop: np.ndarray (H_crop, W_crop, 4) crop isolated on original bbox bounds.
          - binary_mask: np.ndarray (H_crop, W_crop) binary mask (0 or 255)
          - coverage_ratio: float
        """
        h_img, w_img = img_bgr.shape[:2]
        px1, py1, px2, py2 = [int(v) for v in bbox_xyxy]
        
        # 1. Expand bbox by 20px padding for absolute background border
        pad = 20
        x1 = max(0, px1 - pad)
        y1 = max(0, py1 - pad)
        x2 = min(w_img, px2 + pad)
        y2 = min(h_img, py2 + pad)
        
        crop_w = x2 - x1
        crop_h = y2 - y1

        # Crop padded image region
        crop_bgr = img_bgr[y1:y2, x1:x2].copy()

        # 2. Create GrabCut initialization mask
        # Default all pixels to Probable Background (GC_PR_BGD)
        init_mask = np.full((crop_h, crop_w), cv2.GC_PR_BGD, dtype=np.uint8)

        # 3. Fill polygon interior with Probable Foreground (GC_PR_FGD)
        # Shift polygon coordinates to crop-local coordinates
        local_poly = [[pt[0] - x1, pt[1] - y1] for pt in polygon]
        pts = np.array(local_poly, dtype=np.int32).reshape((-1, 1, 2))
        
        if len(pts) >= 3:
            cv2.fillPoly(init_mask, [pts], cv2.GC_PR_FGD)

        # 4. Erode polygon by 10px to create Definite Foreground (GC_FGD)
        poly_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
        if len(pts) >= 3:
            cv2.fillPoly(poly_mask, [pts], 255)
        
        # 3x3 kernel eroded 10 times is a 10px boundary erosion
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        eroded_poly_mask = cv2.erode(poly_mask, erode_kernel, iterations=10)
        init_mask[eroded_poly_mask > 0] = cv2.GC_FGD

        # 5. Label absolute background outside the original bbox as GC_BGD
        local_px1 = max(0, px1 - x1)
        local_py1 = max(0, py1 - y1)
        local_px2 = min(crop_w, px2 - x1)
        local_py2 = min(crop_h, py2 - y1)

        init_mask[0:local_py1, :] = cv2.GC_BGD
        init_mask[local_py2:crop_h, :] = cv2.GC_BGD
        init_mask[:, 0:local_px1] = cv2.GC_BGD
        init_mask[:, local_px2:crop_w] = cv2.GC_BGD

        # Temporary GrabCut storage models
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)

        # 6. Run cv2.grabCut
        try:
            cv2.grabCut(
                crop_bgr,
                init_mask,
                None,
                bgd_model,
                fgd_model,
                iterCount=7,
                mode=cv2.GC_INIT_WITH_MASK
            )
        except Exception as e:
            logger.warning(f"GrabCut cutout optimization failed: {e}. Falling back to polygon interior.")
            # Set init_mask based on direct polygon mask
            init_mask = np.where(poly_mask > 0, cv2.GC_PR_FGD, cv2.GC_BGD).astype(np.uint8)

        # 7. Extract final foreground mask
        # 1 and 3 are GC_FGD and GC_PR_FGD respectively
        mask2 = np.where((init_mask == cv2.GC_FGD) | (init_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

        # 8. Apply as alpha channel and crop back to the original unpadded bbox coordinates
        crop_rgba = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGBA)
        crop_rgba[:, :, 3] = mask2

        # Crop to original unpadded bounding box
        final_crop_rgba = crop_rgba[local_py1:local_py2, local_px1:local_px2]
        final_mask = mask2[local_py1:local_py2, local_px1:local_px2]

        # Calculate coverage ratio relative to the unpadded bbox
        total_pixels = final_mask.size
        fg_pixels = np.sum(final_mask > 0)
        coverage_ratio = float(fg_pixels / total_pixels) if total_pixels > 0 else 0.0

        return final_crop_rgba, final_mask, coverage_ratio

grabcut_cutout = GrabCutCutout()
