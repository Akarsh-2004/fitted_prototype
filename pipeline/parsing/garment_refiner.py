import cv2
import numpy as np
from typing import List, Tuple
from pipeline.config import settings
from pipeline.detectors.sam2_segmenter import sam2_segmenter

class GarmentRefiner:
    """
    SAM2-style high-fidelity boundary refinement engine.
    Uses GrabCut initialized with SCHP semantic garment masks as priors to snap
    coarse segmentation boundaries perfectly to fine clothing edges.
    """

    def refine_garment_mask(
        self,
        img: np.ndarray,
        semantic_mask: np.ndarray,
        bbox: List[float],
        iterations: int = 5,
        allowed_mask: np.ndarray = None,
        blocked_mask: np.ndarray = None
    ) -> np.ndarray:
        """
        Refines a semantic mask using color-contrast GrabCut segmentation.
        Uses the semantic mask to initialize definite and probable labels.
        
        Returns:
          - refined_mask: np.ndarray of shape (H, W) binary mask (0 or 255)
        """
        h, w = img.shape[:2]
        refined_mask = np.zeros((h, w), dtype=np.uint8)

        if not np.any(semantic_mask > 0):
            return refined_mask

        # 1. Format bounding box and add configurable padding
        pad = settings.sam_refinement_padding
        x1, y1, x2, y2 = [int(v) for v in bbox]
        
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        gw = x2 - x1
        gh = y2 - y1

        if gw <= 0 or gh <= 0:
            return refined_mask

        # 2. Initialize GrabCut label mask
        # Default all pixels inside image to Probable Background (GC_PR_BGD)
        cv_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
        
        # Set absolute background outside our padded bounding box
        cv_mask[0:y1, :] = cv2.GC_BGD
        cv_mask[y2:h, :] = cv2.GC_BGD
        cv_mask[:, 0:x1] = cv2.GC_BGD
        cv_mask[:, x2:w] = cv2.GC_BGD

        # 3. Label foreground regions based on the semantic mask
        # We mark the entire semantic mask as Probable Foreground (GC_PR_FGD)
        cv_mask[semantic_mask > 0] = cv2.GC_PR_FGD

        # Hard geometry constraints: never let GrabCut grow outside the
        # selected support or into known occluders/neighbouring people.
        if allowed_mask is not None:
            cv_mask[allowed_mask == 0] = cv2.GC_BGD
        if blocked_mask is not None:
            cv_mask[blocked_mask > 0] = cv2.GC_BGD

        # To build a definite foreground anchor, we erode the semantic mask
        # this ensures that inner pixels of the garment are locked as GC_FGD
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        eroded_mask = cv2.erode(semantic_mask, kernel, iterations=1)
        cv_mask[eroded_mask > 0] = cv2.GC_FGD
        if allowed_mask is not None:
            cv_mask[allowed_mask == 0] = cv2.GC_BGD
        if blocked_mask is not None:
            cv_mask[blocked_mask > 0] = cv2.GC_BGD

        # Temporary models required by OpenCV GrabCut
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)

        try:
            # 4. Execute GrabCut utilizing our hybrid label mask
            cv2.grabCut(
                img,
                cv_mask,
                (x1, y1, gw, gh),
                bgd_model,
                fgd_model,
                iterations,
                cv2.GC_INIT_WITH_MASK
            )
            
            # Create a clean binary mask where GC_PR_FGD and GC_FGD are positive
            refined_mask = np.where(
                (cv_mask == cv2.GC_FGD) | (cv_mask == cv2.GC_PR_FGD),
                255,
                0
            ).astype(np.uint8)
            
        except Exception as e:
            # Fallback to the original semantic mask if GrabCut fails due to extreme cases
            print(f"[GarmentRefiner] GrabCut refinement error: {e}. Falling back to semantic mask.")
            refined_mask = semantic_mask.copy()

        if allowed_mask is not None:
            refined_mask = cv2.bitwise_and(refined_mask, (allowed_mask > 0).astype(np.uint8) * 255)
        if blocked_mask is not None:
            refined_mask[blocked_mask > 0] = 0

        return refined_mask

    def extract_production_crop(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        bbox: List[float]
    ) -> np.ndarray:
        """
        Wrapper that leverages sam2_segmenter utility to extract a transparent
        RGBA crop from the refined binary mask.
        """
        return sam2_segmenter.extract_transparent_crop(img, mask, bbox)

garment_refiner = GarmentRefiner()
