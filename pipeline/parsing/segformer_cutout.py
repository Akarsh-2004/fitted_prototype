import cv2
import numpy as np
import torch
import torch.nn.functional as F
import logging
from typing import Tuple, Optional
from pipeline.services.model_registry import model_registry

logger = logging.getLogger("VestirAI.SegFormerCutout")

class SegFormerCutout:
    """
    Primary person cutout extractor using SegFormer-B2.
    Segments the human and merges all non-background clothing/skin classes
    into a clean foreground cutout mask.
    """
    
    def extract_cutout(self, img_bgr: np.ndarray, bbox_xyxy: list) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Executes SegFormer semantic human segmentation on a bounding box region.
        
        Returns:
          - rgba_crop: np.ndarray (H_crop, W_crop, 4) or None if fails confidence gate.
          - binary_mask: np.ndarray (H_crop, W_crop) binary mask (0 or 255)
          - coverage_ratio: float
        """
        # 1. Retrieve lazy-loaded SegFormer
        segformer_bundle = model_registry.get_segformer()
        if segformer_bundle is None:
            logger.warning("SegFormer model not available (offline/failed). Skipping to fallback.")
            return None

        processor, model = segformer_bundle

        # 2. Extract tight crop from image
        h_img, w_img = img_bgr.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_img, x2)
        y2 = min(h_img, y2)
        
        crop_w = x2 - x1
        crop_h = y2 - y1
        
        if crop_w <= 10 or crop_h <= 10:
            logger.warning(f"Crop box too small: {crop_w}x{crop_h}. Skipping cutout.")
            return None

        crop_bgr = img_bgr[y1:y2, x1:x2].copy()
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        # 3. Process image and run SegFormer inference
        try:
            device = next(model.parameters()).device
            
            # SegFormer expects 512x512 by default
            inputs = processor(images=crop_rgb, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits  # Shape: [1, num_labels, 128, 128]

            # 4. Upsample logits back to original crop resolution
            upsampled_logits = F.interpolate(
                logits,
                size=(crop_h, crop_w),
                mode="bilinear",
                align_corners=False
            )
            
            # Argmax along class channel dimension
            preds = upsampled_logits.argmax(dim=1).squeeze(0)  # Shape [crop_h, crop_w]
            seg_map = preds.cpu().numpy().astype(np.uint8)

            # 5. Merge all non-background classes into a single foreground mask
            # Class 0 is background. All classes > 0 comprise the person.
            mask = (seg_map > 0).astype(np.uint8) * 255

            # 6. Apply post-processing (CLOSE morphology to fill holes)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask_refined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            # 7. Confidence Gate: Coverage ratio check
            total_pixels = crop_h * crop_w
            fg_pixels = np.sum(mask_refined > 0)
            coverage_ratio = float(fg_pixels / total_pixels)

            if coverage_ratio < 0.05:
                logger.warning(f"SegFormer mask coverage too low: {coverage_ratio:.2%}. Treating as failed.")
                return None

            # Create RGBA crop
            crop_rgba = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGBA)
            crop_rgba[:, :, 3] = mask_refined

            return crop_rgba, mask_refined, coverage_ratio

        except Exception as e:
            logger.error(f"SegFormer inference failed with error: {e}")
            return None

segformer_cutout = SegFormerCutout()
