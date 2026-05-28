import cv2
import numpy as np
from typing import List, Tuple, Dict, Any

class SAM2SegmenterFallback:
    """
    High-fidelity, responsive boundary refinement engine that mimics SAM2 behaviors
    using local color-contrast GrabCut segmentation.
    This runs entirely locally, requires no heavy GPU downloads, and snaps coarse boundaries
    precisely to garment edges.
    """
    
    def refine_mask_from_box(self, img: np.ndarray, box: List[float], iterations: int = 5) -> np.ndarray:
        """
        Runs OpenCV GrabCut within the bounding box to extract a high-fidelity mask.
        Returns a binary mask (0 or 255) of the same width and height as the image.
        """
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Format bounding box for GrabCut: (x, y, w, h)
        x1, y1, x2, y2 = [int(v) for v in box]
        # Restrict within boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        gw = x2 - x1
        gh = y2 - y1
        
        if gw <= 0 or gh <= 0:
            return mask
            
        grabcut_rect = (x1, y1, gw, gh)
        
        # Temporary buffers required by GrabCut
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)
        
        # Coarse initialization
        cv_mask = np.zeros((h, w), dtype=np.uint8)
        
        try:
            # Run GrabCut
            cv2.grabCut(img, cv_mask, grabcut_rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
            # Create binary mask where GC_PR_FGD (probable foreground) and GC_FGD (foreground) are true
            mask = np.where((cv_mask == cv2.GC_FGD) | (cv_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        except Exception as e:
            print(f"GrabCut refinement failed: {e}. Falling back to standard box mask.")
            # Fallback to simple rectangular box mask
            mask[y1:y2, x1:x2] = 255
            
        return mask

    def refine_mask_from_polygon(self, img: np.ndarray, polygon: List[List[float]]) -> np.ndarray:
        """
        Refines a coarse polygon mask by using GrabCut initialized with the polygon mask.
        """
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        poly_pts = np.array(polygon, dtype=np.int32)
        if len(poly_pts) < 3:
            return mask
            
        # Draw the coarse polygon on our GrabCut mask
        # We label the interior as probable foreground and the polygon boundary as foreground
        cv_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(cv_mask, [poly_pts], cv2.GC_PR_FGD)
        cv2.polylines(cv_mask, [poly_pts], True, cv2.GC_FGD, thickness=2)
        
        # Bounding box of the polygon
        x, y, gw, gh = cv2.boundingRect(poly_pts)
        
        # Add padding
        pad = 10
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + gw + pad)
        y2 = min(h, y + gh + pad)
        
        # Label outside bounding box as absolute background
        cv_mask[0:y1, :] = cv2.GC_BGD
        cv_mask[y2:h, :] = cv2.GC_BGD
        cv_mask[:, 0:x1] = cv2.GC_BGD
        cv_mask[:, x2:w] = cv2.GC_BGD
        
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)
        
        try:
            cv2.grabCut(img, cv_mask, (x1, y1, x2-x1, y2-y1), bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
            mask = np.where((cv_mask == cv2.GC_FGD) | (cv_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        except Exception as e:
            print(f"GrabCut mask refinement failed: {e}. Falling back to standard fillPoly.")
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [poly_pts], 255)
            
        return mask

    def extract_transparent_crop(self, img: np.ndarray, mask: np.ndarray, box: List[float]) -> np.ndarray:
        """
        Cuts out the region defined by the binary mask, makes the background transparent (RGBA),
        and crops to the padded bounding box of the mask.
        Returns a 4-channel RGBA numpy image.
        """
        h, w = img.shape[:2]
        
        # Create 4-channel BGRA image
        bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = mask
        
        # Crop bounds
        x1, y1, x2, y2 = [int(v) for v in box]
        pad = 8
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        
        cropped = bgra[y1:y2, x1:x2]
        
        # Convert BGR to RGB channel order for standard web formats, keeping alpha
        rgba = cv2.cvtColor(cropped, cv2.COLOR_BGRA2RGBA)
        return rgba

sam2_segmenter = SAM2SegmenterFallback()
