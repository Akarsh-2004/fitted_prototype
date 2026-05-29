import cv2
import numpy as np
from typing import List, Tuple
from pipeline.config import settings

class MaskCleanup:
    """
    Cleans up raw semantic segmentations using advanced mathematical morphology,
    connected component analysis, hole filling, and contour smoothing/simplification.
    """

    @classmethod
    def fill_holes(cls, mask: np.ndarray) -> np.ndarray:
        """
        Fills internal holes and negative spaces within a binary mask.
        Uses OpenCV's findContours to identify and fill all nested structures.
        """
        if not np.any(mask > 0):
            return mask.copy()
            
        filled = mask.copy()
        contours, hierarchy = cv2.findContours(filled, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours and hierarchy is not None:
            # Draw contours with thickness=-1 to fill all nested internal structures
            for i in range(len(contours)):
                cv2.drawContours(filled, contours, i, 255, -1)
                
        return filled

    @classmethod
    def apply_morphology(cls, mask: np.ndarray, kernel_size: int = None) -> np.ndarray:
        """
        Applies morphological Opening (removes noise) and Closing (bridges gaps).
        Uses a circular structuring element to preserve natural garment curves.
        """
        if kernel_size is None:
            kernel_size = settings.mask_smoothing_kernel
            
        if kernel_size <= 0:
            return mask.copy()
            
        # Ensure odd kernel size
        ksize = kernel_size | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        
        # Opening removes isolated stray noise pixels
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Closing bridges small internal gaps/disconnected threads
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        return closed

    @classmethod
    def filter_connected_components(
        cls,
        mask: np.ndarray,
        min_area: int = None,
        max_components: int = None
    ) -> np.ndarray:
        """
        Runs connected component analysis to remove tiny isolated noise regions and
        optionally isolate the largest contiguous components.
        """
        if min_area is None:
            min_area = settings.min_component_area
            
        if not np.any(mask > 0):
            return mask.copy()
            
        # Run connected components with statistics
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        cleaned = np.zeros_like(mask)
        
        # Background is label 0, start from 1
        components = []
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= min_area:
                components.append((int(area), label))

        components.sort(reverse=True)
        if max_components is not None:
            components = components[:max_components]

        for _, label in components:
            cleaned[labels == label] = 255
                
        return cleaned

    @classmethod
    def smooth_and_simplify(
        cls,
        mask: np.ndarray,
        factor: float = 0.002,
        max_components: int = 1
    ) -> Tuple[np.ndarray, List[List[float]], List[float]]:
        """
        Smooths binary mask contours and simplifies them using the Ramer-Douglas-Peucker (RDP) algorithm.
        Returns the refined binary mask, simplified polygon coordinates, and bounding box [x1, y1, x2, y2].
        """
        h, w = mask.shape[:2]
        refined_mask = np.zeros_like(mask)
        
        if not np.any(mask > 0):
            return refined_mask, [], [0.0, 0.0, 0.0, 0.0]
            
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return refined_mask, [], [0.0, 0.0, 0.0, 0.0]
            
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= settings.min_component_area]
        if not valid_contours:
            return refined_mask, [], [0.0, 0.0, 0.0, 0.0]

        valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:max_components]
        largest_cnt = valid_contours[0]
            
        # 1. Bounding box coordinates
        xs = []
        ys = []
        x2s = []
        y2s = []
        for cnt in valid_contours:
            x, y, gw, gh = cv2.boundingRect(cnt)
            xs.append(x)
            ys.append(y)
            x2s.append(x + gw)
            y2s.append(y + gh)
        bbox = [float(min(xs)), float(min(ys)), float(max(x2s)), float(max(y2s))]
        
        # 2. Polygon simplification (RDP)
        epsilon = factor * cv2.arcLength(largest_cnt, True)
        approx = cv2.approxPolyDP(largest_cnt, epsilon, True)
        polygon = approx.reshape((-1, 2)).astype(float).tolist()
        
        # 3. Redraw smoothed, simplified mask
        cv2.drawContours(refined_mask, [approx], -1, 255, -1)
        for cnt in valid_contours[1:]:
            epsilon_i = factor * cv2.arcLength(cnt, True)
            approx_i = cv2.approxPolyDP(cnt, epsilon_i, True)
            cv2.drawContours(refined_mask, [approx_i], -1, 255, -1)
        
        return refined_mask, polygon, bbox

    @classmethod
    def clean_mask(
        cls,
        mask: np.ndarray,
        *,
        fill_holes: bool = True,
        allowed_mask: np.ndarray = None,
        blocked_mask: np.ndarray = None,
        preserve_components: int = 1,
        min_area: int = None
    ) -> Tuple[np.ndarray, List[List[float]], List[float]]:
        """
        Orchestrates full mask cleanup: hole filling, morphology, size filtering,
        and contour simplification.
        
        Returns:
          - clean_mask: np.ndarray (H, W) cleaned binary mask
          - polygon: List[List[float]] simplified boundary coordinates
          - bbox: List[float] bounding box [x1, y1, x2, y2]
        """
        constrained = mask.copy()
        if allowed_mask is not None:
            constrained = cv2.bitwise_and(constrained, (allowed_mask > 0).astype(np.uint8) * 255)
        if blocked_mask is not None:
            constrained[blocked_mask > 0] = 0

        # Step 1: Fill internal holes when the caller allows it
        step1 = cls.fill_holes(constrained) if fill_holes else constrained
        
        # Step 2: Smooth boundaries via circular opening/closing morphology
        step2 = cls.apply_morphology(step1)
        
        # Step 3: Remove tiny disconnected components
        step3 = cls.filter_connected_components(step2, min_area=min_area, max_components=preserve_components)

        if allowed_mask is not None:
            step3 = cv2.bitwise_and(step3, (allowed_mask > 0).astype(np.uint8) * 255)
        if blocked_mask is not None:
            step3[blocked_mask > 0] = 0
        
        # Step 4: Perform RDP contour simplification and extract bbox/polygon
        clean_mask, polygon, bbox = cls.smooth_and_simplify(step3, max_components=preserve_components)
        
        return clean_mask, polygon, bbox
