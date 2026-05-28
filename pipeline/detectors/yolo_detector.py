import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from ultralytics import YOLO
from pipeline.config import settings

class YoloDetector:
    def __init__(self):
        self.model = None

    def get_model(self) -> YOLO:
        """Lazy loader for YOLO11-seg model."""
        if self.model is None:
            model_path = settings.yolo_model_path
            try:
                self.model = YOLO(model_path)
            except Exception as e:
                print(f"Error loading YOLO model from {model_path}: {e}. Trying auto-download of yolo11s-seg.pt...")
                self.model = YOLO("yolo11s-seg.pt")
        return self.model

    def detect_people(self, img: np.ndarray, conf_threshold: float = 0.25) -> List[Dict[str, Any]]:
        """
        Runs YOLO11 segmentation filtering specifically for humans (class 0).
        """
        yolo_model = self.get_model()
        results = yolo_model.predict(img, classes=[0], conf=conf_threshold, verbose=False)
        result = results[0]
        
        people = []
        if result.masks is not None and len(result.masks.xy) > 0:
            boxes = result.boxes
            masks = result.masks
            for i in range(len(boxes)):
                box = boxes[i]
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
                poly = masks.xy[i]  # shape (N, 2)
                
                # Simplify polygon using Ramer-Douglas-Peucker
                simplified_poly = self._simplify_polygon(poly)
                
                people.append({
                    "id": i,
                    "confidence": conf,
                    "box": xyxy,
                    "polygon": simplified_poly,
                    "label": "person",
                    "class_id": 0
                })
        return people

    def detect_flat_lay_garments(self, img: np.ndarray, conf_threshold: float = 0.15) -> List[Dict[str, Any]]:
        """
        Detects flat lay garments using a hybrid approach:
        1. Class-agnostic YOLO11 segmentation (filtering out people)
        2. High-contrast Otsu visual contour finder for maximum reliability
        Detections are combined and returned.
        """
        h, w = img.shape[:2]
        garments = []
        
        # 1. Class-agnostic YOLO detections
        try:
            yolo_model = self.get_model()
            results = yolo_model.predict(img, conf=conf_threshold, verbose=False)
            result = results[0]
            
            if result.masks is not None and len(result.masks.xy) > 0:
                boxes = result.boxes
                masks = result.masks
                for i in range(len(boxes)):
                    box = boxes[i]
                    class_id = int(box.cls[0])
                    # Skip people in flat lay
                    if class_id == 0:
                        continue
                        
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    poly = masks.xy[i]
                    
                    simplified_poly = self._simplify_polygon(poly)
                    garments.append({
                        "confidence": conf,
                        "box": xyxy,
                        "polygon": simplified_poly,
                        "label": "garment",
                        "class_id": class_id,
                        "source": "yolo"
                    })
        except Exception as e:
            print(f"YOLO flat lay inference error: {e}. Falling back entirely to classic CV detector.")

        # 2. Classic CV contrast-based detector (essential for zero-failure on random clothing items)
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Apply Gaussian Blur to smooth texture
            blurred = cv2.GaussianBlur(gray, (9, 9), 0)
            
            # Use Otsu's thresholding to separate foreground from background
            # Assumes a relatively flat/uniform background
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # If the image background is dark, Otsu INV might select the background.
            # We check the average color along the outer boundary to verify if we need to invert.
            boundary_pixels = np.concatenate([thresh[0, :], thresh[-1, :], thresh[:, 0], thresh[:, -1]])
            if np.mean(boundary_pixels) > 127:
                thresh = cv2.bitwise_not(thresh)
                
            # Perform morphological operations to clean up masks
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                # Ignore very small spots (noise)
                min_area = (h * w) * 0.005 # at least 0.5% of the image size
                if area < min_area:
                    continue
                    
                # Bounding box
                x, y, gw, gh = cv2.boundingRect(cnt)
                box = [float(x), float(y), float(x + gw), float(y + gh)]
                
                # Simplify polygon
                epsilon = 0.002 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                poly_list = approx.reshape((-1, 2)).astype(float).tolist()
                
                # Bounding box center and confidence (size relative to image)
                conf = float(area / (h * w))
                
                garments.append({
                    "confidence": conf,
                    "box": box,
                    "polygon": poly_list,
                    "label": "garment",
                    "class_id": -1,
                    "source": "cv"
                })
        except Exception as e:
            print(f"Classic CV flat lay detector error: {e}")

        return garments

    def _simplify_polygon(self, poly: np.ndarray, factor: float = 0.003) -> List[List[float]]:
        """Simplifies a 2D numpy polygon using Ramer-Douglas-Peucker algorithm."""
        if len(poly) > 2:
            poly_reshaped = poly.astype(np.float32).reshape((-1, 1, 2))
            epsilon = factor * cv2.arcLength(poly_reshaped, True)
            approx = cv2.approxPolyDP(poly_reshaped, epsilon, True)
            return approx.reshape((-1, 2)).tolist()
        return poly.tolist()

yolo_detector = YoloDetector()
