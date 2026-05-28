import os
import cv2
import numpy as np
import urllib.request
from pathlib import Path
from typing import Tuple
from pipeline.config import settings

class FaceBlurrer:
    def __init__(self):
        # Path to Haar cascade file
        self.cascade_dir = settings.storage_dir / "cascades"
        self.cascade_dir.mkdir(parents=True, exist_ok=True)
        self.cascade_path = self.cascade_dir / "haarcascade_frontalface_default.xml"
        
        self._ensure_cascade_exists()
        self.face_cascade = cv2.CascadeClassifier(str(self.cascade_path))

    def _ensure_cascade_exists(self):
        """Downloads the face cascade XML file if it doesn't exist locally."""
        if not self.cascade_path.exists():
            url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            try:
                print("Downloading face cascade XML for offline face detection...")
                urllib.request.urlretrieve(url, self.cascade_path)
                print("Face cascade XML downloaded successfully.")
            except Exception as e:
                print(f"Error downloading face cascade XML: {e}. Face blurring will fall back to passing.")

    def blur_faces(self, img: np.ndarray, blur_factor: int = 35) -> Tuple[np.ndarray, int]:
        """
        Detects faces in the image and applies a heavy box blur to them.
        Returns the blurred image and the count of faces detected.
        """
        if self.face_cascade.empty():
            print("Warning: Face cascade classifier not loaded. Skipping face blurring.")
            return img.copy(), 0
            
        blurred_img = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = blurred_img[y:y+h, x:x+w]
            
            # Apply strong box blur (ensure kernel size is odd and > 0)
            ksize = (blur_factor | 1)  # makes odd
            blurred_roi = cv2.blur(face_roi, (ksize, ksize))
            
            # Put the blurred face back
            blurred_img[y:y+h, x:x+w] = blurred_roi
            
        return blurred_img, len(faces)

face_blurrer = FaceBlurrer()
