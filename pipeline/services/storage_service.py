import os
import cv2
import uuid
import numpy as np
from pathlib import Path
from typing import Tuple, Union
from pipeline.config import settings

class StorageService:
    def __init__(self):
        self.storage_dir = settings.storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories for organized storage
        self.uploads_dir = self.storage_dir / "uploads"
        self.crops_dir = self.storage_dir / "crops"
        
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.crops_dir.mkdir(parents=True, exist_ok=True)

    def save_upload(self, file_bytes: bytes, original_filename: str) -> Path:
        """Saves an uploaded raw image to the uploads folder with a unique name."""
        ext = Path(original_filename).suffix or ".jpg"
        unique_name = f"{uuid.uuid4()}{ext}"
        target_path = self.uploads_dir / unique_name
        
        with open(target_path, "wb") as f:
            f.write(file_bytes)
            
        return target_path

    def save_crop_rgba(self, image_rgba: np.ndarray, item_id: str) -> Path:
        """Saves a 4-channel RGBA numpy image as a transparent PNG crop."""
        filename = f"{item_id}.png"
        target_path = self.crops_dir / filename
        
        # OpenCV expects BGRA for 4-channel image writing
        image_bgra = cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(str(target_path), image_bgra)
        
        return target_path

    def load_image(self, path: Union[str, Path]) -> np.ndarray:
        """Helper to read an image file using OpenCV."""
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not load image at path: {path}")
        return img

    def get_relative_path(self, absolute_path: Path) -> str:
        """Converts an absolute path inside settings.base_dir into a relative URL/path."""
        try:
            return str(absolute_path.relative_to(settings.base_dir))
        except ValueError:
            return str(absolute_path)

storage_service = StorageService()
