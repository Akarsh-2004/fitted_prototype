import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from pipeline.config import settings

logger = logging.getLogger("VestirAI.ModelRegistry")

class ModelRegistry:
    """
    Lazy-load singleton model registry.
    Manages SegFormer-B2 cutout and YOLOv8 accessories model lifecycles,
    caching models locally and incorporating connection error safeguards
    for offline high-availability.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ModelRegistry, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self.segformer_model = None
            self.segformer_processor = None
            self.segformer_loaded = False

            self.accessories_model = None
            self.accessories_loaded = False

            # Model cache paths
            self.segformer_cache_dir = settings.base_dir / "data" / "models" / "segformer"
            self.segformer_cache_dir.mkdir(parents=True, exist_ok=True)

    def get_segformer(self) -> Optional[tuple]:
        """
        Lazily loads and returns (processor, model) for SegFormer-B2.
        If loading fails (offline mode / missing connection), logs warning and returns None.
        """
        if self.segformer_loaded and self.segformer_model is not None:
            return self.segformer_processor, self.segformer_model

        try:
            logger.info("🔮 [ModelRegistry] Lazy-loading SegFormer-B2 clothes parser model...")
            from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
            
            model_id = "mattmdjaga/segformer_b2_clothes"
            
            # Load processor
            self.segformer_processor = SegformerImageProcessor.from_pretrained(
                model_id,
                cache_dir=str(self.segformer_cache_dir),
                local_files_only=False
            )
            
            # Load model
            self.segformer_model = SegformerForSemanticSegmentation.from_pretrained(
                model_id,
                cache_dir=str(self.segformer_cache_dir),
                local_files_only=False
            )
            
            self.segformer_loaded = True
            logger.info("🎉 [ModelRegistry] SegFormer-B2 loaded successfully and warmed up!")
            return self.segformer_processor, self.segformer_model
            
        except Exception as e:
            logger.warning(
                f"⚠️ [ModelRegistry] SegFormer-B2 load failed: {e}. "
                "Hugging Face hub may be unreachable or offline. Falling through to local GrabCut..."
            )
            # Try to load locally if cached
            try:
                from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
                model_id = "mattmdjaga/segformer_b2_clothes"
                self.segformer_processor = SegformerImageProcessor.from_pretrained(
                    model_id,
                    cache_dir=str(self.segformer_cache_dir),
                    local_files_only=True
                )
                self.segformer_model = SegformerForSemanticSegmentation.from_pretrained(
                    model_id,
                    cache_dir=str(self.segformer_cache_dir),
                    local_files_only=True
                )
                self.segformer_loaded = True
                logger.info("🎉 [ModelRegistry] SegFormer-B2 loaded successfully from local offline cache!")
                return self.segformer_processor, self.segformer_model
            except Exception as inner_e:
                logger.warning(f"⚠️ [ModelRegistry] Local offline cache load also failed: {inner_e}.")
            
            self.segformer_loaded = False
            return None

    def get_accessories_model(self) -> Optional[Any]:
        """
        Lazily loads and returns the YOLOv8 accessories detector.
        If loading fails (offline mode / missing connection), logs warning and returns None.
        """
        if self.accessories_loaded and self.accessories_model is not None:
            return self.accessories_model

        try:
            logger.info("🔮 [ModelRegistry] Lazy-loading YOLOv8 accessories model...")
            from ultralytics import YOLO
            
            model_id = "akahana/yolov8n-accessories"
            self.accessories_model = YOLO(model_id)
            self.accessories_loaded = True
            logger.info("🎉 [ModelRegistry] YOLOv8 accessories model loaded and warmed up!")
            return self.accessories_model
            
        except Exception as e:
            logger.warning(
                f"⚠️ [ModelRegistry] YOLOv8 accessories model load failed: {e}. "
                "Hugging Face/Ultralytics Hub unreachable. Gracefully skipping wrist accessory detection..."
            )
            self.accessories_loaded = False
            return None

    def get_status(self) -> Dict[str, str]:
        """Returns the warm/cold state of registered models."""
        return {
            "segformer": "warm" if self.segformer_loaded else "cold",
            "accessories": "warm" if self.accessories_loaded else "cold"
        }

# Global singleton registry instance
model_registry = ModelRegistry()
