import os
import logging
import shutil
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

    def _prepare_segformer_model(self) -> None:
        """Move SegFormer to the best available device and set inference mode."""
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.segformer_model.to(device)
        except Exception as exc:
            logger.warning(f"⚠️ [ModelRegistry] SegFormer device placement skipped: {exc}")
        self.segformer_model.eval()

    def _materialize_hf_snapshot_pointers(self, repo_name: str) -> None:
        """Repair Windows/OneDrive caches where snapshot files contain blob pointers."""
        repo_cache = self.segformer_cache_dir / f"models--{repo_name.replace('/', '--')}"
        snapshots_dir = repo_cache / "snapshots"
        blobs_dir = repo_cache / "blobs"
        if not snapshots_dir.exists() or not blobs_dir.exists():
            return

        for snapshot_file in snapshots_dir.rglob("*"):
            if not snapshot_file.is_file() or snapshot_file.stat().st_size > 512:
                continue
            try:
                raw = snapshot_file.read_bytes().strip()
            except OSError:
                continue
            if not raw.startswith(b"../../blobs/"):
                continue
            blob_name = raw.decode("utf-8", errors="ignore").replace("../../blobs/", "").strip()
            blob_path = blobs_dir / blob_name
            if not blob_path.exists() or not blob_path.is_file():
                continue
            logger.info(f"🔧 [ModelRegistry] Materializing cached SegFormer file: {snapshot_file.name}")
            shutil.copyfile(blob_path, snapshot_file)

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
            self._materialize_hf_snapshot_pointers(model_id)
            
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

            self._prepare_segformer_model()
            self.segformer_loaded = True
            logger.info("🎉 [ModelRegistry] SegFormer-B2 loaded successfully and warmed up!")
            return self.segformer_processor, self.segformer_model
            
        except Exception as e:
            logger.warning(
                f"⚠️ [ModelRegistry] SegFormer-B2 load failed: {e}. "
                "Hugging Face hub may be unreachable or offline. Falling through to local GrabCut..."
            )
            # Corrupt partial Hugging Face cache entries can shadow remote loads.
            # Force a clean fetch once before falling back to local-only cache.
            try:
                from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
                model_id = "mattmdjaga/segformer_b2_clothes"
                self.segformer_processor = SegformerImageProcessor.from_pretrained(
                    model_id,
                    cache_dir=str(self.segformer_cache_dir),
                    local_files_only=False,
                    force_download=True
                )
                self.segformer_model = SegformerForSemanticSegmentation.from_pretrained(
                    model_id,
                    cache_dir=str(self.segformer_cache_dir),
                    local_files_only=False,
                    force_download=True
                )
                self._prepare_segformer_model()
                self.segformer_loaded = True
                logger.info("🎉 [ModelRegistry] SegFormer-B2 loaded successfully after cache refresh!")
                return self.segformer_processor, self.segformer_model
            except Exception as refresh_e:
                logger.warning(f"⚠️ [ModelRegistry] SegFormer cache refresh failed: {refresh_e}.")

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
                self._prepare_segformer_model()
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
