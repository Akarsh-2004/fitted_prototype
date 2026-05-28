import os
from pathlib import Path


def _load_dotenv(env_path: Path) -> None:
    """Minimal .env loader used to keep configuration in sync between the
    direct-Python scripts and the uvicorn-launched backend.

    We deliberately avoid the `python-dotenv` dependency since the .env file
    here is tiny and we already have a one-line parser elsewhere in the repo.
    """
    if not env_path.is_file():
        return
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception as exc:  # noqa: BLE001 - best-effort, never block startup
        print(f"[config] failed to parse {env_path}: {exc}")


# Load .env before any settings are read so Settings sees the variables.
_load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class Settings:
    def __init__(self):
        # Base Paths
        self.base_dir: Path = Path(__file__).resolve().parent.parent
        self.storage_dir: Path = self.base_dir / "data" / "storage"
        self.vector_index_dir: Path = self.base_dir / "data" / "vector"
        
        # Ensure directories exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.vector_index_dir.mkdir(parents=True, exist_ok=True)
        
        # Config & DB Settings
        self.debug: bool = os.getenv("DEBUG", "True").lower() == "true"
        self.database_url: str = os.getenv("DATABASE_URL", f"sqlite:///{self.base_dir}/wardrobe.db")
        self.sqlite_db_path: Path = self.base_dir / "wardrobe.db"
        
        # Model Identifiers
        self.yolo_model_path: str = os.getenv("YOLO_MODEL_PATH", str(self.base_dir / "yolo11n-seg.pt"))
        if not Path(self.yolo_model_path).exists():
            # Fall back to automatic download
            self.yolo_model_path = "yolo11s-seg.pt"
            
        self.gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
        
        # SCHP & SAM2 Semantic Refinement settings
        self.schp_confidence_threshold: float = float(os.getenv("SCHP_CONFIDENCE_THRESHOLD", "0.25"))
        self.min_component_area: int = int(os.getenv("MIN_COMPONENT_AREA", "250"))
        self.mask_smoothing_kernel: int = int(os.getenv("MASK_SMOOTHING_KERNEL", "5"))
        self.sam_refinement_padding: int = int(os.getenv("SAM_REFINEMENT_PADDING", "8"))
        self.max_garment_overlap: float = float(os.getenv("MAX_GARMENT_OVERLAP", "0.15"))
        self.debug_exports_enabled: bool = os.getenv("DEBUG_EXPORTS_ENABLED", "True").lower() == "true"
        
        # High-Precision Waist Splits and Foot Plausibility settings
        self.waist_split_iou_trigger: float = 0.30
        self.same_color_delta_e_threshold: float = 15.0
        self.foot_plausibility_min_centroid_y: float = 0.88
        self.foot_plausibility_min_horizontal_spread: float = 0.10

settings = Settings()
