import os
import sys
import torch
import urllib.request
import logging
from pathlib import Path
from pipeline.config import settings

logger = logging.getLogger("VestirAI.SCHPModelLoader")
logging.basicConfig(level=logging.INFO)

# --- Runtime InPlaceABN Mocking ---
# This is a critical architectural pattern that dynamically registers a PyTorch-only
# Mock class for inplace_abn, allowing the pre-trained SCHP weights (exp-schp-201908261155-lip.pth)
# to load and run on standard CPU environments without requiring any C++ or CUDA compilation!
import types
import torch.nn as nn

class MockInPlaceABN(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, activation="leaky_relu", activation_param=0.01):
        super().__init__(num_features, eps=eps, momentum=momentum, affine=affine)
        self.activation = activation
        self.activation_param = activation_param
        
        # InPlaceABN uses Leaky ReLU by default
        if activation == "leaky_relu":
            self.act = nn.LeakyReLU(negative_slope=activation_param, inplace=True)
        elif activation == "elu":
            self.act = nn.ELU(alpha=activation_param, inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(super().forward(x))

# Mock inplace_abn in sys.modules so imports of 'inplace_abn' succeed anywhere in the pipeline
if "inplace_abn" not in sys.modules:
    inplace_abn_mock = types.ModuleType("inplace_abn")
    inplace_abn_mock.InPlaceABN = MockInPlaceABN
    inplace_abn_mock.InPlaceABNSync = MockInPlaceABN
    sys.modules["inplace_abn"] = inplace_abn_mock
    logger.info("✅ Dynamically mocked 'inplace_abn' module with high-performance CPU BatchNorm2d fallback.")


class SCHPModelLoader:
    """
    Singleton class responsible for managing and loading the PyTorch SCHP model.
    Includes automated checkpoint downloading and offline fallback handling.
    """
    _instance = None
    _model = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SCHPModelLoader, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        # Establish checkpoints path under data/storage/checkpoints
        self.checkpoints_dir = settings.storage_dir / "checkpoints"
        self.model_path = self.checkpoints_dir / "exp-schp-201908261155-lip.pth"
        
        # Public download URL from levihsu/OOTDiffusion (highly reliable HuggingFace CDN)
        self.download_url = "https://huggingface.co/levihsu/OOTDiffusion/resolve/main/checkpoints/humanparsing/exp-schp-201908261155-lip.pth"

    def get_model(self, force_reload=False) -> nn.Module:
        """
        Loads the pre-trained SCHP PyTorch model.
        Returns the PyTorch model module, or raises a RuntimeError if the checkpoint
        cannot be loaded or downloaded.
        """
        if self._model is not None and not force_reload:
            return self._model

        # Ensure directory exists
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # Download checkpoint if not found
        if not self.model_path.exists():
            logger.info(f"Checkpoint not found at {self.model_path}. Attempting to download pre-trained SCHP weights...")
            try:
                self._download_weights()
            except Exception as e:
                logger.error(f"Failed to download SCHP weights: {e}. Ingestion will fall back to vertical parser.")
                raise RuntimeError(f"SCHP weights download failed: {str(e)}")

        # Initialize the network architecture
        logger.info(f"Loading SCHP weights from {self.model_path}...")
        try:
            model = self._build_network()
            
            # Load state dict safely
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            state_dict = torch.load(self.model_path, map_location=device)
            
            # Strip 'module.' prefix if present (common if trained with DataParallel)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
                
            clean_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith("module.") else k
                clean_state_dict[name] = v
                
            model.load_state_dict(clean_state_dict, strict=False)
            model.to(device)
            model.eval()
            
            self._model = model
            logger.info("🎉 SCHP Human Parser initialized successfully!")
            return self._model
        except Exception as e:
            logger.error(f"Error compiling/loading SCHP model state: {e}")
            raise RuntimeError(f"SCHP model load failure: {str(e)}")

    def _download_weights(self):
        """Downloads the weights from Hugging Face with user logging."""
        logger.info(f"Downloading pre-trained SCHP LIP model weights from {self.download_url}...")
        
        # Temp path to avoid corruption during download
        temp_path = self.model_path.with_suffix(".tmp")
        try:
            # Zero-dependency download via urllib
            urllib.request.urlretrieve(self.download_url, temp_path)
            temp_path.rename(self.model_path)
            logger.info(f"✅ Download completed. Checkpoint saved to {self.model_path}")
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise e

    def _build_network(self) -> nn.Module:
        """
        Builds the PyTorch CE2P network architecture compatible with standard ResNet101.
        We emulate the exact layer hierarchy of PeikeLi's ResNet-101 based CE2P network,
        re-mapping InPlaceABN to our CPU-efficient MockInPlaceABN.
        """
        from pipeline.parsing.schp.inference import ResNet101_CE2P
        return ResNet101_CE2P(num_classes=20)

# Global singleton loader instance
schp_model_loader = SCHPModelLoader()
