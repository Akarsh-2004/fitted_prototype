import cv2
import numpy as np
import torch

def preprocess_image(img: np.ndarray, target_size: int = 473) -> torch.Tensor:
    """
    Preprocesses an RGB image for SCHP model inference.
    - Resizes to target_size x target_size
    - Normalizes with ImageNet mean & standard deviation
    - Transposes to channel-first (C, H, W)
    - Returns a batch tensor [1, C, H, W]
    """
    # 1. Resize
    resized = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    # 2. Convert to float32 and scale to [0, 1]
    resized = resized.astype(np.float32) / 255.0
    
    # 3. Apply ImageNet normalization (standard for ResNet segmenters)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    normalized = (resized - mean) / std
    
    # 4. Transpose from HWC to CHW
    chw = normalized.transpose((2, 0, 1))
    
    # 5. Convert to tensor and add batch dimension
    tensor = torch.from_numpy(chw).unsqueeze(0)
    return tensor
