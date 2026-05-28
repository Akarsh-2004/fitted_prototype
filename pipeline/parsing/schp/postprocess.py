import cv2
import numpy as np
import torch
from typing import Dict, Tuple
from pipeline.parsing.schp.labels import LIP_CLASSES

def postprocess_logits(
    logits: torch.Tensor, 
    original_size: Tuple[int, int], 
    confidence_threshold: float = 0.55
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Postprocesses the raw logits from the SCHP network.
    - Applies softmax to get probability maps
    - Computes argmax for class segmentation map
    - Computes confidence map (max probability of the chosen class)
    - Maps output back to original size (width, height)
    - Separates individual class masks
    
    Returns:
      - segmentation_map: np.ndarray of shape (H, W) [integer label indices]
      - confidence_map: np.ndarray of shape (H, W) [probabilities 0.0 - 1.0]
      - label_masks: Dict[str, np.ndarray] mapping class names to binary masks
    """
    orig_w, orig_h = original_size
    
    # 1. Softmax to obtain probabilities
    probs = torch.softmax(logits, dim=1).squeeze(0)  # Shape: [20, 473, 473]
    probs_np = probs.cpu().numpy()  # Convert to NumPy for spatial filtering
    
    # Apply Gaussian blur per class channel to suppress blocky high-frequency noise
    smoothed_probs = np.zeros_like(probs_np)
    for c in range(probs_np.shape[0]):
        smoothed_probs[c] = cv2.GaussianBlur(
            probs_np[c],
            ksize=(5, 5),
            sigmaX=1.2
        )
    
    # 2. Get class indices and confidence maps from smoothed probabilities
    seg_np = np.argmax(smoothed_probs, axis=0).astype(np.uint8)
    conf_np = np.max(smoothed_probs, axis=0).astype(np.float32)
    
    # 3. Resize back to original crop resolution
    # Use INTER_NEAREST for segmentation label map to avoid interpolation anomalies
    segmentation_map = cv2.resize(seg_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    confidence_map = cv2.resize(conf_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    
    # 4. Extract separate binary masks for each LIP class
    label_masks = {}
    for i, name in enumerate(LIP_CLASSES):
        # Create a clean binary mask for the class
        class_mask = (segmentation_map == i).astype(np.uint8) * 255
        
        # Apply confidence mask thresholding for safety (except background)
        if i > 0:
            confidence_mask = (confidence_map >= confidence_threshold).astype(np.uint8) * 255
            class_mask = cv2.bitwise_and(class_mask, confidence_mask)
            
        label_masks[name] = class_mask
        
    return segmentation_map, confidence_map, label_masks
