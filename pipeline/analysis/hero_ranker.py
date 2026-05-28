import numpy as np
from typing import List, Dict, Any, Tuple

def rank_items_by_salience(detections: List[Dict[str, Any]], img_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
    """
    Ranks detected flat lay garments by visual salience (size + centrality + confidence).
    Appends a 'salience_score' to each detection and returns them sorted descending.
    """
    if not detections:
        return []
        
    img_h, img_w = img_shape[:2]
    img_center_x = img_w / 2.0
    img_center_y = img_h / 2.0
    img_area = img_h * img_w
    
    ranked_dets = []
    
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        w = x2 - x1
        h = y2 - y1
        det_area = w * h
        
        # Center of detection
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        
        # 1. Size score (normalized area, capped at 0.5 for realistic scaling)
        size_score = min(1.0, det_area / (img_area * 0.4))
        
        # 2. Centrality score (distance from image center, normalized)
        max_dist = np.sqrt(img_center_x**2 + img_center_y**2)
        dist = np.sqrt((cx - img_center_x)**2 + (cy - img_center_y)**2)
        centrality_score = 1.0 - (dist / max_dist) if max_dist > 0 else 1.0
        
        # 3. Confidence score
        conf_score = det.get("confidence", 0.5)
        
        # Combine parameters (weights: 40% size, 40% centrality, 20% confidence)
        salience = (0.4 * size_score) + (0.4 * centrality_score) + (0.2 * conf_score)
        
        # Create a new copy of detection to append score
        new_det = det.copy()
        new_det["salience_score"] = float(round(salience, 3))
        ranked_dets.append(new_det)
        
    # Sort descending by salience score
    ranked_dets.sort(key=lambda x: x["salience_score"], reverse=True)
    
    return ranked_dets
