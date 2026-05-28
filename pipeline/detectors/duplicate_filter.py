from typing import List, Dict, Any

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.
    Boxes are in format [x1, y1, x2, y2].
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Area of boxes
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    # Intersection coordinates
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    # Intersection area
    intersection_w = max(0.0, x2_i - x1_i)
    intersection_h = max(0.0, y2_i - y1_i)
    intersection_area = intersection_w * intersection_h

    # Union area
    union_area = area1 + area2 - intersection_area

    if union_area == 0.0:
        return 0.0

    return float(intersection_area / union_area)

def suppress_duplicates(detections: List[Dict[str, Any]], iou_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Suppresses overlapping duplicate detections based on IoU threshold.
    Detections should be a list of dicts with:
      - 'box': [x1, y1, x2, y2]
      - 'confidence': float
    Sorts detections by confidence descending, then keeps the ones that don't overlap too much.
    """
    if not detections:
        return []

    # Sort by confidence descending
    sorted_dets = sorted(detections, key=lambda x: x.get("confidence", 0.0), reverse=True)
    keep = []

    while sorted_dets:
        best = sorted_dets.pop(0)
        keep.append(best)
        
        # Filter remaining detections
        remaining = []
        for det in sorted_dets:
            iou = calculate_iou(best["box"], det["box"])
            if iou < iou_threshold:
                remaining.append(det)
        sorted_dets = remaining

    return keep
