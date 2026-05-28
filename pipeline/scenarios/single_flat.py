import uuid
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from pipeline.detectors.yolo_detector import yolo_detector
from pipeline.detectors.sam2_segmenter import sam2_segmenter
from pipeline.analysis.siglip_tagger import siglip_tagger
from pipeline.analysis.gemini_client import gemini_client
from pipeline.analysis.oklch_scorer import extract_dominant_colors
from pipeline.services.storage_service import storage_service
from pipeline.services.vector_store import vector_store
from pipeline.database.storage import insert_wardrobe_item

def process_single_flat_lay(img_path: Path, scene_type: str = "flat_single") -> Dict[str, Any]:
    """
    S1 Ingestion Pipeline:
    1. Load image
    2. Detect garment bounds (YOLO or Otsu threshold)
    3. Run SAM2 (GrabCut) boundary refinement
    4. Save transparent crop
    5. Analyze dominant colors, tags, and embeddings
    6. Request semantic attributes (Gemini / Offline)
    7. Save to FAISS vector index & SQL database
    """
    # 1. Load image
    img = storage_service.load_image(img_path)
    h, w = img.shape[:2]
    
    # 2. Detect garments
    garments = yolo_detector.detect_flat_lay_garments(img)
    
    if not garments:
        # Fallback: Assume whole image (excluding outer 2% margin) is the garment
        margin_x = int(w * 0.02)
        margin_y = int(h * 0.02)
        box = [float(margin_x), float(margin_y), float(w - margin_x), float(h - margin_y)]
        polygon = [
            [box[0], box[1]],
            [box[2], box[1]],
            [box[2], box[3]],
            [box[0], box[3]]
        ]
        garment = {"box": box, "polygon": polygon, "confidence": 1.0}
    else:
        # Take the most salient garment (largest bounding box area)
        garment = max(garments, key=lambda x: (x["box"][2] - x["box"][0]) * (x["box"][3] - x["box"][1]))
        
    box = garment["box"]
    polygon = garment["polygon"]
    
    # 3. Refine boundaries using SAM2 GrabCut algorithm
    mask = sam2_segmenter.refine_mask_from_polygon(img, polygon)
    
    # 4. Extract 4-channel transparent crop
    cropped_rgba = sam2_segmenter.extract_transparent_crop(img, mask, box)
    
    # Generate unique item ID
    item_id = f"item_{uuid.uuid4().hex[:8]}"
    
    # Save the crop
    crop_path = storage_service.save_crop_rgba(cropped_rgba, item_id)
    
    # 5. Extract colors
    dominant_colors = extract_dominant_colors(cropped_rgba, num_colors=3)
    
    # 6. Extract visual tags & embedding vector representation
    tags, embedding = siglip_tagger.extract_visual_features(cropped_rgba)
    
    # Calculate aspect ratio of cropped garment
    crop_h, crop_w = cropped_rgba.shape[:2]
    aspect_ratio = crop_h / crop_w if crop_w > 0 else 1.0
    
    # 7. Extract semantic attributes
    fashion_meta = gemini_client.analyze_garment(tags, dominant_colors, aspect_ratio)
    
    # 8. Add dominant colors, tags, crop path, and scene type to item data
    item_data = {
        "id": item_id,
        "colors": dominant_colors,
        "tags": list(set(tags + [fashion_meta.get("subtype", "garment"), fashion_meta.get("material", "cotton")])),
        "image_path": storage_service.get_relative_path(crop_path),
        "scene_type": scene_type,
        **fashion_meta
    }
    
    # 9. Write to Local Vector Database (NumPy index)
    vector_store.add_item(item_id, embedding)
    
    # 10. Save to SQLite database
    insert_wardrobe_item(item_id, item_data)
    
    return item_data
