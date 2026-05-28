import uuid
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from pipeline.detectors.yolo_detector import yolo_detector
from pipeline.detectors.face_blur import face_blurrer
from pipeline.detectors.schp_parser import human_clothing_parser
from pipeline.detectors.sam2_segmenter import sam2_segmenter
from pipeline.analysis.siglip_tagger import siglip_tagger
from pipeline.analysis.gemini_client import gemini_client
from pipeline.analysis.oklch_scorer import extract_dominant_colors
from pipeline.services.storage_service import storage_service
from pipeline.services.vector_store import vector_store
from pipeline.database.storage import insert_wardrobe_item

def process_single_worn_outfit(img_path: Path, scene_type: str = "single_person") -> List[Dict[str, Any]]:
    """
    S3 Ingestion Pipeline:
    1. Load image and apply face blurring for privacy
    2. Detect the person using YOLO11-seg
    3. Generate the person binary mask
    4. Run HumanClothingParser to segment garments semantically (upper body, lower body, shoes)
    5. Run SAM2 GrabCut boundary refinement for each garment layer
    6. Extract transparent crop PNGs
    7. Generate color, tagging, and Gemini fashion analytics for each item
    8. Index in vector index and save to SQLite
    """
    # 1. Load image and apply face blur
    img = storage_service.load_image(img_path)
    blurred_img, _ = face_blurrer.blur_faces(img)
    
    # 2. Detect the person
    people = yolo_detector.detect_people(img)
    if not people:
        raise ValueError("No person detected in the worn outfit image.")
        
    # Take the largest person box
    person = max(people, key=lambda x: (x["box"][2] - x["box"][0]) * (x["box"][3] - x["box"][1]))
    box = person["box"]
    polygon = person["polygon"]
    
    # 3. Re-create binary mask for the person
    h, w = img.shape[:2]
    person_mask = np.zeros((h, w), dtype=np.uint8)
    poly_pts = np.array(polygon, dtype=np.int32)
    if len(poly_pts) > 0:
        cv2.fillPoly(person_mask, [poly_pts], 255)
        
    # 4. Partition human mask semantically into clothing layers
    layers = human_clothing_parser.parse_clothing_layers(blurred_img, person_mask, box)
    
    ingested_items = []
    
    # 5. Process each layer
    for layer in layers:
        layer_type = layer["layer_type"]
        layer_box = layer["box"]
        layer_poly = layer["polygon"]
        
        # We don't ingest shoes if they are too tiny to avoid noisy crop boxes
        if layer_type == "shoes":
            lw = layer_box[2] - layer_box[0]
            lh = layer_box[3] - layer_box[1]
            if lw < 25 or lh < 25:
                continue
                
        # 6. Run boundary GrabCut refinement specifically for the layer
        mask_refined = sam2_segmenter.refine_mask_from_polygon(blurred_img, layer_poly)
        
        # Extract transparent crop
        cropped_rgba = sam2_segmenter.extract_transparent_crop(blurred_img, mask_refined, layer_box)
        
        # Unique item ID
        item_id = f"item_{uuid.uuid4().hex[:8]}"
        
        # Save crop
        crop_path = storage_service.save_crop_rgba(cropped_rgba, item_id)
        
        # Extract dominant colors
        dominant_colors = extract_dominant_colors(cropped_rgba, num_colors=3)
        
        # Extract visual tags & embedding vector
        tags, embedding = siglip_tagger.extract_visual_features(cropped_rgba)
        
        # Adjust tags based on layer type
        if layer_type == "upper":
            tags.extend(["layer-top", "upper-body"])
        elif layer_type == "lower":
            tags.extend(["layer-bottom", "lower-body"])
            
        crop_h, crop_w = cropped_rgba.shape[:2]
        aspect_ratio = crop_h / crop_w if crop_w > 0 else 1.0
        
        # Extract semantic parameters (Gemini / local offline rule engine)
        fashion_meta = gemini_client.analyze_garment(tags, dominant_colors, aspect_ratio, layer_hint=layer_type)
        
        # Make sure garment type matches layer
        if layer_type == "upper" and fashion_meta["garment_type"] not in ["top", "outerwear", "dress"]:
            fashion_meta["garment_type"] = "top"
        elif layer_type == "lower" and fashion_meta["garment_type"] != "bottom":
            fashion_meta["garment_type"] = "bottom"
        elif layer_type == "shoes" and fashion_meta["garment_type"] != "shoes":
            fashion_meta["garment_type"] = "shoes"
            
        item_data = {
            "id": item_id,
            "colors": dominant_colors,
            "tags": list(set(tags + [fashion_meta.get("subtype", "garment"), fashion_meta.get("material", "cotton")])),
            "image_path": storage_service.get_relative_path(crop_path),
            "scene_type": scene_type,
            **fashion_meta
        }
        
        # Index in FAISS/NumPy & Save in SQL
        vector_store.add_item(item_id, embedding)
        insert_wardrobe_item(item_id, item_data)
        
        ingested_items.append(item_data)
        
    return ingested_items
