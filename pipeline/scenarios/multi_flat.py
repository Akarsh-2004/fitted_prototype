import uuid
from pathlib import Path
from typing import List, Dict, Any
from pipeline.config import settings
from pipeline.detectors.yolo_detector import yolo_detector
from pipeline.detectors.duplicate_filter import suppress_duplicates
from pipeline.analysis.hero_ranker import rank_items_by_salience
from pipeline.detectors.sam2_segmenter import sam2_segmenter
from pipeline.analysis.siglip_tagger import siglip_tagger
from pipeline.analysis.gemini_client import gemini_client
from pipeline.analysis.oklch_scorer import extract_dominant_colors
from pipeline.services.storage_service import storage_service
from pipeline.services.cache import upload_cache
from pipeline.services.vector_store import vector_store
from pipeline.database.storage import update_job, insert_wardrobe_item

def detect_multi_flat_lay(img_path: Path, job_id: str) -> List[Dict[str, Any]]:
    """
    S2 Stage 1: Detects, suppresses duplicates, ranks garments, and generates transparent previews.
    Caches intermediate data for subsequent confirmation.
    """
    img = storage_service.load_image(img_path)
    
    # 1. Detect flat-lay garments
    garments = yolo_detector.detect_flat_lay_garments(img)
    
    # 2. Suppress duplicate bounding boxes (IoU threshold 0.4)
    filtered_garments = suppress_duplicates(garments, iou_threshold=0.4)
    
    # 3. Rank items by salience (size + centrality)
    ranked_garments = rank_items_by_salience(filtered_garments, img.shape)
    
    # 4. Generate transparent preview crops for confirmation
    detected_items = []
    cached_data = []
    
    for i, garment in enumerate(ranked_garments):
        box = garment["box"]
        polygon = garment["polygon"]
        
        # Extract beautiful transparent preview crop
        mask = sam2_segmenter.refine_mask_from_polygon(img, polygon)
        cropped_rgba = sam2_segmenter.extract_transparent_crop(img, mask, box)
        
        # Save temporary preview crop
        preview_id = f"preview_{job_id}_{i}"
        preview_path = storage_service.save_crop_rgba(cropped_rgba, preview_id)
        
        item_preview = {
            "index": i,
            "preview_id": preview_id,
            "confidence": garment["confidence"],
            "box": box,
            "image_url": storage_service.get_relative_path(preview_path),
            "salience_score": garment.get("salience_score", 0.5)
        }
        
        detected_items.append(item_preview)
        
        # Cache detailed data for stage 2
        cached_data.append({
            "box": box,
            "polygon": polygon,
            "mask": mask.tolist(), # serialize to list for safety or cache directly
            "preview_path": preview_path
        })
        
    # Store in memory cache
    upload_cache.set(f"job_data_{job_id}", {
        "img_path": img_path,
        "cached_items": cached_data
    })
    
    # Update Job in database
    update_job(
        job_id=job_id,
        status="requires_confirmation",
        scene_type="flat_multi",
        detected_items=detected_items
    )
    
    return detected_items

def ingest_confirmed_multi_items(job_id: str, confirmed_indices: List[int]) -> List[Dict[str, Any]]:
    """
    S2 Stage 2: Ingests only the items confirmed by the user.
    Runs asynchronously, extracting full fashion intelligence, indexing, and storing.
    """
    job_cache = upload_cache.get(f"job_data_{job_id}")
    if not job_cache:
        raise ValueError(f"Cache expired or not found for job: {job_id}")
        
    img_path = job_cache["img_path"]
    cached_items = job_cache["cached_items"]
    
    img = storage_service.load_image(img_path)
    
    ingested_items = []
    
    for idx in confirmed_indices:
        if idx >= len(cached_items):
            continue
            
        cached_item = cached_items[idx]
        box = cached_item["box"]
        polygon = cached_item["polygon"]
        
        # Run boundary refinement and crop
        mask = sam2_segmenter.refine_mask_from_polygon(img, polygon)
        cropped_rgba = sam2_segmenter.extract_transparent_crop(img, mask, box)
        
        # Generate permanent unique item ID
        item_id = f"item_{uuid.uuid4().hex[:8]}"
        
        # Save permanent crop
        crop_path = storage_service.save_crop_rgba(cropped_rgba, item_id)
        
        # Extract colors and tags
        dominant_colors = extract_dominant_colors(cropped_rgba, num_colors=3)
        tags, embedding = siglip_tagger.extract_visual_features(cropped_rgba)
        
        crop_h, crop_w = cropped_rgba.shape[:2]
        aspect_ratio = crop_h / crop_w if crop_w > 0 else 1.0
        
        # Extract semantic attributes
        fashion_meta = gemini_client.analyze_garment(tags, dominant_colors, aspect_ratio)
        
        item_data = {
            "id": item_id,
            "colors": dominant_colors,
            "tags": list(set(tags + [fashion_meta.get("subtype", "garment"), fashion_meta.get("material", "cotton")])),
            "image_path": storage_service.get_relative_path(crop_path),
            "scene_type": "flat_multi",
            **fashion_meta
        }
        
        # Save to vector index & DB
        vector_store.add_item(item_id, embedding)
        insert_wardrobe_item(item_id, item_data)
        
        ingested_items.append(item_data)
        
        # Clean up temporary preview file
        try:
            temp_path = cached_item["preview_path"]
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass
            
    # Update job state to completed
    update_job(
        job_id=job_id,
        status="completed",
        result=ingested_items
    )
    
    # Clear cache
    upload_cache.delete(f"job_data_{job_id}")
    
    return ingested_items
