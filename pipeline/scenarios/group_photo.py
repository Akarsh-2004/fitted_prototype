import uuid
import cv2
import io
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
from pipeline.config import settings
from pipeline.detectors.yolo_detector import yolo_detector
from pipeline.detectors.face_blur import face_blurrer
from pipeline.detectors.schp_parser import human_clothing_parser
from pipeline.detectors.sam2_segmenter import sam2_segmenter
from pipeline.analysis.siglip_tagger import siglip_tagger
from pipeline.analysis.gemini_client import gemini_client
from pipeline.analysis.oklch_scorer import extract_dominant_colors
from pipeline.services.storage_service import storage_service
from pipeline.services.cache import upload_cache
from pipeline.services.vector_store import vector_store
from pipeline.database.storage import update_job, insert_wardrobe_item
from pipeline.composer.gemini_image import (
    GeminiImageError,
    cutout_area_ratio,
    extract_cutout_rgba,
)

logger = logging.getLogger("VestirAI.GroupPhoto")

def detect_group_photo(img_path: Path, job_id: str) -> Dict[str, Any]:
    """
    S4 Stage 1: Detects all people, generates contour coordinates, and caches.
    Marks job as 'requires_selection'.
    """
    img = storage_service.load_image(img_path)
    h, w = img.shape[:2]
    
    # Run face blur for display safety
    blurred_img, face_count = face_blurrer.blur_faces(img)
    
    # Save blurred version for display on canvas
    display_id = f"display_{job_id}"
    display_path = storage_service.crops_dir / f"{display_id}.jpg"
    cv2.imwrite(str(display_path), blurred_img)
    
    # Detect people using YOLO
    people = yolo_detector.detect_people(img)
    
    # Construct persons list for UI representation
    ui_people = []
    cached_people = []
    
    for i, person in enumerate(people):
        ui_people.append({
            "id": i,
            "confidence": person["confidence"],
            "box": person["box"],
            "polygon": person["polygon"]
        })
        cached_people.append({
            "box": person["box"],
            "polygon": person["polygon"]
        })
        
    # Cache original image and people details
    upload_cache.set(f"group_job_{job_id}", {
        "img_path": img_path,
        "cached_people": cached_people
    })
    
    result_data = {
        "width": w,
        "height": h,
        "image_url": storage_service.get_relative_path(display_path),
        "people": ui_people
    }
    
    update_job(
        job_id=job_id,
        status="requires_selection",
        scene_type="group_photo",
        detected_items=ui_people
    )
    
    return result_data

def find_clicked_person(job_id: str, click_x: float, click_y: float) -> Optional[int]:
    """
    Finds which cached person's polygon contains the clicked coordinate (x, y).
    Uses standard OpenCV pointPolygonTest (Point-in-Polygon).
    """
    job_cache = upload_cache.get(f"group_job_{job_id}")
    if not job_cache:
        return None
        
    cached_people = job_cache["cached_people"]
    
    for idx, person in enumerate(cached_people):
        poly = np.array(person["polygon"], dtype=np.float32)
        if len(poly) < 3:
            continue
            
        # Check if point is inside or on the contour boundary
        dist = cv2.pointPolygonTest(poly, (click_x, click_y), False)
        if dist >= 0:
            return idx
            
    return None

def ingest_selected_group_person(job_id: str, person_idx: int) -> Dict[str, Any]:
    """
    S4 Stage 2: Ingests the clothing layers for the selected person.
    Integrates cutout extraction and fine-grained human parsing.
    """
    from pipeline.parsing.cutout_extractor import cutout_extractor
    from pipeline.parsing.fine_parser import fine_parser
    
    job_cache = upload_cache.get(f"group_job_{job_id}")
    if not job_cache:
        raise ValueError(f"Cache expired or not found for group job: {job_id}")
        
    img_path = job_cache["img_path"]
    cached_people = job_cache["cached_people"]
    
    if person_idx >= len(cached_people):
        raise ValueError(f"Invalid person index: {person_idx}")
        
    person = cached_people[person_idx]
    box = person["box"]
    polygon = person["polygon"]
    
    # Load original image and apply face blurring for privacy
    img = storage_service.load_image(img_path)
    blurred_img, _ = face_blurrer.blur_faces(img)
    h, w = img.shape[:2]
    
    # Recreate the person's mask
    person_mask = np.zeros((h, w), dtype=np.uint8)
    poly_pts = np.array(polygon, dtype=np.int32)
    if len(poly_pts) > 0:
        cv2.fillPoly(person_mask, [poly_pts], 255)
        
    # 1. Run cutout_extractor.py -> get RGBA person PNG
    logger.info(f"🔮 [S4 Stage 2] Extracting person cutout for person {person_idx}...")
    cutout_result = cutout_extractor.extract_person_cutout(
        blurred_img,
        box,
        polygon,
        job_id,
        person_idx
    )

    # 2. Run fine_parser.py -> get parts list
    logger.info(f"🔮 [S4 Stage 2] Parsing fine-grained human parts for person {person_idx}...")
    fine_result = fine_parser.parse_granular_clothing_layers(
        blurred_img,
        person_mask,
        box,
        job_id,
        person_idx
    )

    ingested_items = []
    
    # 3. For each part where ingest = True:
    for part in fine_result["parts"]:
        if not part["ingest"]:
            continue

        label_type = part["label"]
        layer_box = part["bbox"] # [x, y, w, h] format
        
        # Load the transparent crop saved by the fine parser
        rgba_abs = settings.base_dir / part["rgba_crop_path"]
        cropped_bgra = cv2.imread(str(rgba_abs), cv2.IMREAD_UNCHANGED)
        if cropped_bgra is None:
            continue
        cropped_rgba = cv2.cvtColor(cropped_bgra, cv2.COLOR_BGRA2RGBA)
        
        # Unique item ID
        item_id = f"item_{uuid.uuid4().hex[:8]}"
        
        # Save permanent crop crop to "/data/storage/crops/"
        perm_crop_path = storage_service.save_crop_rgba(cropped_rgba, item_id)
        
        # Extract dominant colors
        dominant_colors = extract_dominant_colors(cropped_rgba, num_colors=3)
        
        # Extract visual tags & embedding vector
        tags, embedding = siglip_tagger.extract_visual_features(cropped_rgba)
        
        # Append layer information for tagging
        tags.extend([f"layer-{label_type}", "fine-grained-part"])
            
        crop_h, crop_w = cropped_rgba.shape[:2]
        aspect_ratio = crop_h / crop_w if crop_w > 0 else 1.0
        
        # Gemini Fashion Intelligence analysis
        fashion_meta = gemini_client.analyze_garment(tags, dominant_colors, aspect_ratio, layer_hint=label_type)
        
        # Clean type classifications
        if label_type in ["top_garment", "outerwear"] and fashion_meta["garment_type"] not in ["top", "outerwear", "dress"]:
            fashion_meta["garment_type"] = "top"
        elif label_type == "bottom_garment" and fashion_meta["garment_type"] != "bottom":
            fashion_meta["garment_type"] = "bottom"
        elif label_type in ["left_shoe", "right_shoe", "footwear"] and fashion_meta["garment_type"] != "shoes":
            fashion_meta["garment_type"] = "shoes"
        elif label_type in ["accessory", "hat", "bag", "wrist_accessory"] and fashion_meta["garment_type"] != "accessory":
            fashion_meta["garment_type"] = "accessory"
            
        item_data = {
            "id": item_id,
            "colors": dominant_colors,
            "tags": list(set(tags + [fashion_meta.get("subtype", "garment"), fashion_meta.get("material", "cotton")])),
            "image_path": storage_service.get_relative_path(perm_crop_path),
            "scene_type": "group_photo",
            **fashion_meta
        }
        
        # Vector and SQL Storage
        vector_store.add_item(item_id, embedding)
        insert_wardrobe_item(item_id, item_data)
        
        ingested_items.append(item_id)
        
    result_data = {
        "cutout": cutout_result,
        "parsed_parts": fine_result["parts"],
        "ingested_items": ingested_items
    }

    # 4. Update Job result in Database
    update_job(
        job_id=job_id,
        status="completed",
        result=[result_data]  # We wrap in a list so it remains queryable as a list of dicts for safety
    )
    
    # 5. Clean up cached data
    upload_cache.delete(f"group_job_{job_id}")
    
    return result_data


_GEMINI_PERSON_CATEGORIES = {
    "top": "upper-body garment worn by the selected person (shirt, t-shirt, hoodie, sweater, jacket)",
    "bottom": "lower-body garment worn by the selected person (pants, jeans, trousers, shorts, skirt)",
    "shoes": "pair of shoes worn by the selected person",
}


def _save_rgba_bytes_as_crop(rgba_bytes: bytes, item_id: str) -> Path:
    pil = Image.open(io.BytesIO(rgba_bytes)).convert("RGBA")
    rgba_np = np.array(pil)
    return storage_service.save_crop_rgba(rgba_np, item_id)


def _force_meta_category(fashion_meta: Dict[str, Any], category: str) -> None:
    if category == "top":
        fashion_meta["garment_type"] = "top"
    elif category == "bottom":
        fashion_meta["garment_type"] = "bottom"
    elif category == "shoes":
        fashion_meta["garment_type"] = "shoes"


def _force_meta_for_segformer_label(fashion_meta: Dict[str, Any], label_type: str) -> None:
    if label_type == "top_garment":
        fashion_meta["garment_type"] = "top"
    elif label_type == "bottom_garment":
        fashion_meta["garment_type"] = "bottom"
    elif label_type == "footwear":
        fashion_meta["garment_type"] = "shoes"
    elif label_type in {"hat", "bag", "accessory"}:
        fashion_meta["garment_type"] = "accessory"


def ingest_selected_group_person_segformer(job_id: str, person_idx: int) -> Dict[str, Any]:
    """SegFormer Stage 2 for a clicked person.

    Reuses S4 Stage 1 person detection/click selection, then parses the selected
    crop with SegFormer clothes labels to create deterministic garment layers.
    """
    from pipeline.parsing.occlusion_repair import build_blocked_mask
    from pipeline.parsing.segformer_parser import segformer_parser

    job_cache = upload_cache.get(f"group_job_{job_id}")
    if not job_cache:
        raise ValueError(f"Cache expired or not found for group job: {job_id}")

    img_path = job_cache["img_path"]
    cached_people = job_cache["cached_people"]

    if person_idx >= len(cached_people):
        raise ValueError(f"Invalid person index: {person_idx}")

    person = cached_people[person_idx]
    box = person["box"]
    polygon = person["polygon"]

    img = storage_service.load_image(img_path)
    blurred_img, _ = face_blurrer.blur_faces(img)
    h, w = blurred_img.shape[:2]

    person_mask = np.zeros((h, w), dtype=np.uint8)
    poly_pts = np.array(polygon, dtype=np.int32)
    if len(poly_pts) > 0:
        cv2.fillPoly(person_mask, [poly_pts], 255)
    blocked_mask = build_blocked_mask((h, w), cached_people, person_idx)

    logger.info(f"🔮 [S4 Stage 2] Parsing selected person {person_idx} with SegFormer...")
    segformer_result = segformer_parser.parse_clothing_layers(
        blurred_img,
        person_mask,
        box,
        job_id,
        person_idx,
        blocked_mask=blocked_mask,
    )

    ingested_items = []
    segformer_layers = []

    for part in segformer_result["parts"]:
        if not part["ingest"]:
            continue

        label_type = part["label"]
        if label_type in {"bag", "accessory"}:
            continue
        rgba_abs = settings.base_dir / part["rgba_crop_path"]
        cropped_bgra = cv2.imread(str(rgba_abs), cv2.IMREAD_UNCHANGED)
        if cropped_bgra is None:
            continue

        cropped_rgba = cv2.cvtColor(cropped_bgra, cv2.COLOR_BGRA2RGBA)
        item_id = f"seg_{uuid.uuid4().hex[:8]}"
        perm_crop_path = storage_service.save_crop_rgba(cropped_rgba, item_id)

        dominant_colors = extract_dominant_colors(cropped_rgba, num_colors=3)
        tags, embedding = siglip_tagger.extract_visual_features(cropped_rgba)
        tags.extend([f"layer-{label_type}", "segformer-selected-person"])

        crop_h, crop_w = cropped_rgba.shape[:2]
        aspect_ratio = crop_h / crop_w if crop_w > 0 else 1.0
        layer_hint = (
            "upper" if label_type == "top_garment"
            else "lower" if label_type == "bottom_garment"
            else "shoes" if label_type == "footwear"
            else "hat" if label_type == "hat"
            else None
        )

        fashion_meta = gemini_client.analyze_garment(
            tags,
            dominant_colors,
            aspect_ratio,
            layer_hint=layer_hint,
        )
        _force_meta_for_segformer_label(fashion_meta, label_type)

        item_data = {
            "id": item_id,
            "colors": dominant_colors,
            "tags": list(set(tags + [fashion_meta.get("subtype", "garment"), fashion_meta.get("material", "cotton")])),
            "image_path": storage_service.get_relative_path(perm_crop_path),
            "scene_type": "group_photo",
            **fashion_meta,
        }

        vector_store.add_item(item_id, embedding)
        insert_wardrobe_item(item_id, item_data)
        ingested_items.append(item_id)
        segformer_layers.append({
            "id": item_id,
            "label": label_type,
            "confidence": part.get("confidence"),
            "pixel_area": part.get("pixel_area"),
        })

    if not ingested_items:
        raise ValueError("SegFormer did not produce any ingestible garment layers for the selected person.")

    result_data = {
        "cutout": segformer_result["cutout"],
        "parsed_parts": segformer_result["parts"],
        "ingested_items": ingested_items,
        "segformer_layers": segformer_layers,
        "segformer_selected_person": True,
    }

    update_job(job_id=job_id, status="completed", result=[result_data])
    upload_cache.delete(f"group_job_{job_id}")
    return result_data


def ingest_selected_group_person_gemini(job_id: str, person_idx: int) -> Dict[str, Any]:
    """Gemini Stage 2 for a clicked person.

    Reuses S4 Stage 1 person detection/click selection, then asks Gemini to
    isolate the selected person's top, bottom and shoes as independent product
    cutouts. This avoids the SCHP parser while preserving the original
    uploaded-image hover UX.
    """
    job_cache = upload_cache.get(f"group_job_{job_id}")
    if not job_cache:
        raise ValueError(f"Cache expired or not found for group job: {job_id}")

    img_path = job_cache["img_path"]
    cached_people = job_cache["cached_people"]

    if person_idx >= len(cached_people):
        raise ValueError(f"Invalid person index: {person_idx}")

    person = cached_people[person_idx]
    box = [int(v) for v in person["box"]]

    img = storage_service.load_image(img_path)
    blurred_img, _ = face_blurrer.blur_faces(img)
    h, w = blurred_img.shape[:2]

    x1, y1, x2, y2 = box
    pad_x = int(max(12, (x2 - x1) * 0.12))
    pad_y = int(max(12, (y2 - y1) * 0.08))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    selected_crop = blurred_img[y1:y2, x1:x2]

    ok, encoded = cv2.imencode(".jpg", selected_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    if not ok:
        raise ValueError("Failed to encode selected person crop for Gemini")
    selected_bytes = encoded.tobytes()

    # Save an opaque selected-person preview so the completed S4 card can still render.
    preview_id = f"person_{job_id}_{person_idx}_gemini"
    preview_rgba = cv2.cvtColor(selected_crop, cv2.COLOR_BGR2RGBA)
    preview_rgba[:, :, 3] = 255
    preview_path = storage_service.save_crop_rgba(preview_rgba, preview_id)

    ingested_ids: List[str] = []
    gemini_layers: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    for category, hint in _GEMINI_PERSON_CATEGORIES.items():
        try:
            rgba_bytes, gemini_meta = extract_cutout_rgba(selected_bytes, category_hint=hint)
            item_id = f"gem_{uuid.uuid4().hex[:8]}"
            crop_path = _save_rgba_bytes_as_crop(rgba_bytes, item_id)
            rgba_np = np.array(Image.open(io.BytesIO(rgba_bytes)).convert("RGBA"))
            crop_h, crop_w = rgba_np.shape[:2]
            aspect_ratio = float(crop_h) / float(crop_w) if crop_w else 1.0
            dominant_colors = extract_dominant_colors(rgba_np, num_colors=3)

            rough_tags = [category, "gemini-selected-person"]
            layer_hint = "upper" if category == "top" else "lower" if category == "bottom" else "shoes"
            fashion_meta = gemini_client.analyze_garment(
                rough_tags,
                dominant_colors,
                aspect_ratio,
                layer_hint=layer_hint,
            )
            _force_meta_category(fashion_meta, category)

            item_data = {
                "id": item_id,
                "colors": dominant_colors,
                "tags": list(set(rough_tags + [fashion_meta.get("subtype", category), fashion_meta.get("material", "cotton")])),
                "image_path": storage_service.get_relative_path(crop_path),
                "scene_type": "group_photo",
                **fashion_meta,
            }
            insert_wardrobe_item(item_id, item_data)
            ingested_ids.append(item_id)
            gemini_layers.append({
                "id": item_id,
                "category": category,
                "source_model": gemini_meta.source_model,
                "cached": gemini_meta.cached,
                "cutout_area_ratio": round(cutout_area_ratio(rgba_bytes), 4),
            })
        except GeminiImageError as exc:
            errors.append({"category": category, "error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - keep extracting other layers
            errors.append({"category": category, "error": f"unexpected: {exc}"})

    if not ingested_ids:
        raise ValueError(f"Gemini could not extract any clothing from selected person: {errors}")

    result_data = {
        "cutout": {"rgba_crop_path": storage_service.get_relative_path(preview_path)},
        "parsed_parts": [],
        "ingested_items": ingested_ids,
        "gemini_layers": gemini_layers,
        "gemini_errors": errors,
        "gemini_selected_person": True,
    }

    update_job(job_id=job_id, status="completed", result=[result_data])
    upload_cache.delete(f"group_job_{job_id}")
    return result_data
