import io
import uuid
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pipeline.config import settings
from pipeline.detectors.scene_classifier import scene_classifier
from pipeline.scenarios.single_flat import process_single_flat_lay
from pipeline.scenarios.multi_flat import detect_multi_flat_lay
from pipeline.scenarios.single_worn import process_single_worn_outfit
from pipeline.scenarios.group_photo import detect_group_photo
from pipeline.detectors.yolo_detector import yolo_detector
from pipeline.services.storage_service import storage_service
from pipeline.database.storage import create_job, update_job, insert_wardrobe_item

# Gemini-only fast path
from pipeline.composer.gemini_image import (
    GeminiImageError,
    extract_cutout_rgba,
    cutout_area_ratio,
)
from pipeline.analysis.oklch_scorer import extract_dominant_colors
from pipeline.analysis.gemini_client import gemini_client
from pipeline.composer.alignment import resolve_composer_category

router = APIRouter(prefix="/api", tags=["Upload"])

def run_s1_background(img_path: Path, job_id: str):
    try:
        update_job(job_id=job_id, status="processing")
        item = process_single_flat_lay(img_path, scene_type="flat_single")
        update_job(job_id=job_id, status="completed", result=[item])
    except Exception as e:
        import traceback
        traceback.print_exc()
        update_job(job_id=job_id, status="failed", error=str(e))

def run_s3_background(img_path: Path, job_id: str):
    try:
        update_job(job_id=job_id, status="processing")
        items = process_single_worn_outfit(img_path, scene_type="single_person")
        update_job(job_id=job_id, status="completed", result=items)
    except Exception as e:
        import traceback
        traceback.print_exc()
        update_job(job_id=job_id, status="failed", error=str(e))

@router.post("/upload")
async def upload_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Accepts an uploaded wardrobe image, auto-classifies the scenario type,
    creates a status job tracker, and routes the processing.
    """
    try:
        # Read file bytes
        file_bytes = await file.read()
        
        # Save original upload to disk
        img_path = storage_service.save_upload(file_bytes, file.filename)
        
        # Load image into memory for classification
        img = storage_service.load_image(img_path)
        h, w = img.shape[:2]
        
        # Run Auto Scene Router
        scene_type, metadata, processed_img = scene_classifier.classify_scene(img)
        
        # Generate unique Job tracking ID
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        
        # Create Job in SQLite database
        create_job(
            job_id=job_id,
            status="queued",
            scene_type=scene_type,
            original_image_path=storage_service.get_relative_path(img_path)
        )
        
        # Route scenario logic
        if scene_type == "flat_single":
            # Run S1 background pipeline
            background_tasks.add_task(run_s1_background, img_path, job_id)
            
        elif scene_type == "flat_multi":
            # Run S2 initial detection (sync/fast) to return preview confirmation coordinates immediately
            try:
                detect_multi_flat_lay(img_path, job_id)
            except Exception as e:
                update_job(job_id=job_id, status="failed", error=f"Multi flat-lay detection failed: {str(e)}")
                
        elif scene_type == "single_person":
            # Run S3 background pipeline
            background_tasks.add_task(run_s3_background, img_path, job_id)
            
        elif scene_type == "group_photo":
            # Run S4 initial detection (sync/fast) to display interactive canvas immediately
            try:
                detect_group_photo(img_path, job_id)
            except Exception as e:
                update_job(job_id=job_id, status="failed", error=f"Group photo detection failed: {str(e)}")
                
        return {
            "job_id": job_id,
            "scene_type": scene_type,
            "original_image_url": storage_service.get_relative_path(img_path),
            "dimensions": {"width": w, "height": h},
            "counts": {
                "faces": metadata["face_count"],
                "people": metadata["person_count"],
                "garments": metadata["garment_count"]
            }
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Image upload & routing failed: {str(e)}")

@router.post("/upload/bulk")
async def upload_images_bulk(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
):
    """
    Accepts multiple uploaded wardrobe images concurrently, auto-classifies
    and tracks separate pipeline jobs for each file in parallel.
    """
    import uuid
    
    responses = []
    
    for file in files:
        try:
            # Read file bytes
            file_bytes = await file.read()
            
            # Save original upload to disk
            img_path = storage_service.save_upload(file_bytes, file.filename)
            
            # Load image into memory for classification
            img = storage_service.load_image(img_path)
            h, w = img.shape[:2]
            
            # Run Auto Scene Router
            scene_type, metadata, processed_img = scene_classifier.classify_scene(img)
            
            # Generate unique Job tracking ID
            job_id = f"job_{uuid.uuid4().hex[:12]}"
            
            # Create Job in SQLite database
            create_job(
                job_id=job_id,
                status="queued",
                scene_type=scene_type,
                original_image_path=storage_service.get_relative_path(img_path)
            )
            
            # Route scenario logic
            if scene_type == "flat_single":
                background_tasks.add_task(run_s1_background, img_path, job_id)
                
            elif scene_type == "flat_multi":
                try:
                    detect_multi_flat_lay(img_path, job_id)
                except Exception as e:
                    update_job(job_id=job_id, status="failed", error=f"Multi flat-lay detection failed: {str(e)}")
                    
            elif scene_type == "single_person":
                background_tasks.add_task(run_s3_background, img_path, job_id)
                
            elif scene_type == "group_photo":
                try:
                    detect_group_photo(img_path, job_id)
                except Exception as e:
                    update_job(job_id=job_id, status="failed", error=f"Group photo detection failed: {str(e)}")
                    
            responses.append({
                "job_id": job_id,
                "scene_type": scene_type,
                "original_image_url": storage_service.get_relative_path(img_path),
                "filename": file.filename,
                "dimensions": {"width": w, "height": h},
                "counts": {
                    "faces": metadata["face_count"],
                    "people": metadata["person_count"],
                    "garments": metadata["garment_count"]
                }
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            
    return responses


# ---------------------------------------------------------------------------
# Gemini Cutout (Fast) - new default upload mode
# ---------------------------------------------------------------------------


_CATEGORY_HINTS = {
    "hat": "headwear (cap, beanie, hat, bucket hat)",
    "top": "upper-body garment (t-shirt, hoodie, shirt, sweater, jacket)",
    "bottom": "lower-body garment (pants, jeans, trousers, shorts, skirt)",
    "shoes": "pair of shoes",
}


def _persist_gemini_crop(rgba_bytes: bytes, item_id: str) -> Path:
    """Save the keyed RGBA PNG via the standard crop store path."""
    pil = Image.open(io.BytesIO(rgba_bytes)).convert("RGBA")
    rgba_np = np.array(pil)
    return storage_service.save_crop_rgba(rgba_np, item_id)


def _load_crop_rgba(path: Path) -> np.ndarray:
    bgra = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgra is None or bgra.ndim != 3 or bgra.shape[2] != 4:
        raise HTTPException(status_code=500, detail="Failed to re-read Gemini crop as RGBA")
    return cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)


def _ingest_gemini_bytes(
    file_bytes: bytes,
    filename: str,
    category_hint: Optional[str] = None,
    *,
    original_path: Optional[Path] = None,
    scene_type: str = "flat_single",
) -> dict:
    """Shared Gemini cutout ingestion used by direct upload and staged flows."""
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty upload")

    # Save the original so the user can audit / fall back to legacy later.
    if original_path is None:
        original_path = storage_service.save_upload(file_bytes, filename or "upload.png")

    hint = (category_hint or "").strip().lower()
    if hint and hint not in _CATEGORY_HINTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown category_hint '{category_hint}'. Use hat | top | bottom | shoes.",
        )
    hint_text = _CATEGORY_HINTS.get(hint) if hint else None

    try:
        rgba_bytes, gemini_meta = extract_cutout_rgba(file_bytes, category_hint=hint_text)
    except GeminiImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    item_id = f"gem_{uuid.uuid4().hex[:8]}"
    try:
        crop_path = _persist_gemini_crop(rgba_bytes, item_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to save Gemini crop: {exc}")

    rgba_np = _load_crop_rgba(crop_path)
    crop_h, crop_w = rgba_np.shape[:2]
    aspect_ratio = float(crop_h) / float(crop_w) if crop_w else 1.0
    dominant_colors = extract_dominant_colors(rgba_np, num_colors=3)

    # Color names as proxy tags (we skip the heavy SigLIP forward pass on the fast path).
    rough_tags: List[str] = []
    if dominant_colors:
        r, g, b = dominant_colors[0]["rgb"]
        rough_tags.append("dark" if (r + g + b) / 3 < 80 else "light" if (r + g + b) / 3 > 200 else "midtone")

    layer_hint = None
    if hint == "hat":
        layer_hint = "hat"
    elif hint == "top":
        layer_hint = "upper"
    elif hint == "bottom":
        layer_hint = "lower"
    elif hint == "shoes":
        layer_hint = "shoes"

    try:
        fashion_meta = gemini_client.analyze_garment(rough_tags, dominant_colors, aspect_ratio, layer_hint=layer_hint)
    except Exception as exc:  # noqa: BLE001 - analyzer has its own offline fallback
        print(f"[upload/gemini] taxonomy analyzer failed, using minimal metadata: {exc}")
        fashion_meta = {
            "garment_type": "top",
            "fit": "regular",
            "material": "cotton",
            "construction": "woven",
            "pattern": "solid",
            "subtype": hint or "garment",
            "brand": "Uniqlo",
            "style": "minimalist",
            "occasion": "daily",
            "season": "all-season",
            "archetype": "minimalist",
            "layering_role": "standalone",
            "pairing_suggestions": [],
        }

    # Honor the user's explicit category hint over whatever Gemini guessed.
    if hint:
        if hint == "hat":
            fashion_meta["garment_type"] = "accessory"
            if not any(k in (fashion_meta.get("subtype") or "").lower() for k in ("cap", "hat", "beanie", "bucket")):
                fashion_meta["subtype"] = "cap"
        elif hint == "top":
            fashion_meta["garment_type"] = "top"
        elif hint == "bottom":
            fashion_meta["garment_type"] = "bottom"
        elif hint == "shoes":
            fashion_meta["garment_type"] = "shoes"

    item_data = {
        "id": item_id,
        "colors": dominant_colors,
        "tags": list(set(rough_tags + [fashion_meta.get("subtype", "garment"), fashion_meta.get("material", "cotton")])),
        "image_path": storage_service.get_relative_path(crop_path),
        "scene_type": scene_type,
        **fashion_meta,
    }
    insert_wardrobe_item(item_id, item_data)

    composer_category = resolve_composer_category(
        item_data.get("garment_type"),
        item_data.get("subtype"),
    )

    return {
        "id": item_id,
        "item": item_data,
        "composer_category": composer_category,
        "original_image_url": storage_service.get_relative_path(original_path),
        "crop_url": storage_service.get_relative_path(crop_path),
        "gemini": {
            "source_model": gemini_meta.source_model,
            "cached": gemini_meta.cached,
            "cutout_area_ratio": round(cutout_area_ratio(rgba_bytes), 4),
        },
    }


@router.post("/upload/gemini-stage")
async def upload_image_via_gemini_stage(
    file: UploadFile = File(...),
    category_hint: Optional[str] = Form(None),
):
    """Hybrid Gemini upload.

    If people are visible, create the same Stage-1 person-selection job the
    legacy pipeline uses so the frontend can show the uploaded image and hover
    over people. If no people are visible, fall back to direct Gemini cutout.
    """
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty upload")

    original_path = storage_service.save_upload(file_bytes, file.filename or "upload.png")
    img = storage_service.load_image(original_path)
    h, w = img.shape[:2]
    people = yolo_detector.detect_people(img)

    if people:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        create_job(
            job_id=job_id,
            status="queued",
            scene_type="group_photo",
            original_image_path=storage_service.get_relative_path(original_path),
        )
        stage = detect_group_photo(original_path, job_id)
        return {
            "mode": "select_person",
            "job_id": job_id,
            "scene_type": "group_photo",
            "original_image_url": stage.get("image_url") or storage_service.get_relative_path(original_path),
            "dimensions": {"width": w, "height": h},
            "counts": {"faces": 0, "people": len(stage.get("people", people)), "garments": 0},
            "detected_items": stage.get("people", []),
            "message": "People detected. Select a person and Gemini will extract their outfit.",
        }

    direct = _ingest_gemini_bytes(
        file_bytes,
        file.filename or "upload.png",
        category_hint=category_hint,
        original_path=original_path,
    )
    return {"mode": "direct", **direct}


@router.post("/upload/gemini")
async def upload_image_via_gemini(
    file: UploadFile = File(...),
    category_hint: Optional[str] = Form(None),
):
    """Single-request Gemini cutout pipeline.

    This endpoint is still available for pure direct extraction. The frontend
    now uses ``/upload/gemini-stage`` so photos containing people can keep the
    Stage-1 hover/click selection UX.
    """
    file_bytes = await file.read()
    return _ingest_gemini_bytes(file_bytes, file.filename or "upload.png", category_hint=category_hint)
