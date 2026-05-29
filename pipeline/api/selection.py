from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional
from pipeline.scenarios.group_photo import (
    find_clicked_person,
    ingest_selected_group_person,
    ingest_selected_group_person_gemini,
    ingest_selected_group_person_segformer,
)
from pipeline.database.storage import get_job, update_job

router = APIRouter(prefix="/api", tags=["Selection"])

class ClickRequest(BaseModel):
    job_id: str
    x: float
    y: float

def run_s4_person_ingestion(job_id: str, person_index: int):
    try:
        update_job(job_id=job_id, status="processing")
        ingest_selected_group_person(job_id, person_index)
    except Exception as e:
        import traceback
        traceback.print_exc()
        update_job(job_id=job_id, status="failed", error=f"Group person clothing ingestion failed: {str(e)}")


def run_s4_person_gemini_ingestion(job_id: str, person_index: int):
    try:
        update_job(job_id=job_id, status="processing")
        ingest_selected_group_person_gemini(job_id, person_index)
    except Exception as e:
        import traceback
        traceback.print_exc()
        update_job(job_id=job_id, status="failed", error=f"Gemini selected-person ingestion failed: {str(e)}")


def run_s4_person_segformer_ingestion(job_id: str, person_index: int):
    try:
        update_job(job_id=job_id, status="processing")
        ingest_selected_group_person_segformer(job_id, person_index)
    except Exception as e:
        import traceback
        traceback.print_exc()
        update_job(job_id=job_id, status="failed", error=f"SegFormer selected-person ingestion failed: {str(e)}")

@router.post("/select-person")
def select_person_by_click(request: ClickRequest, background_tasks: BackgroundTasks):
    """
    Identifies which person's mask in a group photo contains the clicked pixel.
    If matched, kicks off the layered garment segmentation process in a background task.
    """
    job = get_job(request.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    if job["status"] != "requires_selection":
        raise HTTPException(
            status_code=400,
            detail=f"Job status is {job['status']}; selection is only valid when status is 'requires_selection'."
        )
        
    # Match click coordinate (x, y) to a cached person
    matched_idx = find_clicked_person(request.job_id, request.x, request.y)
    
    if matched_idx is None:
        return {
            "matched": False,
            "message": "Click did not hit any detected individuals. Please try clicking closer to a person's center."
        }
        
    # Spawn background task to ingest the selected person's wardrobe layers
    background_tasks.add_task(run_s4_person_ingestion, request.job_id, matched_idx)
    
    return {
        "matched": True,
        "person_index": matched_idx,
        "status": "processing",
        "message": f"Successfully matched clicked person {matched_idx}. Processing clothing layers..."
    }


@router.post("/select-person/gemini")
def select_person_by_click_gemini(request: ClickRequest, background_tasks: BackgroundTasks):
    """Select a detected person, then use Gemini to extract outfit layers."""
    job = get_job(request.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] != "requires_selection":
        raise HTTPException(
            status_code=400,
            detail=f"Job status is {job['status']}; selection is only valid when status is 'requires_selection'."
        )

    matched_idx = find_clicked_person(request.job_id, request.x, request.y)

    if matched_idx is None:
        return {
            "matched": False,
            "message": "Click did not hit any detected individuals. Please try clicking closer to a person's center."
        }

    background_tasks.add_task(run_s4_person_gemini_ingestion, request.job_id, matched_idx)

    return {
        "matched": True,
        "person_index": matched_idx,
        "status": "processing",
        "message": f"Successfully matched clicked person {matched_idx}. Gemini is extracting outfit layers..."
    }


@router.post("/select-person/segformer")
def select_person_by_click_segformer(request: ClickRequest, background_tasks: BackgroundTasks):
    """Select a detected person, then use SegFormer clothes labels to parse outfit layers."""
    job = get_job(request.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] != "requires_selection":
        raise HTTPException(
            status_code=400,
            detail=f"Job status is {job['status']}; selection is only valid when status is 'requires_selection'."
        )

    matched_idx = find_clicked_person(request.job_id, request.x, request.y)

    if matched_idx is None:
        return {
            "matched": False,
            "message": "Click did not hit any detected individuals. Please try clicking closer to a person's center."
        }

    background_tasks.add_task(run_s4_person_segformer_ingestion, request.job_id, matched_idx)

    return {
        "matched": True,
        "person_index": matched_idx,
        "status": "processing",
        "message": f"Successfully matched clicked person {matched_idx}. SegFormer is parsing outfit layers..."
    }
