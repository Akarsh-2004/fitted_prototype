from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pipeline.database.storage import get_job, update_job
from pipeline.scenarios.multi_flat import ingest_confirmed_multi_items

router = APIRouter(prefix="/api", tags=["Processing"])

class ConfirmationRequest(BaseModel):
    job_id: str
    confirmed_indices: List[int]

def run_s2_batch_ingestion(job_id: str, confirmed_indices: List[int]):
    try:
        update_job(job_id=job_id, status="processing")
        ingest_confirmed_multi_items(job_id, confirmed_indices)
    except Exception as e:
        import traceback
        traceback.print_exc()
        update_job(job_id=job_id, status="failed", error=f"Batch ingestion failed: {str(e)}")

@router.get("/job/{job_id}")
def get_job_status(job_id: str):
    """
    Returns the processing status of a background job.
    Used by the frontend to poll status updates.
    """
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    res = {
        "job_id": job["job_id"],
        "status": job["status"],
        "scene_type": job["scene_type"],
        "original_image_url": job["original_image_path"],
        "detected_items": job["detected_items"],
        "result": job["result"],
        "error": job["error"],
        "created_at": job["created_at"]
    }
    
    if job["status"] == "completed" and job["scene_type"] == "group_photo" and job["result"]:
        res_data = job["result"][0] if isinstance(job["result"], list) and len(job["result"]) > 0 else job["result"]
        if isinstance(res_data, dict):
            if "cutout" in res_data and isinstance(res_data["cutout"], dict):
                res["cutout_url"] = res_data["cutout"].get("rgba_crop_path")
            if "parsed_parts" in res_data:
                res["parsed_parts"] = res_data["parsed_parts"]
            
            # Map item IDs back to detailed WardrobeItem objects for UI rendering compatibility
            from pipeline.database.storage import get_wardrobe_item
            ingested_ids = res_data.get("ingested_items", [])
            detailed_items = []
            for item_id in ingested_ids:
                item_obj = get_wardrobe_item(item_id)
                if item_obj:
                    detailed_items.append(item_obj)
            res["result"] = detailed_items
                
    return res



@router.post("/confirm-items")
def confirm_multi_items(request: ConfirmationRequest, background_tasks: BackgroundTasks):
    """
    Kicks off background parallel/sequential ingestion for the user-selected
    flat lay items.
    """
    job = get_job(request.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    if job["status"] != "requires_confirmation":
        raise HTTPException(
            status_code=400,
            detail=f"Job status is {job['status']}; confirmation is only valid when status is 'requires_confirmation'."
        )
        
    # Queue the full ingestion as a background task
    background_tasks.add_task(run_s2_batch_ingestion, request.job_id, request.confirmed_indices)
    
    return {"status": "processing", "message": "Batch garment ingestion started."}

@router.get("/jobs")
def get_all_jobs_endpoint():
    """
    Returns the list of all background processing jobs.
    Used by the frontend to render the batch processing queue.
    """
    from pipeline.database.storage import get_all_jobs
    raw_jobs = get_all_jobs()
    
    mapped_jobs = []
    for job in raw_jobs:
        mapped_jobs.append({
            "job_id": job["job_id"],
            "status": job["status"],
            "scene_type": job["scene_type"],
            "original_image_url": job.get("original_image_path") or job.get("original_image_url"),
            "detected_items": job.get("detected_items"),
            "result": job.get("result"),
            "error": job.get("error"),
            "created_at": job.get("created_at")
        })
    return mapped_jobs
