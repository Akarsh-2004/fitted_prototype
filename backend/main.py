from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from pipeline.io_models import ProcessResponse
from pipeline.mannequin import make_base_mannequin
from pipeline.metrics import encode_png_base64, image_hash
from pipeline.refine_mask import (
    RefineParams,
    compute_mask_quality,
    ensure_nonempty_mask,
    refine_mask,
)
from pipeline.render import composite_with_realism
from pipeline.segment import SegmentConfig, segment_with_fallback
from pipeline.warp import warp_garment_to_mannequin

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="vestir Local Try-On Pipeline", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _read_upload_as_bgr(upload: UploadFile) -> np.ndarray:
    raw = upload.file.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    return image


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/process-single", response_model=ProcessResponse)
def process_single(
    image: UploadFile = File(...),
    edge_feather: int = Form(3),
    morph_kernel: int = Form(3),
    smoothness: float = Form(0.5),
):
    start = time.perf_counter()
    garment_bgr = _read_upload_as_bgr(image)
    garment_key = image_hash(garment_bgr, {"edge_feather": edge_feather, "morph_kernel": morph_kernel, "smoothness": smoothness})
    cache_cutout = CACHE_DIR / f"{garment_key}_cutout.png"
    cache_comp = CACHE_DIR / f"{garment_key}_comp.png"
    cache_mask = CACHE_DIR / f"{garment_key}_mask.png"

    used_fallback = False
    warp_mode = "affine"

    if cache_cutout.exists() and cache_comp.exists() and cache_mask.exists():
        cutout = cv2.imread(str(cache_cutout), cv2.IMREAD_UNCHANGED)
        comp = cv2.imread(str(cache_comp), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(cache_mask), cv2.IMREAD_GRAYSCALE)
        if cutout is None or comp is None or mask is None:
            raise HTTPException(status_code=500, detail="Corrupt cache entry")
        mask = ensure_nonempty_mask(garment_bgr, mask)
        quality = float(compute_mask_quality(mask))
    else:
        seg = segment_with_fallback(garment_bgr, SegmentConfig(cache_dir=CACHE_DIR, request_key=garment_key))
        used_fallback = seg.used_fallback
        params = RefineParams(edge_feather=edge_feather, morph_kernel=morph_kernel, smoothness=smoothness)
        mask = refine_mask(seg.mask, params=params)
        mask = ensure_nonempty_mask(garment_bgr, mask)
        quality = float(compute_mask_quality(mask))

        mannequin = make_base_mannequin(height=max(640, garment_bgr.shape[0] + 180), width=max(480, garment_bgr.shape[1] + 120))
        warped = warp_garment_to_mannequin(garment_bgr, mask, mannequin)
        warp_mode = warped.warp_mode
        cutout = warped.cutout_rgba
        comp = composite_with_realism(mannequin, warped)

        cv2.imwrite(str(cache_cutout), cutout)
        cv2.imwrite(str(cache_comp), comp)
        cv2.imwrite(str(cache_mask), mask)

    elapsed = time.perf_counter() - start
    return ProcessResponse(
        cutout=encode_png_base64(cutout),
        composite=encode_png_base64(comp),
        mask=encode_png_base64(mask),
        meta={
            "mask_quality": round(quality, 4),
            "processing_time": round(elapsed, 3),
            "used_fallback": used_fallback,
            "warp_mode": warp_mode,
            "request_key": garment_key,
        },
    )


@app.post("/refine-single", response_model=ProcessResponse)
def refine_single(
    request_key: str = Form(...),
    edge_feather: int = Form(3),
    morph_kernel: int = Form(3),
    smoothness: float = Form(0.5),
):
    cache_mask = CACHE_DIR / f"{request_key}_mask.png"
    cache_cutout = CACHE_DIR / f"{request_key}_cutout.png"
    cache_comp = CACHE_DIR / f"{request_key}_comp.png"
    if not cache_mask.exists() or not cache_cutout.exists() or not cache_comp.exists():
        raise HTTPException(status_code=404, detail="No cached item for request key")

    base_mask = cv2.imread(str(cache_mask), cv2.IMREAD_GRAYSCALE)
    cutout = cv2.imread(str(cache_cutout), cv2.IMREAD_UNCHANGED)
    comp = cv2.imread(str(cache_comp), cv2.IMREAD_UNCHANGED)
    if base_mask is None or cutout is None or comp is None:
        raise HTTPException(status_code=500, detail="Corrupt cache artifact")

    refined = refine_mask(base_mask, RefineParams(edge_feather=edge_feather, morph_kernel=morph_kernel, smoothness=smoothness))
    refined = ensure_nonempty_mask(cutout[:, :, :3], refined)
    quality = float(compute_mask_quality(refined))
    cv2.imwrite(str(cache_mask), refined)

    return ProcessResponse(
        cutout=encode_png_base64(cutout),
        composite=encode_png_base64(comp),
        mask=encode_png_base64(refined),
        meta={
            "mask_quality": round(quality, 4),
            "processing_time": 0.0,
            "used_fallback": False,
            "warp_mode": "affine",
            "request_key": request_key,
        },
    )
