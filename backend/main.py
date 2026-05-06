from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("vestir.pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

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
PIPELINE_VERSION = "2026-05-06-v4-fast-cpu-seg"

app = FastAPI(title="vestir Local Try-On Pipeline", version="0.2.0")
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


def _mask_stats(mask: np.ndarray) -> dict[str, float]:
    fg = (mask > 10).astype(np.uint8)
    h, w = fg.shape
    area_ratio = float(fg.mean())
    ys, xs = np.where(fg > 0)
    if xs.size == 0:
        return {
            "area_ratio": 0.0,
            "bbox_ratio": 0.0,
            "border_ratio": 1.0,
            "compactness": 0.0,
            "component_count": 0.0,
            "fill_ratio": 0.0,
        }
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bbox_area = float((x1 - x0 + 1) * (y1 - y0 + 1))
    bbox_ratio = float(bbox_area / float(h * w))
    border_px = np.concatenate((fg[0, :], fg[-1, :], fg[:, 0], fg[:, -1]))
    border_ratio = float(border_px.mean())

    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_area = 0.0
    compactness = 0.0
    component_count = 0.0
    if contours:
        big_threshold = max(64.0, 0.005 * float(h * w))
        component_count = float(sum(1 for c in contours if cv2.contourArea(c) >= big_threshold))
        main = max(contours, key=cv2.contourArea)
        main_area = float(cv2.contourArea(main))
        perimeter = float(cv2.arcLength(main, True)) + 1e-5
        compactness = float(4.0 * np.pi * main_area / (perimeter * perimeter))

    fill_ratio = float(main_area / bbox_area) if bbox_area > 0 else 0.0
    return {
        "area_ratio": area_ratio,
        "bbox_ratio": bbox_ratio,
        "border_ratio": border_ratio,
        "compactness": compactness,
        "component_count": component_count,
        "fill_ratio": fill_ratio,
    }


def _looks_like_outdoor_scene(image_bgr: np.ndarray) -> bool:
    """Detect sky-dominated upper region (blue + low saturation) typical of person-outdoor shots."""
    h, w = image_bgr.shape[:2]
    upper = image_bgr[: max(1, h // 3)]
    hsv = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    sky_mask = (
        ((hue >= 90) & (hue <= 130) & (sat > 25) & (val > 140))
        | ((sat < 35) & (val > 200))
    )
    return float(sky_mask.mean()) > 0.55


def _ensure_likely_garment(mask: np.ndarray, quality: float, image_bgr: np.ndarray) -> None:
    s = _mask_stats(mask)
    reasons: list[str] = []
    if quality < 0.18:
        reasons.append(f"low mask quality ({quality:.2f})")
    if s["area_ratio"] < 0.03:
        reasons.append(f"garment too small in frame ({s['area_ratio'] * 100:.1f}%)")
    if s["bbox_ratio"] > 0.94:
        reasons.append("subject fills the entire frame; add some padding")
    if s["border_ratio"] > 0.5:
        reasons.append("foreground touches image borders (looks like a full scene)")
    if s["component_count"] >= 2:
        reasons.append(
            f"multiple distinct regions detected ({int(s['component_count'])}); expected a single garment"
        )
    if s["compactness"] < 0.18:
        reasons.append("foreground shape is too irregular for a single garment")
    if s["fill_ratio"] < 0.4 and s["bbox_ratio"] > 0.4:
        reasons.append("foreground is sparse inside its bounding box")
    if _looks_like_outdoor_scene(image_bgr):
        reasons.append("sky/scene detected in the upper region")
    if reasons:
        logger.info(
            "Garment guardrail rejected upload: %s | stats=%s quality=%.3f",
            "; ".join(reasons),
            s,
            quality,
        )
        raise HTTPException(
            status_code=422,
            detail=(
                "Could not isolate a clothing item: "
                + "; ".join(reasons)
                + ". Use a flat-lay or product photo with a plain background."
            ),
        )
    logger.info(
        "Garment accepted: stats=%s quality=%.3f",
        s,
        quality,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "pipeline_version": PIPELINE_VERSION}


@app.post("/process-single", response_model=ProcessResponse)
def process_single(
    image: UploadFile = File(...),
    edge_feather: int = Form(3),
    morph_kernel: int = Form(3),
    smoothness: float = Form(0.5),
):
    start = time.perf_counter()
    garment_bgr = _read_upload_as_bgr(image)
    garment_key = image_hash(
        garment_bgr,
        {
            "edge_feather": int(edge_feather),
            "morph_kernel": int(morph_kernel),
            "smoothness": float(smoothness),
            "pipeline_version": PIPELINE_VERSION,
        },
    )
    cache_cutout = CACHE_DIR / f"{garment_key}_cutout.png"
    cache_comp = CACHE_DIR / f"{garment_key}_comp.png"
    cache_mask = CACHE_DIR / f"{garment_key}_mask.png"

    used_fallback = False
    seg_method = "cached"
    warp_mode = "similarity"

    if cache_cutout.exists() and cache_comp.exists() and cache_mask.exists():
        cutout = cv2.imread(str(cache_cutout), cv2.IMREAD_UNCHANGED)
        comp = cv2.imread(str(cache_comp), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(cache_mask), cv2.IMREAD_GRAYSCALE)
        if cutout is None or comp is None or mask is None:
            raise HTTPException(status_code=500, detail="Corrupt cache entry")
        quality = float(compute_mask_quality(mask))
        _ensure_likely_garment(mask, quality, garment_bgr)
    else:
        seg = segment_with_fallback(
            garment_bgr,
            SegmentConfig(cache_dir=CACHE_DIR, request_key=garment_key),
        )
        used_fallback = seg.used_fallback
        seg_method = seg.method

        params = RefineParams(
            edge_feather=int(edge_feather),
            morph_kernel=int(morph_kernel),
            smoothness=float(smoothness),
        )
        mask = refine_mask(seg.mask, params=params, image_bgr=garment_bgr)
        mask = ensure_nonempty_mask(garment_bgr, mask)
        quality = float(compute_mask_quality(mask))
        _ensure_likely_garment(mask, quality, garment_bgr)

        mannequin = make_base_mannequin(
            height=max(720, garment_bgr.shape[0] + 200),
            width=max(540, garment_bgr.shape[1] + 140),
        )
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
            "segmentation_method": seg_method,
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

    cutout = cv2.imread(str(cache_cutout), cv2.IMREAD_UNCHANGED)
    comp = cv2.imread(str(cache_comp), cv2.IMREAD_UNCHANGED)
    if cutout is None or comp is None:
        raise HTTPException(status_code=500, detail="Corrupt cache artifact")
    if cutout.ndim != 3 or cutout.shape[2] != 4:
        raise HTTPException(status_code=500, detail="Cached cutout missing alpha channel")

    cutout_rgb = cutout[:, :, :3]
    cutout_alpha = cutout[:, :, 3]

    refined_alpha = refine_mask(
        cutout_alpha,
        RefineParams(
            edge_feather=int(edge_feather),
            morph_kernel=int(morph_kernel),
            smoothness=float(smoothness),
        ),
        image_bgr=cutout_rgb,
    )
    refined_alpha = ensure_nonempty_mask(cutout_rgb, refined_alpha)
    quality = float(compute_mask_quality(refined_alpha))

    cutout[:, :, 3] = refined_alpha
    cv2.imwrite(str(cache_cutout), cutout)
    cv2.imwrite(str(cache_mask), refined_alpha)

    return ProcessResponse(
        cutout=encode_png_base64(cutout),
        composite=encode_png_base64(comp),
        mask=encode_png_base64(refined_alpha),
        meta={
            "mask_quality": round(quality, 4),
            "processing_time": 0.0,
            "used_fallback": False,
            "warp_mode": "similarity",
            "request_key": request_key,
            "segmentation_method": "refine_only",
        },
    )
