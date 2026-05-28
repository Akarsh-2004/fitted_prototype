"""Generate a curated starter closet using Gemini image generation.

This replaces the old OpenCV-seeded sample wardrobe with a fully Gemini-driven
flow that lets you wipe the database and regenerate a small, balanced closet
of 12 items (3 hats / 3 tops / 3 bottoms / 3 shoes) in one click.

Each seed prompt is a self-contained spec that lets us skip the YOLO+SAM
detection stack: we know the category, subtype, style and palette ahead of
time, so we drop the generated PNG straight into the crop store and run the
existing Outfit Composer alignment to render the 1024x1024 layered asset.
"""
from __future__ import annotations

import io
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from pipeline.analysis.oklch_scorer import extract_dominant_colors
from pipeline.composer.gemini_image import (
    GeminiImageError,
    generate_apparel_rgba,
)
from pipeline.config import settings
from pipeline.database.storage import (
    delete_wardrobe_item,
    get_all_wardrobe_items,
    insert_wardrobe_item,
)
from pipeline.services.storage_service import storage_service


# ---------------------------------------------------------------------------
# Prompt catalogue (12 items, designed for visual contrast across the slider)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedSpec:
    """Everything the seeder needs to insert one wardrobe item end-to-end."""

    category: str       # composer category: hat | top | bottom | shoes
    garment_type: str   # taxonomy mapping
    subtype: str        # hoodie / cap / trousers / sneakers / ...
    name: str
    prompt: str
    style: str
    material: str
    fit: str
    pattern: str
    occasion: str
    season: str
    archetype: str
    layering_role: str
    brand: str
    tags: List[str]


_PROMPT_BASE = (
    "Front-facing realistic {subtype} apparel cutout, isolated clothing only, "
    "no mannequin, no human, no shadows, PURE solid white #FFFFFF background, "
    "studio ecommerce lighting, highly detailed fabric folds, centered, "
    "symmetrical, fashion catalog style, 1024x1024."
)


def _prompt(subtype: str, detail: str) -> str:
    base = _PROMPT_BASE.format(subtype=subtype)
    return f"{detail}. {base}"


SEED_CATALOGUE: List[SeedSpec] = [
    # --- Tops -------------------------------------------------------------
    SeedSpec(
        category="top",
        garment_type="top",
        subtype="hoodie",
        name="Black Oversized Hoodie",
        prompt=_prompt("oversized hoodie", "Black heavyweight cotton hoodie, oversized fit, drop shoulders, kangaroo pocket, drawcord hood"),
        style="streetwear",
        material="cotton",
        fit="oversized",
        pattern="solid",
        occasion="daily",
        season="all-season",
        archetype="utilitarian",
        layering_role="mid",
        brand="Carhartt WIP",
        tags=["hoodie", "black", "streetwear", "oversized", "cotton"],
    ),
    SeedSpec(
        category="top",
        garment_type="top",
        subtype="t-shirt",
        name="Cream Boxy T-Shirt",
        prompt=_prompt("boxy t-shirt", "Cream off-white heavyweight cotton t-shirt, boxy relaxed fit, ribbed crew neck"),
        style="minimalist",
        material="cotton",
        fit="relaxed",
        pattern="solid",
        occasion="daily",
        season="summer",
        archetype="minimalist",
        layering_role="base",
        brand="Uniqlo",
        tags=["t-shirt", "cream", "minimalist", "boxy", "cotton"],
    ),
    SeedSpec(
        category="top",
        garment_type="top",
        subtype="sweater",
        name="Olive Cable-Knit Sweater",
        prompt=_prompt("cable-knit sweater", "Olive green chunky cable-knit wool sweater, regular fit, ribbed cuffs and hem"),
        style="smart-casual",
        material="wool",
        fit="regular",
        pattern="textured",
        occasion="daily",
        season="winter",
        archetype="classic",
        layering_role="mid",
        brand="Our Legacy",
        tags=["sweater", "olive", "wool", "knit", "cable"],
    ),

    # --- Bottoms ----------------------------------------------------------
    SeedSpec(
        category="bottom",
        garment_type="bottom",
        subtype="jeans",
        name="Indigo Baggy Jeans",
        prompt=_prompt("baggy jeans", "Indigo wash baggy fit denim jeans, wide leg, traditional 5-pocket construction"),
        style="streetwear",
        material="denim",
        fit="oversized",
        pattern="solid",
        occasion="daily",
        season="all-season",
        archetype="skater",
        layering_role="standalone",
        brand="Carhartt WIP",
        tags=["jeans", "indigo", "denim", "baggy", "wide-leg"],
    ),
    SeedSpec(
        category="bottom",
        garment_type="bottom",
        subtype="trousers",
        name="Charcoal Pleated Trousers",
        prompt=_prompt("pleated trousers", "Charcoal grey wool blend pleated trousers, tapered leg, sharp creases"),
        style="smart-casual",
        material="wool",
        fit="regular",
        pattern="solid",
        occasion="formal",
        season="all-season",
        archetype="classic",
        layering_role="standalone",
        brand="Acne Studios",
        tags=["trousers", "charcoal", "wool", "pleated", "tailored"],
    ),
    SeedSpec(
        category="bottom",
        garment_type="bottom",
        subtype="trousers",
        name="Sand Cargo Pants",
        prompt=_prompt("cargo pants", "Sand beige technical cargo pants, multiple utility pockets, ankle cuffs, ripstop fabric"),
        style="gorpcore",
        material="nylon",
        fit="relaxed",
        pattern="solid",
        occasion="outdoor",
        season="spring/autumn",
        archetype="utilitarian",
        layering_role="standalone",
        brand="Arc'teryx",
        tags=["cargo", "sand", "nylon", "utility", "gorpcore"],
    ),

    # --- Shoes ------------------------------------------------------------
    SeedSpec(
        category="shoes",
        garment_type="shoes",
        subtype="sneakers",
        name="White Low-Top Sneakers",
        prompt=_prompt("low-top sneakers", "Pair of white leather low-top sneakers, side view, rubber outsole, minimalist design"),
        style="minimalist",
        material="leather",
        fit="regular",
        pattern="solid",
        occasion="daily",
        season="all-season",
        archetype="minimalist",
        layering_role="standalone",
        brand="Nike",
        tags=["sneakers", "white", "leather", "low-top", "minimalist"],
    ),
    SeedSpec(
        category="shoes",
        garment_type="shoes",
        subtype="boots",
        name="Brown Suede Chelsea Boots",
        prompt=_prompt("chelsea boots", "Pair of cognac brown suede chelsea boots, elasticated side panels, pull tabs"),
        style="smart-casual",
        material="leather",
        fit="regular",
        pattern="solid",
        occasion="daily",
        season="winter",
        archetype="classic",
        layering_role="standalone",
        brand="Acne Studios",
        tags=["boots", "brown", "suede", "chelsea", "smart-casual"],
    ),
    SeedSpec(
        category="shoes",
        garment_type="shoes",
        subtype="sneakers",
        name="Grey Trail Runners",
        prompt=_prompt("trail running sneakers", "Pair of grey and orange chunky trail running sneakers, technical mesh upper, aggressive lugged outsole"),
        style="gorpcore",
        material="nylon",
        fit="regular",
        pattern="graphic",
        occasion="outdoor",
        season="all-season",
        archetype="outdoor-enthusiast",
        layering_role="standalone",
        brand="Arc'teryx",
        tags=["sneakers", "grey", "trail", "gorpcore", "technical"],
    ),

    # --- Hats -------------------------------------------------------------
    SeedSpec(
        category="hat",
        garment_type="accessory",
        subtype="cap",
        name="Black Six-Panel Cap",
        prompt=_prompt("baseball cap", "Black six-panel cotton baseball cap, curved brim, adjustable strap, no logo"),
        style="streetwear",
        material="cotton",
        fit="regular",
        pattern="solid",
        occasion="daily",
        season="all-season",
        archetype="minimalist",
        layering_role="standalone",
        brand="Stussy",
        tags=["cap", "black", "cotton", "baseball", "streetwear"],
    ),
    SeedSpec(
        category="hat",
        garment_type="accessory",
        subtype="beanie",
        name="Oatmeal Ribbed Beanie",
        prompt=_prompt("ribbed beanie", "Oatmeal cream ribbed wool beanie, cuffed brim"),
        style="minimalist",
        material="wool",
        fit="regular",
        pattern="textured",
        occasion="daily",
        season="winter",
        archetype="minimalist",
        layering_role="standalone",
        brand="Acne Studios",
        tags=["beanie", "oatmeal", "wool", "ribbed", "winter"],
    ),
    SeedSpec(
        category="hat",
        garment_type="accessory",
        subtype="bucket",
        name="Olive Bucket Hat",
        prompt=_prompt("bucket hat", "Olive green cotton ripstop bucket hat, wide brim, top-stitched panels"),
        style="gorpcore",
        material="nylon",
        fit="regular",
        pattern="solid",
        occasion="outdoor",
        season="spring/autumn",
        archetype="outdoor-enthusiast",
        layering_role="standalone",
        brand="Stussy",
        tags=["bucket", "olive", "cotton", "hat", "gorpcore"],
    ),
]


# ---------------------------------------------------------------------------
# Wipe + seed orchestration
# ---------------------------------------------------------------------------


def _safe_remove(path: Path) -> None:
    try:
        if path.is_file():
            path.unlink()
    except Exception as exc:  # noqa: BLE001 - best-effort cleanup
        print(f"[seed] failed to remove {path}: {exc}")


def wipe_closet(*, remove_uploads: bool = False) -> Dict[str, Any]:
    """Delete every wardrobe item along with its crop and aligned PNG.

    Originals in ``data/storage/uploads/`` are preserved by default so the
    user does not accidentally lose their reference photos. Set
    ``remove_uploads=True`` to do a full hard reset.
    """
    items = get_all_wardrobe_items()
    crops_dir = settings.storage_dir / "crops"
    aligned_dir = settings.storage_dir / "aligned"
    uploads_dir = settings.storage_dir / "uploads"

    removed_rows = 0
    removed_files = 0

    for item in items:
        item_id = str(item.get("id", "")).strip()
        if not item_id:
            continue
        for candidate in (
            crops_dir / f"{item_id}.png",
            aligned_dir / f"{item_id}.png",
        ):
            if candidate.exists():
                _safe_remove(candidate)
                removed_files += 1
        if delete_wardrobe_item(item_id):
            removed_rows += 1

    # Mop up any orphan PNGs in crops/ and aligned/ that no longer have a row.
    for stray_dir in (crops_dir, aligned_dir):
        if stray_dir.exists():
            for stray in stray_dir.glob("*.png"):
                _safe_remove(stray)
                removed_files += 1

    removed_uploads = 0
    if remove_uploads and uploads_dir.exists():
        for stray in uploads_dir.iterdir():
            if stray.is_file():
                _safe_remove(stray)
                removed_uploads += 1

    return {
        "removed_rows": removed_rows,
        "removed_files": removed_files,
        "removed_uploads": removed_uploads,
    }


def _save_seed_crop(rgba_bytes: bytes, item_id: str) -> Path:
    """Persist the Gemini PNG bytes as a crop PNG keyed to ``item_id``."""
    pil = Image.open(io.BytesIO(rgba_bytes)).convert("RGBA")
    rgba = np.array(pil)  # H,W,4 in RGBA order
    return storage_service.save_crop_rgba(rgba, item_id)


def _load_rgba(path: Path) -> np.ndarray:
    bgra = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgra is None:
        raise RuntimeError(f"Failed to re-read seed crop at {path}")
    if bgra.ndim != 3 or bgra.shape[2] != 4:
        raise RuntimeError(f"Seed crop at {path} is not RGBA")
    return cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)


def _build_item_data(spec: SeedSpec, item_id: str, crop_path: Path, colors: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "id": item_id,
        "garment_type": spec.garment_type,
        "fit": spec.fit,
        "material": spec.material,
        "construction": "knit" if spec.material in ("wool", "knit") else "woven",
        "pattern": spec.pattern,
        "subtype": spec.subtype,
        "brand": spec.brand,
        "style": spec.style,
        "occasion": spec.occasion,
        "season": spec.season,
        "archetype": spec.archetype,
        "layering_role": spec.layering_role,
        "pairing_suggestions": [
            f"Pairs nicely with neutral {spec.style} basics",
            f"Layer with a {spec.season} jacket for cooler days",
            "Anchor of a Gemini-curated starter capsule",
        ],
        "colors": colors,
        "tags": spec.tags + [spec.style, spec.subtype],
        "image_path": storage_service.get_relative_path(crop_path),
        "scene_type": "flat_single",
    }


def seed_closet(
    *,
    per_category: int = 3,
    wipe: bool = True,
    remove_uploads: bool = False,
) -> Dict[str, Any]:
    """Generate ``per_category`` items per composer category using Gemini.

    Returns a structured summary including any per-item errors so the UI can
    surface them. If a single item fails we keep going - partial seeds are
    better than nothing.
    """
    started = time.perf_counter()
    wipe_summary: Optional[Dict[str, Any]] = None
    if wipe:
        wipe_summary = wipe_closet(remove_uploads=remove_uploads)

    # Pick the first ``per_category`` specs for each composer category, in catalog order.
    selected: List[SeedSpec] = []
    by_cat: Dict[str, List[SeedSpec]] = {"hat": [], "top": [], "bottom": [], "shoes": []}
    for spec in SEED_CATALOGUE:
        by_cat.setdefault(spec.category, []).append(spec)
    for cat in ("hat", "top", "bottom", "shoes"):
        selected.extend(by_cat.get(cat, [])[:per_category])

    created: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    for spec in selected:
        try:
            rgba_bytes, original = generate_apparel_rgba(spec.prompt)
            item_id = f"seed_{uuid.uuid4().hex[:8]}"
            crop_path = _save_seed_crop(rgba_bytes, item_id)
            rgba_np = _load_rgba(crop_path)
            colors = extract_dominant_colors(rgba_np, num_colors=3)
            item_data = _build_item_data(spec, item_id, crop_path, colors)
            insert_wardrobe_item(item_id, item_data)
            created.append({
                "id": item_id,
                "name": spec.name,
                "category": spec.category,
                "subtype": spec.subtype,
                "source_model": original.source_model,
                "cached": original.cached,
            })
        except GeminiImageError as exc:
            errors.append({"name": spec.name, "category": spec.category, "error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - report and continue
            errors.append({"name": spec.name, "category": spec.category, "error": f"unexpected: {exc}"})

    return {
        "created": created,
        "errors": errors,
        "wipe": wipe_summary,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "total_requested": len(selected),
        "total_succeeded": len(created),
    }
