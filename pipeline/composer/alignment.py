"""Canvas alignment for the Outfit Composer.

Every wardrobe item with an RGBA cutout is normalized onto a fixed
1024x1024 transparent canvas so the frontend can stack four PNGs of
identical size without any per-item geometry.

Each slot specifies which axis to anchor against (``height`` or ``width``)
plus a target px and a hard cap on the perpendicular axis. The mix is
deliberate:

* **Hat / top / bottom / shoes** anchor by *height* so each body section
  lands in a predictable vertical band.
* **Bottoms** also get a minimum visual width so trousers don't render as
  skinny slivers when Gemini returns a tall/narrow pants bbox.

Adjacent slots overlap by ~40-120px so the shirt hem covers the pant
waistband and the pant cuff covers the shoe tongue - matching a real
layered outfit silhouette.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from pipeline.config import settings

CANVAS_SIZE: int = 1024
ALPHA_THRESHOLD: int = 8

# Category -> { y, anchor, size, cap, min_width? }
#   y       : top-left Y at which the bbox starts on the canvas
#   anchor  : "height" or "width" - which axis to scale to fit
#   size    : target px along the anchored axis
#   cap     : max px along the perpendicular axis (avoids slot overflow)
#   min_width: optional post-scale visual width floor. Used for pants because
#              Gemini often produces a very tall/narrow alpha bbox.
CATEGORY_LAYOUT: Dict[str, Dict[str, Any]] = {
    "hat":    {"y":  45, "anchor": "height", "size": 145, "cap": 280},
    "top":    {"y": 185, "anchor": "height", "size": 315, "cap": 420},
    "bottom": {"y": 430, "anchor": "height", "size": 390, "cap": 360, "min_width": 300},
    "shoes":  {"y": 805, "anchor": "height", "size": 145, "cap": 340},
}

# Subtype hints used to promote 'accessory' items into the hat slot.
HAT_SUBTYPE_KEYWORDS = ("cap", "hat", "beanie", "headwear", "bucket")


def _aligned_dir() -> Path:
    target = settings.storage_dir / "aligned"
    target.mkdir(parents=True, exist_ok=True)
    return target


def aligned_path_for(item_id: str) -> Path:
    """Absolute filesystem path for an item's aligned PNG."""
    return _aligned_dir() / f"{item_id}.png"


def aligned_relative_path(item_id: str) -> str:
    """Path stored in the DB; relative to settings.base_dir so it can be
    served by the /data/storage static mount."""
    abs_path = aligned_path_for(item_id)
    try:
        return str(abs_path.relative_to(settings.base_dir)).replace("\\", "/")
    except ValueError:
        return str(abs_path).replace("\\", "/")


def resolve_composer_category(garment_type: Optional[str], subtype: Optional[str]) -> Optional[str]:
    """Map a wardrobe item's taxonomy to one of the 4 composer categories.

    Returns None if the item does not belong in any composer slider
    (e.g. bags, wrist accessories).
    """
    gt = (garment_type or "").lower().strip()
    st = (subtype or "").lower().strip()

    if gt == "top":
        return "top"
    if gt == "outerwear":
        # Outerwear shares the top slot so it still surfaces in the composer.
        return "top"
    if gt == "bottom":
        return "bottom"
    if gt == "shoes":
        return "shoes"
    if gt == "accessory" and any(k in st for k in HAT_SUBTYPE_KEYWORDS):
        return "hat"
    return None


def _alpha_bbox(rgba: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Tight bounding box of non-transparent pixels. (left, top, right, bottom)."""
    if rgba.ndim != 3 or rgba.shape[2] < 4:
        return None
    alpha = rgba[:, :, 3]
    mask = alpha > ALPHA_THRESHOLD
    if not mask.any():
        return None
    ys = np.where(mask.any(axis=1))[0]
    xs = np.where(mask.any(axis=0))[0]
    return int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1


def _solve_scale(src_w: int, src_h: int, anchor: str, size: int, cap: int) -> float:
    """Compute the uniform scale that puts the source bbox in its slot.

    Scales the source so the anchored axis hits ``size`` px, then clamps if
    the perpendicular axis would exceed ``cap`` px.
    """
    if src_w <= 0 or src_h <= 0:
        return 1.0
    if anchor == "width":
        scale = size / float(src_w)
        if src_h * scale > cap:
            scale = cap / float(src_h)
    else:  # default: anchor by height
        scale = size / float(src_h)
        if src_w * scale > cap:
            scale = cap / float(src_w)
    return scale


def align_to_canvas(rgba_source: Path, category: str, item_id: str) -> Path:
    """Crop the source RGBA cutout to its alpha bbox, resize it according to
    the category's anchor axis, and paste it centered on a 1024x1024
    transparent canvas at the category Y anchor.

    Returns the absolute path to the saved aligned PNG.
    """
    if category not in CATEGORY_LAYOUT:
        raise ValueError(f"Unknown composer category: {category!r}")

    layout = CATEGORY_LAYOUT[category]
    target_y = int(layout["y"])
    anchor = str(layout["anchor"])
    target_size = int(layout["size"])
    cap = int(layout["cap"])
    min_width = int(layout.get("min_width", 0))

    rgba_source = Path(rgba_source)
    if not rgba_source.exists():
        raise FileNotFoundError(rgba_source)

    src = Image.open(rgba_source).convert("RGBA")
    arr = np.array(src)
    bbox = _alpha_bbox(arr)
    if bbox is None:
        # Nothing visible; fall back to the whole image bounds.
        bbox = (0, 0, src.width, src.height)

    left, top, right, bottom = bbox
    cropped = src.crop((left, top, right, bottom))

    src_w, src_h = cropped.size
    scale = _solve_scale(src_w, src_h, anchor, target_size, cap)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))

    # Preserve garment height for the body stack, but widen extra-narrow pants
    # so trousers visually match top width. This is intentionally only a
    # horizontal correction; vertical scaling still respects the slot cap.
    if min_width and new_w < min_width:
        new_w = min(min_width, CANVAS_SIZE)

    resized = cropped.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
    paste_x = (CANVAS_SIZE - new_w) // 2
    paste_y = target_y
    canvas.alpha_composite(resized, (paste_x, paste_y))

    out_path = aligned_path_for(item_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG", optimize=True)
    return out_path


def align_item(item: Dict[str, object]) -> Optional[Tuple[Path, str]]:
    """Convenience helper that pulls source path + category out of a wardrobe
    item record and runs alignment. Returns (aligned_abs_path, category) or
    None if the item is not composable.
    """
    item_id = str(item.get("id") or "")
    if not item_id:
        return None

    category = resolve_composer_category(
        item.get("garment_type") if isinstance(item.get("garment_type"), str) else None,
        item.get("subtype") if isinstance(item.get("subtype"), str) else None,
    )
    if category is None:
        return None

    image_rel = item.get("image_path")
    if not isinstance(image_rel, str) or not image_rel:
        return None

    src_path = settings.base_dir / image_rel
    if not src_path.exists():
        return None

    aligned_abs = align_to_canvas(src_path, category, item_id)
    return aligned_abs, category


def realign_all() -> Dict[str, object]:
    """Regenerate aligned PNGs for every composable item in the DB.

    Returns a summary dict with counts and any per-item errors.
    Lazy-imported storage helpers avoid a circular import at module load.
    """
    from pipeline.database.storage import (
        get_all_wardrobe_items,
        update_wardrobe_item_alignment,
    )

    items = get_all_wardrobe_items()
    aligned = 0
    skipped = 0
    errors: List[Dict[str, str]] = []

    for item in items:
        try:
            result = align_item(item)
            if result is None:
                skipped += 1
                continue
            aligned_abs, category = result
            rel = aligned_relative_path(str(item["id"]))
            update_wardrobe_item_alignment(str(item["id"]), rel, category)
            aligned += 1
        except Exception as exc:  # noqa: BLE001 - report and continue
            errors.append({"id": str(item.get("id", "?")), "error": str(exc)})

    return {
        "total": len(items),
        "aligned": aligned,
        "skipped_not_composable": skipped,
        "errors": errors,
    }
