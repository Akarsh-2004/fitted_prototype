"""Gemini-powered image generation + cutout extraction for the Outfit Composer.

Two operations:

1. ``generate_apparel(prompt, category)`` - text-to-image: generates a fashion
   catalog photo of a single garment on a pure white background. Used to seed
   the closet without uploading real photos.

2. ``extract_cutout(image_bytes, category_hint)`` - image-to-image: takes an
   uploaded photo (worn outfit, flat-lay, anything containing clothing) and
   asks Gemini to redraw the matching garment as an isolated apparel cutout on
   a pure white background. Used by the new "Gemini Cutout (Fast)" upload mode.

Both flows return RGBA bytes by piping the Gemini PNG output through
:func:`key_out_background`, which floods the canvas-corner color into an alpha
channel so the closet ends up with proper transparent cutouts.

We borrow two patterns from the reference VTON repo
(github.com/Akarsh-2004/fitted_prototype @ feature/vton-pipeline-refine):

* Hash-based caching of generated PNGs (skip the API call when an identical
  prompt/image was processed before).
* A quality-quality "looks like a garment" guardrail that rejects responses
  with empty / huge / bordering masks before they pollute the closet.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from pipeline.config import settings

GEMINI_HOST = "https://generativelanguage.googleapis.com"
# Models that support multimodal image generation + editing via :generateContent.
# Order matters: first that succeeds wins. ``gemini-2.5-flash-image`` is the
# stable production model; the preview models are newer alternatives that may
# return better quality if available on the key.
IMAGE_GEN_MODELS: Tuple[str, ...] = (
    "gemini-2.5-flash-image",
    "gemini-3.1-flash-image-preview",
    "gemini-3-pro-image-preview",
)
# Imagen :predict fallback for pure text-to-image generation (no image editing).
# We use the "fast" variant to keep seed runs snappy.
IMAGEN_MODEL_FALLBACK = "imagen-4.0-fast-generate-001"
HTTP_TIMEOUT = 60

# Pure-white pixels (R,G,B >= 244 and channel spread <= 12) are treated as
# background by the key-out step. Tuned for catalog-style backgrounds; works
# for both Imagen and Gemini-multimodal outputs.
WHITE_THRESHOLD = 244
WHITE_SPREAD = 12

# Minimum / maximum opaque area ratio for a result to be considered a valid
# garment cutout (between 4% and 88% of the canvas).
MIN_AREA_RATIO = 0.04
MAX_AREA_RATIO = 0.88


@dataclass
class GeminiImage:
    """Container for a successful Gemini image response."""

    png_bytes: bytes
    source_model: str
    cached: bool = False


class GeminiImageError(Exception):
    """Raised when every Gemini image-gen model attempt fails."""


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_dir() -> Path:
    target = settings.storage_dir / "gemini_cache"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _hash_payload(*parts: bytes) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
        h.update(b"\x00")
    return h.hexdigest()[:24]


# ---------------------------------------------------------------------------
# Background keying (white -> transparent)
# ---------------------------------------------------------------------------


def key_out_background(png_bytes: bytes) -> bytes:
    """Convert a solid-white background PNG into an RGBA cutout.

    Strategy: any pixel whose RGB is near-white *and* whose channel spread is
    small (i.e. not a cream / off-white garment) gets alpha=0. We then run a
    small alpha feather to soften the cutout edges.
    """
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    arr = np.array(img)
    rgb = arr[:, :, :3].astype(np.int16)
    near_white = (rgb >= WHITE_THRESHOLD).all(axis=2)
    spread_ok = (rgb.max(axis=2) - rgb.min(axis=2)) <= WHITE_SPREAD
    bg_mask = near_white & spread_ok

    alpha = np.where(bg_mask, 0, 255).astype(np.uint8)

    # Feather: blur the alpha by a 1px Gaussian so the cutout edge isn't jagged.
    from PIL import ImageFilter

    alpha_img = Image.fromarray(alpha, mode="L").filter(ImageFilter.GaussianBlur(radius=0.8))
    arr[:, :, 3] = np.array(alpha_img)

    out = Image.fromarray(arr, mode="RGBA")
    buf = io.BytesIO()
    out.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def cutout_area_ratio(rgba_bytes: bytes) -> float:
    """Return the fraction of pixels with alpha > 32 (opaque body of cutout)."""
    img = Image.open(io.BytesIO(rgba_bytes)).convert("RGBA")
    alpha = np.array(img)[:, :, 3]
    return float((alpha > 32).mean())


def ensure_garment_cutout(rgba_bytes: bytes) -> None:
    """Mirrors the reference VTON repo's `_ensure_likely_garment` guardrail.

    Raises :class:`GeminiImageError` with an actionable message if the cutout
    is empty, near-empty, or covers almost the entire canvas.
    """
    ratio = cutout_area_ratio(rgba_bytes)
    if ratio < MIN_AREA_RATIO:
        raise GeminiImageError(
            f"Generated cutout is too small ({ratio * 100:.1f}% of canvas). "
            "Gemini likely returned an empty or off-frame image."
        )
    if ratio > MAX_AREA_RATIO:
        raise GeminiImageError(
            f"Generated cutout fills almost the entire canvas ({ratio * 100:.1f}%). "
            "Gemini did not isolate the garment from its background."
        )


# ---------------------------------------------------------------------------
# Gemini HTTP plumbing
# ---------------------------------------------------------------------------


def _api_key() -> str:
    key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise GeminiImageError("GEMINI_API_KEY is not configured in .env")
    return key


def _post(url: str, payload: Dict, timeout: int = HTTP_TIMEOUT) -> Dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
        raise GeminiImageError(f"Gemini HTTP {exc.code}: {detail[:300]}") from exc
    except urllib.error.URLError as exc:
        raise GeminiImageError(f"Gemini network error: {exc}") from exc


def _extract_inline_png(response: Dict) -> Optional[bytes]:
    """Walk a generateContent response and return the first inline PNG bytes."""
    candidates = response.get("candidates") or []
    for cand in candidates:
        content = cand.get("content") or {}
        for part in content.get("parts") or []:
            inline = part.get("inlineData") or part.get("inline_data")
            if not inline:
                continue
            mime = inline.get("mimeType") or inline.get("mime_type") or ""
            data = inline.get("data")
            if data and mime.startswith("image/"):
                return base64.b64decode(data)
    return None


def _gemini_multimodal_call(
    model: str,
    text_prompt: str,
    image_b64: Optional[str] = None,
    image_mime: str = "image/png",
) -> bytes:
    """Single attempt against a Gemini multimodal image-gen model."""
    url = f"{GEMINI_HOST}/v1beta/models/{model}:generateContent?key={_api_key()}"
    parts: List[Dict] = [{"text": text_prompt}]
    if image_b64:
        parts.append({"inlineData": {"mimeType": image_mime, "data": image_b64}})

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
    }
    resp = _post(url, payload)
    png = _extract_inline_png(resp)
    if png is None:
        # Surface any safety / quota / error message for easier debugging.
        snippet = json.dumps(resp)[:400]
        raise GeminiImageError(f"{model} returned no image data: {snippet}")
    return png


def _imagen_predict_call(prompt: str) -> bytes:
    """Imagen 4 :predict fallback (text-to-image only, billed tier)."""
    url = f"{GEMINI_HOST}/v1beta/models/{IMAGEN_MODEL_FALLBACK}:predict?key={_api_key()}"
    payload = {
        "instances": [{"prompt": prompt}],
        "parameters": {"sampleCount": 1, "aspectRatio": "1:1"},
    }
    resp = _post(url, payload)
    preds = resp.get("predictions") or []
    if not preds:
        raise GeminiImageError(f"Imagen returned no predictions: {json.dumps(resp)[:300]}")
    b64 = preds[0].get("bytesBase64Encoded") or preds[0].get("bytes_base64_encoded")
    if not b64:
        raise GeminiImageError(f"Imagen response missing bytesBase64Encoded: {json.dumps(preds[0])[:300]}")
    return base64.b64decode(b64)


# ---------------------------------------------------------------------------
# Public generation helpers
# ---------------------------------------------------------------------------


def _try_models(text_prompt: str, image_b64: Optional[str], image_mime: str) -> GeminiImage:
    errors: List[str] = []
    for model in IMAGE_GEN_MODELS:
        try:
            png = _gemini_multimodal_call(model, text_prompt, image_b64, image_mime)
            return GeminiImage(png_bytes=png, source_model=model)
        except GeminiImageError as exc:
            errors.append(f"{model}: {exc}")
            continue

    # Imagen fallback only supports pure text-to-image.
    if image_b64 is None:
        try:
            png = _imagen_predict_call(text_prompt)
            return GeminiImage(png_bytes=png, source_model=IMAGEN_MODEL_FALLBACK)
        except GeminiImageError as exc:
            errors.append(f"{IMAGEN_MODEL_FALLBACK}: {exc}")

    raise GeminiImageError(
        "All Gemini image-gen models failed. Last errors: " + " | ".join(errors[-3:])
    )


def generate_apparel(prompt: str, *, cache_key_suffix: str = "") -> GeminiImage:
    """Text-to-image generation of a single apparel cutout on white BG."""
    cache_id = _hash_payload(prompt.encode("utf-8"), cache_key_suffix.encode("utf-8"))
    cache_path = _cache_dir() / f"gen_{cache_id}.png"
    if cache_path.exists():
        return GeminiImage(png_bytes=cache_path.read_bytes(), source_model="cache", cached=True)

    image = _try_models(prompt, image_b64=None, image_mime="image/png")
    cache_path.write_bytes(image.png_bytes)
    return image


def extract_cutout(image_bytes: bytes, *, category_hint: Optional[str] = None) -> GeminiImage:
    """Image-to-image extraction of a garment cutout from an uploaded photo."""
    cache_id = _hash_payload(image_bytes, (category_hint or "").encode("utf-8"))
    cache_path = _cache_dir() / f"cut_{cache_id}.png"
    if cache_path.exists():
        return GeminiImage(png_bytes=cache_path.read_bytes(), source_model="cache", cached=True)

    hint = (
        f"Focus on the {category_hint} item if multiple garments are visible. "
        if category_hint
        else "Focus on the most prominent garment if multiple items are visible. "
    )

    hint_l = (category_hint or "").lower()
    category_rule = (
        "Extract exactly ONE clothing item. "
        "Do not include body parts, face, hair, mannequin, or other garments. "
    )
    if any(term in hint_l for term in ("upper", "top", "shirt", "hoodie", "sweater", "jacket")):
        category_rule = (
            "Extract ONLY the upper-body garment. Include the garment sleeves, "
            "collar, hood, and hem if present, but remove the human body, arms, "
            "hands, head, pants, and shoes. Do not redraw a full outfit. "
        )
    elif any(term in hint_l for term in ("lower", "bottom", "pants", "jeans", "trousers", "shorts", "skirt")):
        category_rule = (
            "Extract ONLY the lower-body garment from waistband to cuffs/hem. "
            "Remove the torso, shirt, skin, feet, and shoes. The pants should be "
            "front-facing and centered as a product cutout. "
        )
    elif any(term in hint_l for term in ("shoe", "sneaker", "footwear", "boots")):
        category_rule = (
            "Extract ONLY the pair of shoes. Remove legs, socks, pants, body, "
            "floor, and background. Output the shoes as a centered ecommerce "
            "product pair, not as part of a full outfit. "
        )
    elif any(term in hint_l for term in ("hat", "headwear", "cap", "beanie")):
        category_rule = (
            "Extract ONLY the headwear item. Remove head, hair, face, neck, body, "
            "and every other garment. "
        )

    prompt = (
        "You are an ecommerce fashion product photographer. "
        f"{hint}"
        f"{category_rule}"
        "Redraw the garment from the supplied photo as a clean, centered apparel "
        "cutout on a PURE white #FFFFFF background. The garment must be "
        "front-facing, symmetrical, well-lit with soft studio lighting, no "
        "mannequin, no human body, no shadows on the background, no text, no "
        "watermarks. Preserve the garment's original color, texture, fabric "
        "folds and silhouette as faithfully as possible. Output a single 1024x1024 image."
    )
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    # Most browser uploads are JPEG; Gemini accepts a wide range, default to jpeg.
    mime = _sniff_mime(image_bytes)
    image = _try_models(prompt, image_b64=image_b64, image_mime=mime)
    cache_path.write_bytes(image.png_bytes)
    return image


def _sniff_mime(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def generate_apparel_rgba(prompt: str, *, cache_key_suffix: str = "") -> Tuple[bytes, GeminiImage]:
    """Generate an apparel image and return ``(rgba_png_bytes, original)``."""
    original = generate_apparel(prompt, cache_key_suffix=cache_key_suffix)
    rgba = key_out_background(original.png_bytes)
    ensure_garment_cutout(rgba)
    return rgba, original


def extract_cutout_rgba(image_bytes: bytes, *, category_hint: Optional[str] = None) -> Tuple[bytes, GeminiImage]:
    """Extract a garment cutout from an uploaded image and return RGBA bytes."""
    original = extract_cutout(image_bytes, category_hint=category_hint)
    rgba = key_out_background(original.png_bytes)
    ensure_garment_cutout(rgba)
    return rgba, original


def quick_health_check() -> Dict[str, object]:
    """Cheap sanity-check used by the seed/upload routes before running long jobs."""
    summary: Dict[str, object] = {"models_tried": [], "ok": False, "elapsed_ms": 0}
    start = time.perf_counter()
    try:
        for model in IMAGE_GEN_MODELS:
            summary["models_tried"].append(model)  # type: ignore[index]
            url = f"{GEMINI_HOST}/v1beta/models/{model}?key={_api_key()}"
            try:
                with urllib.request.urlopen(url, timeout=10) as resp:
                    if resp.status == 200:
                        summary["ok"] = True
                        summary["model"] = model
                        break
            except Exception:
                continue
    finally:
        summary["elapsed_ms"] = int((time.perf_counter() - start) * 1000)
    return summary
