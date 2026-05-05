from __future__ import annotations

from pydantic import BaseModel


class ProcessMeta(BaseModel):
    mask_quality: float
    processing_time: float
    used_fallback: bool
    warp_mode: str
    request_key: str


class ProcessResponse(BaseModel):
    cutout: str
    composite: str
    mask: str
    meta: ProcessMeta
