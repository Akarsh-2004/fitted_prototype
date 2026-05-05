from __future__ import annotations

import base64
import hashlib
import json
from typing import Any

import cv2
import numpy as np


def image_hash(image: np.ndarray, params: dict[str, Any]) -> str:
  payload = image.tobytes() + json.dumps(params, sort_keys=True).encode("utf-8")
  return hashlib.sha256(payload).hexdigest()[:24]


def encode_png_base64(image: np.ndarray) -> str:
  ok, buf = cv2.imencode(".png", image)
  if not ok:
    raise RuntimeError("Failed to encode image")
  return base64.b64encode(buf.tobytes()).decode("utf-8")
