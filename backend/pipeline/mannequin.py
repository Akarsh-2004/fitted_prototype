from __future__ import annotations

import cv2
import numpy as np


def make_base_mannequin(height: int = 720, width: int = 540) -> np.ndarray:
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    center = (width // 2, int(height * 0.44))
    torso_w, torso_h = int(width * 0.36), int(height * 0.56)

    overlay = canvas.copy()
    cv2.ellipse(overlay, center, (torso_w // 2, torso_h // 2), 0, 0, 360, (228, 222, 214), -1)
    cv2.addWeighted(overlay, 0.86, canvas, 0.14, 0, canvas)

    shade = np.zeros_like(canvas)
    cv2.ellipse(shade, (center[0] + 18, center[1] + 10), (torso_w // 2 - 18, torso_h // 2 - 30), 0, 0, 360, (32, 32, 32), -1)
    shade = cv2.GaussianBlur(shade, (0, 0), 21)
    canvas = cv2.addWeighted(canvas, 1.0, shade, 0.08, 0)
    return canvas
