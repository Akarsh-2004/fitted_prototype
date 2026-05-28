import cv2
import numpy as np
import math
from typing import List, Dict, Any, Tuple

def rgb_to_oklch(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """
    Converts sRGB [0..255] to OKLCH color space.
    - L: Lightness [0..1]
    - C: Chroma [0..0.4+]
    - H: Hue [0..360] degrees
    """
    # Normalize RGB to [0, 1]
    r, g, b = [v / 255.0 for v in rgb]
    
    # Convert sRGB to linear RGB
    def to_linear(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        
    r_l = to_linear(r)
    g_l = to_linear(g)
    b_l = to_linear(b)
    
    # Convert to LMS space
    l = 0.4122214708 * r_l + 0.5363325363 * g_l + 0.0514459929 * b_l
    m = 0.2119034982 * r_l + 0.6806995451 * g_l + 0.1073970000 * b_l
    s = 0.0883024619 * r_l + 0.2817188376 * g_l + 0.6302613616 * b_l
    
    # Non-linear LMS
    l_ = l ** (1.0/3.0) if l > 0 else 0
    m_ = m ** (1.0/3.0) if m > 0 else 0
    s_ = s ** (1.0/3.0) if s > 0 else 0
    
    # Convert to Oklab
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    
    # Convert Oklab to OKLCH
    C = math.sqrt(a * a + b * b)
    H = math.atan2(b, a) * (180.0 / math.pi)
    if H < 0:
        H += 360.0
        
    return float(L), float(C), float(H)

def extract_dominant_colors(rgba_img: np.ndarray, num_colors: int = 3) -> List[Dict[str, Any]]:
    """
    Uses OpenCV K-Means to cluster non-transparent pixels in an RGBA image.
    Returns a list of dicts: [{'rgb': [r,g,b], 'weight': float, 'oklch': [l,c,h]}]
    """
    # 1. Filter out background pixels (where alpha is 0 or low)
    alpha = rgba_img[:, :, 3]
    pixels_idx = np.where(alpha > 50)
    
    if len(pixels_idx[0]) < 100:
        # Fallback if too few pixels
        return [{"rgb": [128, 128, 128], "weight": 1.0, "oklch": rgb_to_oklch((128, 128, 128))}]
        
    # Get standard BGR channels for clustering
    pixels = rgba_img[pixels_idx][:, :3]
    
    # Convert to float32 for CV K-Means
    pixels_f = np.float32(pixels)
    
    # K-Means criteria and flags
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    
    try:
        # Run K-Means
        _, labels, centers = cv2.kmeans(pixels_f, num_colors, None, criteria, 10, flags)
        
        # Calculate weights of each cluster
        unique_labels, counts = np.unique(labels, return_counts=True)
        total_counts = np.sum(counts)
        
        dominant_colors = []
        for i, center in enumerate(centers):
            # Center values are in RGB/BGR. Let's convert to standard integer [0..255]
            # cv2 reads BGRA, our input rgba_img is RGBA. Let's make sure channel indices are correct.
            # In save_crop_rgba, we convert to BGRA. When we loaded it, let's assume standard RGB.
            r, g, b = [int(np.clip(v, 0, 255)) for v in center]
            weight = float(counts[i] / total_counts) if i < len(counts) else 0.0
            
            oklch = rgb_to_oklch((r, g, b))
            dominant_colors.append({
                "rgb": [r, g, b],
                "weight": weight,
                "oklch": oklch
            })
            
        # Sort by weight descending
        dominant_colors.sort(key=lambda x: x["weight"], reverse=True)
        return dominant_colors
    except Exception as e:
        print(f"K-Means color extraction failed: {e}. Returning average color.")
        # Fallback to simple average color
        avg_r = int(np.mean(pixels[:, 0]))
        avg_g = int(np.mean(pixels[:, 1]))
        avg_b = int(np.mean(pixels[:, 2]))
        oklch = rgb_to_oklch((avg_r, avg_g, avg_b))
        return [{"rgb": [avg_r, avg_g, avg_b], "weight": 1.0, "oklch": oklch}]

def score_color_harmony(colors1: List[Dict[str, Any]], colors2: List[Dict[str, Any]]) -> float:
    """
    Computes a color harmony compatibility score between two garments.
    Checks Hue relationship of the primary dominant color in OKLCH:
    - Complementary: Hue difference ~180°
    - Analogous: Hue difference ~30° - 60°
    - Monochromatic: Hue difference ~0° (similar colors)
    - Triadic: Hue difference ~120° or ~240°
    Returns a score between 0.0 (clashing) and 1.0 (highly harmonious).
    """
    if not colors1 or not colors2:
        return 0.5
        
    primary1 = colors1[0]["oklch"]
    primary2 = colors2[0]["oklch"]
    
    L1, C1, H1 = primary1
    L2, C2, H2 = primary2
    
    # If one of them is highly neutral (black, white, grey - low Chroma), it matches anything!
    if C1 < 0.04 or C2 < 0.04 or L1 < 0.15 or L1 > 0.85 or L2 < 0.15 or L2 > 0.85:
        # Neutrals are universally harmonious
        return 0.95
        
    # Calculate shortest angular distance on 360° circle
    hue_diff = abs(H1 - H2)
    if hue_diff > 180.0:
        hue_diff = 360.0 - hue_diff
        
    # Harmony target distances
    targets = [
        (0.0, 0.95, "Monochromatic"), # Identical hue
        (30.0, 0.90, "Analogous"),   # Neighbors
        (120.0, 0.85, "Triadic"),    # Split
        (180.0, 0.95, "Complementary") # Opposites
    ]
    
    best_score = 0.0
    for target_hue, target_score, name in targets:
        dist = abs(hue_diff - target_hue)
        # Score decreases as distance from the target increases
        # Using a Gaussian-like decay curve
        score = target_score * math.exp(-((dist / 15.0) ** 2))
        if score > best_score:
            best_score = score
            
    # Add a base similarity compatibility factor
    base_compatibility = max(0.4, best_score)
    return float(round(base_compatibility, 2))
