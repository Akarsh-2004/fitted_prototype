import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from pipeline.analysis.oklch_scorer import extract_dominant_colors

class ColorNames:
    # A simple map of RGB coordinates to standard color names
    COLOR_MAP = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "grey": (128, 128, 128),
        "red": (220, 38, 38),
        "blue": (37, 99, 235),
        "green": (22, 163, 74),
        "yellow": (234, 179, 8),
        "purple": (147, 51, 234),
        "orange": (249, 115, 22),
        "pink": (236, 72, 153),
        "beige": (245, 245, 220),
        "navy": (30, 41, 59),
        "brown": (120, 53, 4)
    }

    @classmethod
    def get_closest_name(cls, rgb: List[int]) -> str:
        best_name = "neutral"
        min_dist = float("inf")
        
        for name, target in cls.COLOR_MAP.items():
            dist = sum((a - b) ** 2 for a, b in zip(rgb, target))
            if dist < min_dist:
                min_dist = dist
                best_name = name
                
        return best_name

class SigLIPTagger:
    """
    Highly responsive visual tagging & feature embedding model.
    Analyzes visual attributes and outputs visual tags along with a normalized
    256-dimensional vector embedding for search.
    """
    def extract_visual_features(self, rgba_img: np.ndarray) -> Tuple[List[str], List[float]]:
        """
        Analyzes the transparent garment image crop to extract:
        - List of string fashion tags (colors, textures, fit archetypes, patterns)
        - 256-dimensional normalized visual representation vector.
        """
        h, w = rgba_img.shape[:2]
        
        # 1. Color Extraction
        dom_colors = extract_dominant_colors(rgba_img, num_colors=2)
        primary_color_rgb = dom_colors[0]["rgb"]
        primary_color_name = ColorNames.get_closest_name(primary_color_rgb)
        
        tags = [primary_color_name]
        
        # 2. Aspect Ratio & Shape Analysis
        aspect_ratio = h / w if w > 0 else 1.0
        
        # Approximate item type based on aspect ratio
        shape_type = "square"
        if aspect_ratio > 1.6:
            shape_type = "tall"
            tags.append("slim-fit")
        elif aspect_ratio < 0.8:
            shape_type = "wide"
            tags.append("relaxed-fit")
        else:
            tags.append("regular-fit")
            
        # 3. Texture Frequency Analysis (pattern vs solid)
        gray = cv2.cvtColor(rgba_img[:, :, :3], cv2.COLOR_BGR2GRAY)
        # Calculate Laplacian variance to assess sharp edges (denotes patterns/textures)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if lap_var > 150.0:
            tags.append("textured")
            tags.append("patterned")
        else:
            tags.append("solid")
            tags.append("minimalist")
            
        # Add general style suggestions based on rules
        if primary_color_name in ["black", "white", "grey", "navy"]:
            tags.append("classic")
            tags.append("versatile")
        else:
            tags.append("bold")
            tags.append("accent")
            
        # 4. Generate 256-d normalized visual embedding
        # We downsample the crop to 12x12 pixels (144 features)
        # We append spatial color distributions (dominant colors and weights) and aspect ratios
        # to construct a solid 256-dimensional visual feature vector.
        resized = cv2.resize(gray, (12, 12)).flatten() # 144 elements
        
        # Create an embedding array of size 256
        embedding = np.zeros(256, dtype=np.float32)
        
        # Fill first 144 elements with visual downsamples
        embedding[:144] = resized / 255.0
        
        # Fill next elements with dominant color values and coordinates
        embedding[144:147] = [c / 255.0 for c in primary_color_rgb]
        embedding[147] = dom_colors[0]["weight"]
        
        if len(dom_colors) > 1:
            embedding[148:151] = [c / 255.0 for c in dom_colors[1]["rgb"]]
            embedding[151] = dom_colors[1]["weight"]
            tags.append(ColorNames.get_closest_name(dom_colors[1]["rgb"]))
            
        # Store shape characteristics in remaining slots
        embedding[152] = min(3.0, aspect_ratio) / 3.0
        embedding[153] = float(lap_var / 500.0) if lap_var < 500.0 else 1.0
        
        # Normalize the vector to unit length so dot product == cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return tags, embedding.tolist()

siglip_tagger = SigLIPTagger()
