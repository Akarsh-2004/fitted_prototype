import cv2
import numpy as np
import logging
from typing import Dict, List, Any, Tuple

logger = logging.getLogger("VestirAI.HumanClothingParser")

class HumanClothingParser:
    """
    Drop-in replacement for Human Clothing Ingestion Parser.
    Interfaces with the new production-grade SCHP + SAM2 semantic human parsing architecture.
    Includes a robust geometry-based fallback system in case of offline execution,
    missing weights, or model inference failures.
    """
    
    def parse_clothing_layers(self, img: np.ndarray, person_mask: np.ndarray, person_box: List[float]) -> List[Dict[str, Any]]:
        """
        Partitions the human mask into semantic clothing layers.
        Attempts real semantic parsing first, falling back to anatomical guidelines on any failure.
        """
        try:
            logger.info("🔮 [Vestir AI] Attempting Real Semantic Human Parsing (SCHP + SAM2)...")
            from pipeline.parsing.garment_mask_builder import garment_mask_builder
            
            # Run the hybrid semantic human parsing pipeline
            layers = garment_mask_builder.parse_garments(img, person_mask, person_box)
            
            # Check for failure conditions (empty masks, corrupted parsing)
            if not layers:
                logger.warning("⚠️ [Vestir AI] Semantic parser returned empty layers. Activating anatomical fallback...")
                return self._parse_clothing_layers_geometric_fallback(img, person_mask, person_box)
                
            logger.info(f"✅ [Vestir AI] Semantic parser successfully extracted {len(layers)} clothing layers.")
            return layers
            
        except Exception as e:
            logger.error(f"❌ [Vestir AI] Semantic human parsing failed: {e}. Activating high-availability anatomical fallback...")
            return self._parse_clothing_layers_geometric_fallback(img, person_mask, person_box)

    def _parse_clothing_layers_geometric_fallback(
        self, 
        img: np.ndarray, 
        person_mask: np.ndarray, 
        person_box: List[float],
        granular: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Original geometry-based vertical parser.
        Offers high-speed, zero-dependency offline clothing layering fallback.
        Supports granular midpoint splits for left/right upper-body arms and shoes if granular=True.
        """
        h, w = img.shape[:2]
        px1, py1, px2, py2 = [int(v) for v in person_box]
        
        ph = py2 - py1
        pw = px2 - px1
        
        layers = []
        
        if ph <= 50 or pw <= 50:
            return layers
            
        ar = ph / pw
        
        # Adjust vertical dividers based on crop aspect ratio (awareness of what parts are in the image)
        if ar >= 2.35:
            # Full body crop
            head_end = py1 + int(ph * 0.15)       # Head and neck (0% - 15%)
            waist_end = py1 + int(ph * 0.55)      # Torso/Upper Body (15% - 55%)
            legs_end = py1 + int(ph * 0.94)       # Legs/Lower Body (55% - 94%)
            has_shoes = True
            has_bottoms = True
        elif ar >= 1.3:
            # Mid-shot / Thigh-up crop (No shoes)
            head_end = py1 + int(ph * 0.18)       # Head and neck (0% - 18%)
            waist_end = py1 + int(ph * 0.70)      # Torso/Upper Body (18% - 70%)
            legs_end = py2                        # Legs/Lower Body (70% - 100%)
            has_shoes = False
            has_bottoms = True
        else:
            # Upper-body / Waist-up crop (No bottoms, no shoes)
            head_end = py1 + int(ph * 0.22)       # Head and neck (0% - 22%)
            waist_end = py2                       # Torso/Upper Body (22% - 100%)
            legs_end = py2
            has_shoes = False
            has_bottoms = False
        
        # x-midpoint for arm and shoe splits
        mx = px1 + pw // 2

        # --- 1. Upper Body Garment Extraction ---
        upper_mask = person_mask.copy()
        upper_mask[0:head_end, :] = 0
        upper_mask[waist_end:h, :] = 0
        
        # --- 2. Lower Body Garment Extraction ---
        lower_mask = person_mask.copy()
        if has_bottoms:
            lower_mask[0:waist_end, :] = 0
            lower_mask[legs_end:h, :] = 0
        else:
            lower_mask[:, :] = 0
        
        # --- 3. Shoes Extraction ---
        shoes_mask = person_mask.copy()
        if has_shoes:
            shoes_mask[0:legs_end, :] = 0
        else:
            shoes_mask[:, :] = 0
        
        if granular:
            # Map to extended granular schema
            # 1. top_garment
            if np.sum(upper_mask) > 1000:
                contours, _ = cv2.findContours(upper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(cnt) > 200:
                        ux, uy, uw, uh = cv2.boundingRect(cnt)
                        epsilon = 0.003 * cv2.arcLength(cnt, True)
                        approx = cv2.approxPolyDP(cnt, epsilon, True)
                        poly = approx.reshape((-1, 2)).astype(float).tolist()
                        layers.append({
                            "layer_type": "top_garment",
                            "box": [float(ux), float(uy), float(ux + uw), float(uy + uh)],
                            "polygon": poly,
                            "mask": upper_mask
                        })
            
            # 2. bottom_garment
            if np.sum(lower_mask) > 1000:
                contours, _ = cv2.findContours(lower_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(cnt) > 200:
                        lx, ly, lw, lh = cv2.boundingRect(cnt)
                        epsilon = 0.003 * cv2.arcLength(cnt, True)
                        approx = cv2.approxPolyDP(cnt, epsilon, True)
                        poly = approx.reshape((-1, 2)).astype(float).tolist()
                        layers.append({
                            "layer_type": "bottom_garment",
                            "box": [float(lx), float(ly), float(lx + lw), float(ly + lh)],
                            "polygon": poly,
                            "mask": lower_mask
                        })

            # 3. left_arm (left half of upper mask)
            left_arm_mask = upper_mask.copy()
            left_arm_mask[:, mx:w] = 0
            if np.sum(left_arm_mask) > 300:
                contours, _ = cv2.findContours(left_arm_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(cnt) > 100:
                        lax, lay, law, lah = cv2.boundingRect(cnt)
                        epsilon = 0.003 * cv2.arcLength(cnt, True)
                        approx = cv2.approxPolyDP(cnt, epsilon, True)
                        poly = approx.reshape((-1, 2)).astype(float).tolist()
                        layers.append({
                            "layer_type": "left_arm",
                            "box": [float(lax), float(lay), float(lax + law), float(lay + lah)],
                            "polygon": poly,
                            "mask": left_arm_mask
                        })

            # 4. right_arm (right half of upper mask)
            right_arm_mask = upper_mask.copy()
            right_arm_mask[:, 0:mx] = 0
            if np.sum(right_arm_mask) > 300:
                contours, _ = cv2.findContours(right_arm_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(cnt) > 100:
                        rax, ray, raw, rah = cv2.boundingRect(cnt)
                        epsilon = 0.003 * cv2.arcLength(cnt, True)
                        approx = cv2.approxPolyDP(cnt, epsilon, True)
                        poly = approx.reshape((-1, 2)).astype(float).tolist()
                        layers.append({
                            "layer_type": "right_arm",
                            "box": [float(rax), float(ray), float(rax + raw), float(ray + rah)],
                            "polygon": poly,
                            "mask": right_arm_mask
                        })

            # 5. left_shoe (left half of shoes mask)
            left_shoe_mask = shoes_mask.copy()
            left_shoe_mask[:, mx:w] = 0
            if np.sum(left_shoe_mask) > 150:
                contours, _ = cv2.findContours(left_shoe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(cnt) > 50:
                        lsx, lsy, lsw, lsh = cv2.boundingRect(cnt)
                        epsilon = 0.003 * cv2.arcLength(cnt, True)
                        approx = cv2.approxPolyDP(cnt, epsilon, True)
                        poly = approx.reshape((-1, 2)).astype(float).tolist()
                        layers.append({
                            "layer_type": "left_shoe",
                            "box": [float(lsx), float(lsy), float(lsx + lsw), float(lsy + lsh)],
                            "polygon": poly,
                            "mask": left_shoe_mask
                        })

            # 6. right_shoe (right half of shoes mask)
            right_shoe_mask = shoes_mask.copy()
            right_shoe_mask[:, 0:mx] = 0
            if np.sum(right_shoe_mask) > 150:
                contours, _ = cv2.findContours(right_shoe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(cnt) > 50:
                        rsx, rsy, rsw, rsh = cv2.boundingRect(cnt)
                        epsilon = 0.003 * cv2.arcLength(cnt, True)
                        approx = cv2.approxPolyDP(cnt, epsilon, True)
                        poly = approx.reshape((-1, 2)).astype(float).tolist()
                        layers.append({
                            "layer_type": "right_shoe",
                            "box": [float(rsx), float(rsy), float(rsx + rsw), float(rsy + rsh)],
                            "polygon": poly,
                            "mask": right_shoe_mask
                        })
        else:
            # Standard vertical parser output
            if np.sum(upper_mask) > 1000:
                contours, _ = cv2.findContours(upper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    # Find largest contour inside the band
                    cnt = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(cnt) > 200:
                        ux, uy, uw, uh = cv2.boundingRect(cnt)
                        epsilon = 0.003 * cv2.arcLength(cnt, True)
                        approx = cv2.approxPolyDP(cnt, epsilon, True)
                        poly = approx.reshape((-1, 2)).astype(float).tolist()
                        
                        layers.append({
                            "layer_type": "upper",
                            "box": [float(ux), float(uy), float(ux + uw), float(uy + uh)],
                            "polygon": poly,
                            "mask": upper_mask
                        })
                        
            # --- 2. Lower Body Garment Extraction ---
            if np.sum(lower_mask) > 1000:
                contours, _ = cv2.findContours(lower_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(cnt) > 200:
                        lx, ly, lw, lh = cv2.boundingRect(cnt)
                        epsilon = 0.003 * cv2.arcLength(cnt, True)
                        approx = cv2.approxPolyDP(cnt, epsilon, True)
                        poly = approx.reshape((-1, 2)).astype(float).tolist()
                        
                        layers.append({
                            "layer_type": "lower",
                            "box": [float(lx), float(ly), float(lx + lw), float(ly + lh)],
                            "polygon": poly,
                            "mask": lower_mask
                        })
                        
            # --- 3. Shoes Extraction ---
            if np.sum(shoes_mask) > 300:
                contours, _ = cv2.findContours(shoes_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(cnt) > 100:
                        sx, sy, sw, sh = cv2.boundingRect(cnt)
                        epsilon = 0.003 * cv2.arcLength(cnt, True)
                        approx = cv2.approxPolyDP(cnt, epsilon, True)
                        poly = approx.reshape((-1, 2)).astype(float).tolist()
                        
                        layers.append({
                            "layer_type": "shoes",
                            "box": [float(sx), float(sy), float(sx + sw), float(sy + sh)],
                            "polygon": poly,
                            "mask": shoes_mask
                        })
                        
        return layers


human_clothing_parser = HumanClothingParser()
