import json
import urllib.request
from typing import Dict, Any, List, Optional
from pipeline.config import settings

class GeminiClient:
    def __init__(self):
        self.api_key = settings.gemini_api_key

    def analyze_garment(self, tags: List[str], dominant_colors: List[Dict[str, Any]], aspect_ratio: float, layer_hint: str = None) -> Dict[str, Any]:
        """
        Runs garment structural and semantic analysis.
        Uses Gemini Flash API if API key is present, otherwise falls back to the
        local Smart Fashion Intelligence Rule Engine.
        """
        if self.api_key:
            try:
                return self._call_gemini_api(tags, dominant_colors, aspect_ratio, layer_hint)
            except Exception as e:
                print(f"Gemini API execution failed: {e}. Falling back to offline rule engine...")
                
        return self._generate_offline_fashion_meta(tags, dominant_colors, aspect_ratio, layer_hint)

    def _call_gemini_api(self, tags: List[str], dominant_colors: List[Dict[str, Any]], aspect_ratio: float, layer_hint: str = None) -> Dict[str, Any]:
        """Direct, zero-dependency REST call to the Gemini v1beta API."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"
        
        # Build prompt
        colors_summary = ", ".join([f"{c['rgb']} with weight {c['weight']:.2f}" for c in dominant_colors])
        layer_hint_text = f"\n        - Layering hint / Anatomical location: {layer_hint} (Important: this indicates whether the crop is physically on the upper-body top/outerwear, lower-body bottom, or shoes)." if layer_hint else ""
        
        prompt = f"""
        You are a highly precise fashion AI expert.
        Analyze this garment with the following computed visual parameters:
        - Image Aspect Ratio (Height/Width): {aspect_ratio:.2f}{layer_hint_text}
        - Visual computer vision tags: {tags}
        - Dominant colors: {colors_summary}

        You must return a strictly valid JSON object matching this schema EXACTLY:
        {{
            "garment_type": "outerwear" | "top" | "bottom" | "shoes" | "dress" | "accessory",
            "fit": "slim" | "oversized" | "regular" | "relaxed",
            "material": "cotton" | "wool" | "denim" | "leather" | "knit" | "nylon" | "linen",
            "construction": "knit" | "woven" | "canvas" | "jersey",
            "pattern": "solid" | "striped" | "checkered" | "graphic" | "textured",
            "subtype": "jacket" | "hoodie" | "t-shirt" | "sweater" | "trousers" | "jeans" | "sneakers" | "boots" | "cap" | "tote-bag",
            "brand": "Uniqlo" | "Carhartt WIP" | "Our Legacy" | "Stussy" | "Arc'teryx" | "Acne Studios" | "Nike",
            "style": "casual" | "minimalist" | "gorpcore" | "streetwear" | "smart-casual" | "workwear",
            "occasion": "daily" | "formal" | "outdoor" | "nightout" | "sports",
            "season": "spring/autumn" | "winter" | "summer" | "all-season",
            "archetype": "minimalist" | "utilitarian" | "classic" | "skater" | "outdoor-enthusiast",
            "layering_role": "outer" | "mid" | "base" | "standalone",
            "pairing_suggestions": ["string of outfit pairing suggestion 1", "string of outfit pairing suggestion 2", "string of outfit pairing suggestion 3"]
        }}

        Do NOT add any markdown formatting, code blocks (like ```json), or conversational text. Return ONLY the JSON object.
        """
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json"
            }
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=8) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            text_response = res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
            return json.loads(text_response)

    def _generate_offline_fashion_meta(self, tags: List[str], dominant_colors: List[Dict[str, Any]], aspect_ratio: float, layer_hint: str = None) -> Dict[str, Any]:
        """
        Local Smart Fashion Intelligence Rule Engine.
        Synthesizes highly-detailed fashion taxonomies and styling parameters
        without external API dependencies.
        """
        # Determine main color name
        primary_color = tags[0] if tags else "grey"
        
        # Normalize layer_hint to supported offline taxonomy categories
        normalized_hint = None
        if layer_hint:
            lh_lower = layer_hint.lower()
            if lh_lower in ["upper", "top_garment", "top"]:
                normalized_hint = "upper"
            elif lh_lower in ["outerwear"]:
                normalized_hint = "outerwear"
            elif lh_lower in ["lower", "bottom_garment", "bottom"]:
                normalized_hint = "lower"
            elif lh_lower in ["shoes", "left_shoe", "right_shoe", "footwear"]:
                normalized_hint = "shoes"
            elif lh_lower == "hat":
                normalized_hint = "hat"
            elif lh_lower == "bag":
                normalized_hint = "bag"

        # Rule 1: Use normalized_hint if provided to bypass aspect ratio errors
        if normalized_hint == "outerwear":
            garment_type = "outerwear"
            subtype = "jacket"
            material = "nylon" if primary_color in ["black", "blue"] else "wool"
            construction = "woven"
            pattern = "solid"
            fit = "oversized"
            brand = "Arc'teryx" if material == "nylon" else "Acne Studios"
            style = "gorpcore" if material == "nylon" else "minimalist"
            occasion = "outdoor" if style == "gorpcore" else "daily"
            season = "winter"
            archetype = "outdoor-enthusiast" if style == "gorpcore" else "minimalist"
            layering_role = "outer"
            pairings = [
                "Relaxed-fit raw denim jeans and chunky black boots.",
                "A beige waffle-knit long sleeve shirt underneath.",
                "A lightweight nylon cross-body bag."
            ]
        elif normalized_hint == "upper":
            garment_type = "top"
            subtype = "sweater" if "relaxed-fit" in tags or "knit" in tags else "t-shirt"
            material = "knit" if subtype == "sweater" else "cotton"
            construction = "knit" if subtype == "sweater" else "jersey"
            pattern = "solid"
            fit = "relaxed" if "relaxed-fit" in tags else "regular"
            brand = "Uniqlo" if fit == "regular" else "Stussy"
            style = "minimalist" if fit == "regular" else "streetwear"
            occasion = "daily"
            season = "spring/autumn" if subtype == "sweater" else "summer"
            archetype = "minimalist" if style == "minimalist" else "classic"
            layering_role = "mid" if subtype == "sweater" else "base"
            pairings = [
                "Tailored dress trousers and leather chelsea boots.",
                "Layered over an oxford collared white shirt.",
                "An oversized wool trench coat in cold weather."
            ]
        elif normalized_hint == "lower":
            garment_type = "bottom"
            subtype = "jeans" if "blue" in tags or "denim" in tags else "trousers"
            material = "denim" if "blue" in tags or "denim" in tags else "cotton"
            construction = "woven"
            pattern = "solid"
            fit = "relaxed" if "relaxed-fit" in tags else "regular"
            brand = "Our Legacy" if material == "denim" else "Carhartt WIP"
            style = "streetwear" if fit == "relaxed" else "minimalist"
            occasion = "daily"
            season = "all-season"
            archetype = "skater" if fit == "relaxed" else "minimalist"
            layering_role = "standalone"
            pairings = [
                f"A white cropped t-shirt and clean {primary_color} sneakers.",
                "An oversized black zip-up hoodie for a comfortable relaxed silhouette.",
                "A vintage knitted knit sweater to contrast textures."
            ]
        elif normalized_hint == "shoes":
            garment_type = "shoes"
            subtype = "sneakers" if "sportswear" in tags or "relaxed" in tags else "boots"
            material = "leather"
            construction = "canvas" if subtype == "sneakers" else "woven"
            pattern = "solid"
            fit = "regular"
            brand = "Nike" if subtype == "sneakers" else "Our Legacy"
            style = "streetwear" if subtype == "sneakers" else "minimalist"
            occasion = "sports" if subtype == "sneakers" else "nightout"
            season = "all-season"
            archetype = "classic"
            layering_role = "standalone"
            pairings = [
                "Baggy utility carpenter pants and an oversized white tee.",
                "Classic ankle-cropped chinos and an active windbreaker.",
                "A clean grey melange sweat suit."
            ]
        elif normalized_hint == "hat":
            garment_type = "accessory"
            subtype = "cap"
            material = "cotton"
            construction = "woven"
            pattern = "solid"
            fit = "regular"
            brand = "Stussy"
            style = "streetwear"
            occasion = "daily"
            season = "summer"
            archetype = "skater"
            layering_role = "standalone"
            pairings = [
                "A matching casual t-shirt and light shorts.",
                "An oversized hoodie and high-top sneakers."
            ]
        elif normalized_hint == "bag":
            garment_type = "accessory"
            subtype = "tote-bag"
            material = "cotton"
            construction = "canvas"
            pattern = "solid"
            fit = "regular"
            brand = "Our Legacy"
            style = "minimalist"
            occasion = "daily"
            season = "all-season"
            archetype = "minimalist"
            layering_role = "standalone"
            pairings = [
                "A plain white knit sweater and dark blue jeans.",
                "Chunky neutral loafers and a minimalist trench coat."
            ]
        else:
            # Fallback to standard aspect ratio-based classification for flat lays
            if aspect_ratio > 1.4:
                # Bottoms / Pants
                garment_type = "bottom"
                subtype = "jeans" if "blue" in tags else "trousers"
                material = "denim" if "blue" in tags else "cotton"
                construction = "woven"
                pattern = "solid"
                fit = "relaxed" if "relaxed-fit" in tags else "regular"
                brand = "Our Legacy" if "blue" in tags else "Carhartt WIP"
                style = "streetwear" if "relaxed" in tags else "minimalist"
                occasion = "daily"
                season = "all-season"
                archetype = "skater" if "relaxed" in tags else "minimalist"
                layering_role = "standalone"
                pairings = [
                    f"A white cropped t-shirt and clean {primary_color} sneakers.",
                    "An oversized black zip-up hoodie for a comfortable relaxed silhouette.",
                    "A vintage knitted knit sweater to contrast textures."
                ]
            elif aspect_ratio > 0.95:
                # Tops / Jackets / Hoodies
                if "textured" in tags:
                    garment_type = "outerwear"
                    subtype = "jacket"
                    material = "wool" if primary_color in ["brown", "grey"] else "nylon"
                    construction = "woven"
                    pattern = "textured"
                    fit = "oversized"
                    brand = "Arc'teryx" if material == "nylon" else "Acne Studios"
                    style = "gorpcore" if material == "nylon" else "minimalist"
                    occasion = "outdoor" if style == "gorpcore" else "daily"
                    season = "winter"
                    archetype = "outdoor-enthusiast" if style == "gorpcore" else "minimalist"
                    layering_role = "outer"
                    pairings = [
                        "Relaxed-fit raw denim jeans and chunky black boots.",
                        "A beige waffle-knit long sleeve shirt underneath.",
                        "A lightweight nylon cross-body bag."
                    ]
                else:
                    garment_type = "top"
                    subtype = "hoodie" if "relaxed-fit" in tags else "t-shirt"
                    material = "cotton"
                    construction = "jersey"
                    pattern = "solid"
                    fit = "relaxed" if "relaxed-fit" in tags else "regular"
                    brand = "Stussy" if fit == "relaxed" else "Uniqlo"
                    style = "streetwear" if fit == "relaxed" else "casual"
                    occasion = "daily"
                    season = "spring/autumn" if subtype == "hoodie" else "summer"
                    archetype = "classic"
                    layering_role = "mid" if subtype == "hoodie" else "base"
                    pairings = [
                        "Wide-leg pleated trousers and clean leather loafers.",
                        "An unbuttoned flannel shirt over top.",
                        "Dark wash utilitarian cargo pants."
                    ]
            elif aspect_ratio < 0.6:
                # Accessories / Shoes
                garment_type = "shoes"
                subtype = "sneakers"
                material = "leather"
                construction = "canvas"
                pattern = "solid"
                fit = "regular"
                brand = "Nike"
                style = "casual"
                occasion = "daily"
                season = "all-season"
                archetype = "classic"
                layering_role = "standalone"
                pairings = [
                    "Baggy utility carpenter pants and an oversized white tee.",
                    "Classic ankle-cropped chinos and an active windbreaker.",
                    "A clean grey melange sweat suit."
                ]
            else:
                # Fallback
                garment_type = "top"
                subtype = "sweater"
                material = "knit"
                construction = "knit"
                pattern = "solid"
                fit = "regular"
                brand = "Uniqlo"
                style = "smart-casual"
                occasion = "daily"
                season = "spring/autumn"
                archetype = "minimalist"
                layering_role = "mid"
                pairings = [
                    "Tailored dress trousers and leather chelsea boots.",
                    "Layered over an oxford collared white shirt.",
                    "An oversized wool trench coat in cold weather."
                ]

        return {
            "garment_type": garment_type,
            "fit": fit,
            "material": material,
            "construction": construction,
            "pattern": pattern,
            "subtype": subtype,
            "brand": brand,
            "style": style,
            "occasion": occasion,
            "season": season,
            "archetype": archetype,
            "layering_role": layering_role,
            "pairing_suggestions": pairings
        }

    def generate_perfect_outfit(self, items: List[Dict[str, Any]], style_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Uses Gemini flash API to coordinate a perfect, harmonious outfit from a list of
        available digital closet items and predicts optimal 2D try-on offsets for perfect fitting.

        ``style_prompt`` is an optional free-form aesthetic description (e.g.
        ``"cyberpunk streetwear"``) that biases the AI stylist's selection.
        """
        if not items:
            return {
                "style_name": "Default Minimalist",
                "explanation": "No wardrobe items are available in your digital closet yet. Upload some clothes to start auto-styling!",
                "active_items": {},
                "fitting": {}
            }

        if self.api_key:
            try:
                return self._call_gemini_styling_api(items, style_prompt=style_prompt)
            except Exception as e:
                print(f"Gemini auto-styling API failed: {e}. Falling back to offline rule matching...")

        return self._generate_offline_perfect_outfit(items, style_prompt=style_prompt)

    def _call_gemini_styling_api(self, items: List[Dict[str, Any]], style_prompt: Optional[str] = None) -> Dict[str, Any]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"
        
        # Summarize closet items for Gemini
        closet_summary = []
        for i in items:
            closet_summary.append({
                "id": i["id"],
                "garment_type": i.get("garment_type", "unknown"),
                "subtype": i.get("subtype", "unknown"),
                "brand": i.get("brand", "unknown"),
                "colors": i.get("colors", []),
                "tags": i.get("tags", []),
                "style": i.get("style", "casual"),
                "layering_role": i.get("layering_role", "base")
            })

        style_prompt_block = (
            f"\n        Aesthetic direction from the user (treat as a strong style constraint): {style_prompt!r}\n"
            if style_prompt
            else ""
        )

        prompt = f"""
        You are a world-class high-fashion digital stylist.{style_prompt_block}
        We have a flat 1024x1024 outfit composer (no mannequin, just stacked apparel cutouts) and a digital closet containing the following items:
        {json.dumps(closet_summary, indent=2)}

        Your tasks are:
        1. Select a highly harmonious, premium, and aesthetically cohesive outfit combination from the list.
           - Try to include a 'top', a 'bottom', and 'shoes'.
           - (Optional) Include 'outerwear' and an 'accessory' (especially a hat / cap) if they fit the theme perfectly.
           - You can select AT MOST one item per category (tops, bottoms, shoes, outerwear, accessories).
           - When a user aesthetic is provided above, prioritise items whose tags/style/material align with it.
        2. Predict the **optimal 2D try-on fitting parameters** (scale, x-offset, y-offset) for EACH chosen item ID so that they fit the front-facing mannequin silhouette perfectly.
           - We place clothes using relative overlay boxes. Predict offsets as numbers:
             - scale: percentage scale, standard is 100 (range: 80 to 130).
             - x: horizontal offset in pixels, standard is 0 (range: -30 to 30).
             - y: vertical offset in pixels, standard is 0 (range: -50 to 50).
             - Predict adjustments specifically to correct any visual gaps. For example, if a jacket should be styled oversized, set scale: 110, y: 5.
        3. Formulate a catchy Style Combination name (e.g. "Tokyo Workwear Utility") and a detailed explanation of why these pieces are harmonious (e.g. why their colors, materials, and silhouettes sync up beautifully).

        You must return a strictly valid JSON object matching this schema EXACTLY:
        {{
            "style_name": "String name of outfit style",
            "explanation": "Detailed professional styling advice and explanation",
            "active_items": {{
                "top": "selected_item_id_or_empty_string",
                "bottom": "selected_item_id_or_empty_string",
                "shoes": "selected_item_id_or_empty_string",
                "outerwear": "selected_item_id_or_empty_string",
                "accessory": "selected_item_id_or_empty_string"
            }},
            "fitting": {{
                "selected_item_id_1": {{ "scale": 100, "x": 0, "y": 0 }},
                "selected_item_id_2": {{ "scale": 95, "x": 5, "y": -10 }}
            }}
        }}

        Do NOT add any markdown formatting, code blocks (like ```json), or conversational text. Return ONLY the JSON object.
        """
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=12) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            text = res_data["contents"][0]["parts"][0]["text"]
            
            # Clean up potential markdown formatting block wrappers
            text_cleaned = text.strip()
            if text_cleaned.startswith("```"):
                lines = text_cleaned.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].strip() == "```":
                    lines = lines[:-1]
                text_cleaned = "\n".join(lines).strip()
                
            return json.loads(text_cleaned)

    def _generate_offline_perfect_outfit(self, items: List[Dict[str, Any]], style_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Local fallback rule engine to style a neat coordinate with standard fitting values.

        When a ``style_prompt`` is provided we bias the per-category pick toward
        items whose tags/style/material overlap with the prompt keywords.
        """
        prompt_terms: List[str] = []
        if style_prompt:
            prompt_terms = [t.strip().lower() for t in style_prompt.replace(",", " ").split() if t.strip()]

        def _score(item: Dict[str, Any]) -> int:
            if not prompt_terms:
                return 0
            haystack = " ".join(
                str(v).lower()
                for v in (
                    item.get("style"),
                    item.get("material"),
                    item.get("archetype"),
                    item.get("occasion"),
                    " ".join(item.get("tags") or []),
                )
                if v
            )
            return sum(1 for term in prompt_terms if term in haystack)

        def _best(pool: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            if not pool:
                return None
            return sorted(pool, key=_score, reverse=True)[0]

        tops = [i for i in items if i.get("garment_type", "").lower() == "top"]
        bottoms = [i for i in items if i.get("garment_type", "").lower() == "bottom"]
        shoes = [i for i in items if i.get("garment_type", "").lower() == "shoes"]
        outerwear = [i for i in items if i.get("garment_type", "").lower() == "outerwear"]
        accessories = [i for i in items if i.get("garment_type", "").lower() == "accessory"]

        active_items: Dict[str, str] = {}
        fitting: Dict[str, Dict[str, int]] = {}

        for label, pool in (("top", tops), ("bottom", bottoms), ("shoes", shoes), ("outerwear", outerwear), ("accessory", accessories)):
            pick = _best(pool)
            if pick is not None:
                active_items[label] = pick["id"]
                fitting[pick["id"]] = {"scale": 100, "x": 0, "y": 0}

        if style_prompt:
            style_name = f"{style_prompt.title()} Capsule"
            explanation = (
                f"Offline rule engine selected the closet items whose tags best match '{style_prompt}'. "
                "Connect a Gemini API key in .env to unlock the full AI stylist for richer aesthetic reasoning."
            )
        else:
            style_name = "Standard Studio Coordination (Offline Fallback)"
            explanation = (
                "We composed a sleek, balanced base profile using your first closet items. "
                "Connect your Gemini API Key in .env to unlock AI-assisted aesthetic recommendations and spatial try-on positioning!"
            )

        return {
            "style_name": style_name,
            "explanation": explanation,
            "active_items": active_items,
            "fitting": fitting,
        }

gemini_client = GeminiClient()
