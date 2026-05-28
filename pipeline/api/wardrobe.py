from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
from pipeline.database.storage import get_all_wardrobe_items, get_wardrobe_item, delete_wardrobe_item
from pipeline.analysis.oklch_scorer import score_color_harmony
from pipeline.services.vector_store import vector_store

router = APIRouter(prefix="/api/wardrobe", tags=["Wardrobe"])

@router.get("")
def browse_wardrobe(
    category: Optional[str] = None,
    search: Optional[str] = None
):
    """
    Returns all ingested wardrobe items, optionally filtered by category
    or textual keyword search.

    The ``hat`` pseudo-category routes through the composer category resolver
    so accessories with hat/cap/beanie subtypes surface correctly.
    """
    items = get_all_wardrobe_items()
    
    # Apply category filter
    if category and category.lower() != "all":
        from pipeline.composer.alignment import resolve_composer_category

        cat = category.lower()
        if cat == "hat":
            items = [
                i for i in items
                if resolve_composer_category(
                    i.get("garment_type"), i.get("subtype")
                ) == "hat"
            ]
        else:
            items = [i for i in items if i.get("garment_type", "").lower() == cat]
        
    # Apply text keyword search
    if search:
        query = search.lower()
        filtered = []
        for i in items:
            tags_match = any(query in t.lower() for t in i.get("tags", []))
            brand_match = query in i.get("brand", "").lower() if i.get("brand") else False
            subtype_match = query in i.get("subtype", "").lower() if i.get("subtype") else False
            style_match = query in i.get("style", "").lower() if i.get("style") else False
            
            if tags_match or brand_match or subtype_match or style_match:
                filtered.append(i)
        items = filtered
        
    return {"items": items}

@router.delete("/{item_id}")
def delete_item(item_id: str):
    """Deletes an item from the database and vector storage."""
    item = get_wardrobe_item(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
        
    # Delete from database
    delete_wardrobe_item(item_id)
    
    # Delete from vector store
    vector_store.delete_item(item_id)
    
    # Delete cropped image file if possible
    try:
        from pipeline.config import settings
        full_path = settings.base_dir / item["image_path"]
        if full_path.exists():
            full_path.unlink()
    except Exception:
        pass
        
    return {"success": True, "message": "Item successfully deleted."}

@router.get("/{item_id}/harmony")
def calculate_item_harmony(item_id: str):
    """
    Compares the target item's primary OKLCH color profile against
    every other garment in the database, ranking them by color compatibility.
    """
    target = get_wardrobe_item(item_id)
    if not target:
        raise HTTPException(status_code=404, detail="Item not found")
        
    all_items = get_all_wardrobe_items()
    harmony_results = []
    
    for item in all_items:
        if item["id"] == item_id:
            continue
            
        score = score_color_harmony(target["colors"], item["colors"])
        
        harmony_results.append({
            "item": item,
            "harmony_score": score,
            "match_status": "highly_harmonious" if score >= 0.85 else ("compatible" if score >= 0.70 else "contrasting")
        })
        
    # Sort descending by harmony score
    harmony_results.sort(key=lambda x: x["harmony_score"], reverse=True)
    
    return {
        "target_item": target,
        "matches": harmony_results
    }

@router.get("/{item_id}/similar")
def find_visually_similar_items(item_id: str, limit: int = 6):
    """
    Uses vector search to find items that are visually similar in shape,
    texture, and dominant color weights.
    """
    target = get_wardrobe_item(item_id)
    if not target:
        raise HTTPException(status_code=404, detail="Item not found")
        
    # Get the stored visual vector embedding
    emb = vector_store.embeddings.get(item_id)
    if not emb:
        return {"target_item": target, "matches": []}
        
    # Search closest vectors
    search_results = vector_store.search(emb, top_k=limit + 1)
    
    matches = []
    for match_id, score in search_results:
        if match_id == item_id:
            continue
            
        match_item = get_wardrobe_item(match_id)
        if match_item:
            matches.append({
                "item": match_item,
                "similarity_score": float(round(score, 3))
            })
            
    return {
        "target_item": target,
        "matches": matches[:limit]
    }

@router.post("/auto-style")
def auto_style_mannequin():
    """
    Leverages the Gemini API to analyze all wardrobe items in the closet,
    coordinates a perfectly harmonious outfit, and predicts optimal 2D fitting coordinates.
    """
    from pipeline.analysis.gemini_client import gemini_client
    items = get_all_wardrobe_items()
    result = gemini_client.generate_perfect_outfit(items)
    return result
