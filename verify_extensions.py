import sys
import os
import cv2
import numpy as np

# Ensure root folder is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.services.model_registry import model_registry
from pipeline.parsing.cutout_extractor import cutout_extractor
from pipeline.parsing.semantic_mapper import semantic_mapper
from pipeline.parsing.fine_parser import fine_parser
from pipeline.detectors.schp_parser import human_clothing_parser
from pipeline.detectors.sam2_segmenter import sam2_segmenter
from pipeline.config import settings

def test_model_registry():
    print("🧪 [1/5] Testing Model Registry Singleton & Status...")
    status = model_registry.get_status()
    assert "segformer" in status
    assert "accessories" in status
    assert status["segformer"] == "cold"
    assert status["accessories"] == "cold"
    print("  - Default cold statuses verified.")

def test_cutout_extractor_paths():
    print("\n🧪 [2/5] Testing Cutout Extractor Primary and Fallback paths...")
    # Create a synthetic human image: a grey circle centered in black background
    h, w = 200, 200
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.circle(img, (100, 100), 50, (128, 128, 128), -1)

    bbox_xyxy = [50.0, 50.0, 150.0, 150.0]
    # Simple polygon enclosing the circle
    polygon = [
        [100.0, 50.0],
        [150.0, 100.0],
        [100.0, 150.0],
        [50.0, 100.0]
    ]

    # Force SegFormer to fail to test GrabCut fallback path specifically
    original_get_segformer = model_registry.get_segformer
    model_registry.get_segformer = lambda: None

    job_id = "test_run_cutout"
    person_idx = 0
    
    try:
        res = cutout_extractor.extract_person_cutout(img, bbox_xyxy, polygon, job_id, person_idx)
        
        assert res["method_used"] == "grabcut", f"Should have used grabcut fallback, got: {res['method_used']}"
        assert res["bbox"] == [50, 50, 100, 100], f"Bounding box mismatch: {res['bbox']}"
        assert len(res["contour_polygon"]) == 12, f"Polygon should have exactly 12 points, got {len(res['contour_polygon'])}"
        
        # Verify files are saved
        crop_path = settings.base_dir / res["rgba_crop_path"]
        mask_path = settings.base_dir / res["mask_path"]
        
        assert crop_path.exists(), "RGBA transparent crop file not written to disk"
        assert mask_path.exists(), "Debug mask file not written to disk"
        
        # Verify alpha channel transparency is correct
        rgba_img = cv2.imread(str(crop_path), cv2.IMREAD_UNCHANGED)
        assert rgba_img.shape == (100, 100, 4), f"Output crop dimensions mismatch: {rgba_img.shape}"
        assert np.any(rgba_img[:, :, 3] > 0), "Alpha channel should have some foreground pixels"
        
        # Clean up files
        crop_path.unlink()
        mask_path.unlink()
        
        print("  - Cutout Extractor GrabCut fallback path passed successfully.")
    finally:
        # Restore registry
        model_registry.get_segformer = original_get_segformer

def test_semantic_mapper_extensions():
    print("\n🧪 [3/5] Testing Extended Semantic Mapper & Shoe pair union merges...")
    h, w = 100, 100
    
    # 1. Accessories, Shoes, and Bag mapping test
    mock_label_masks = {
        "hat": np.zeros((h, w), dtype=np.uint8),
        "glove": np.zeros((h, w), dtype=np.uint8),
        "socks": np.zeros((h, w), dtype=np.uint8),
        "left-shoe": np.zeros((h, w), dtype=np.uint8),
        "right-shoe": np.zeros((h, w), dtype=np.uint8),
        "left-leg": np.zeros((h, w), dtype=np.uint8),  # maps to bag (class 16 in GRANULAR_PARTS)
        "hair": np.zeros((h, w), dtype=np.uint8)       # maps to accessory (class 2 in GRANULAR_PARTS)
    }
    
    # Set significant pixel count (at least 600px > 500px area gate)
    mock_label_masks["hat"][10:30, 20:50] = 255          # 600px
    mock_label_masks["left-shoe"][80:95, 20:60] = 255    # 600px
    mock_label_masks["right-shoe"][80:95, 60:100] = 255  # 600px
    mock_label_masks["left-leg"][40:60, 10:40] = 255      # 600px (bag)
    
    mapped = semantic_mapper.map_to_granular_parts(mock_label_masks)
    
    assert "hat" in mapped, "Hat should be mapped"
    assert "left_shoe" in mapped, "Left shoe should be mapped"
    assert "right_shoe" in mapped, "Right shoe should be mapped"
    assert "bag" in mapped, "left-leg should map to bag"
    assert "footwear" in mapped, "Should merge left and right shoes into footwear union"
    
    # Check footwear union mask matches both
    assert np.all(mapped["footwear"][80:95, 20:60] == 255)
    assert np.all(mapped["footwear"][80:95, 60:100] == 255)
    
    # 2. Min area gate test
    mock_label_masks_tiny = {
        "hat": np.zeros((h, w), dtype=np.uint8)
    }
    mock_label_masks_tiny["hat"][10:15, 10:15] = 255 # 25px < 500px gate
    mapped_tiny = semantic_mapper.map_to_granular_parts(mock_label_masks_tiny)
    assert "hat" not in mapped_tiny, "Hat should be filtered out by 500px min area gate"
    
    # 3. Bag Independence test (bag is never suppressed by dress or outerwear)
    mock_label_masks_bag = {
        "dress": np.zeros((h, w), dtype=np.uint8),
        "left-leg": np.zeros((h, w), dtype=np.uint8) # bag
    }
    mock_label_masks_bag["dress"][10:80, 10:80] = 255 # large dress
    mock_label_masks_bag["left-leg"][20:60, 20:60] = 255 # overlapping bag
    
    mapped_bag = semantic_mapper.map_to_granular_parts(mock_label_masks_bag)
    assert "bag" in mapped_bag, "Bag mask should bypass suppression and be fully preserved"
    print("  - Remappings, shoe unions, bag independence, and min area filter gates passed successfully.")

def test_extended_anatomical_fallback():
    print("\n🧪 [4/5] Testing Extended Anatomical mid-point splits Fallback...")
    h, w = 200, 200
    person_mask = np.zeros((h, w), dtype=np.uint8)
    person_mask[20:180, 50:110] = 255
    person_box = [50.0, 20.0, 110.0, 180.0]
    
    # Run the fallback with granular=True
    layers = human_clothing_parser._parse_clothing_layers_geometric_fallback(
        img=np.zeros((h, w, 3), dtype=np.uint8),
        person_mask=person_mask,
        person_box=person_box,
        granular=True
    )
    
    types = [layer["layer_type"] for layer in layers]
    assert "top_garment" in types
    assert "bottom_garment" in types
    assert "left_arm" in types
    assert "right_arm" in types
    assert "left_shoe" in types
    assert "right_shoe" in types
    
    # Verify arms split at x midpoint mx = 50 + 60/2 = 80
    left_arm_layer = next(l for l in layers if l["layer_type"] == "left_arm")
    right_arm_layer = next(l for l in layers if l["layer_type"] == "right_arm")
    
    assert np.all(left_arm_layer["mask"][:, 80:] == 0), "Left arm mask should have nothing on the right of mx"
    assert np.all(right_arm_layer["mask"][:, :80] == 0), "Right arm mask should have nothing on the left of mx"
    print("  - Midpoint splitting for anatomical arm and shoe fallbacks passed successfully.")

def test_fine_parser_fallback_routing():
    print("\n🧪 [5/5] Testing Fine Parser Fallback Routing & Coordinates remapping...")
    h, w = 200, 200
    person_mask = np.zeros((h, w), dtype=np.uint8)
    person_mask[20:180, 50:150] = 255
    person_box = [50.0, 20.0, 150.0, 180.0]
    
    job_id = "test_fine_routing"
    
    import pipeline.parsing.fine_parser as fp
    
    # Mock CE2P inference to raise an exception, forcing fallback parser execution
    original_schp_inference = fp.run_schp_inference
    def mock_fail(x):
        raise ValueError("Force CE2P fallback in test")
    fp.run_schp_inference = mock_fail
    
    try:
        # Running fine parser will default to geometric fallback as CE2P is forced to fail
        res = fine_parser.parse_granular_clothing_layers(
            img_bgr=np.zeros((h, w, 3), dtype=np.uint8),
            person_mask=person_mask,
            person_box=person_box,
            job_id=job_id,
            person_index=0
        )
        
        assert res["parser_used"] == "geometric_fallback"
        assert len(res["parts"]) > 0
        
        # Check that parts contain appropriate schemas
        for part in res["parts"]:
            assert "label" in part
            assert "rgba_crop_path" in part
            assert "bbox" in part
            assert "ingest" in part
            
            # Verify transparent crops exist on disk
            crop_abs = settings.base_dir / part["rgba_crop_path"]
            assert crop_abs.exists()
            crop_abs.unlink()
            
            mask_abs = settings.base_dir / part["mask_path"]
            assert mask_abs.exists()
            mask_abs.unlink()
            
    finally:
        fp.run_schp_inference = original_schp_inference
        
    print("  - Fine parser routing and output coordinate schemas passed successfully.")

def run_all_tests():
    print("==========================================================")
    print("🔮 Vestir AI Pipeline Extensions Algorithm Verification Suite")
    print("==========================================================")
    
    try:
        test_model_registry()
        test_cutout_extractor_paths()
        test_semantic_mapper_extensions()
        test_extended_anatomical_fallback()
        test_fine_parser_fallback_routing()
        print("\n🎉 ALL PIPELINE EXTENSIONS AND FALLBACK TESTS PASSED SUCCESSFULLY!")
        print("==========================================================")
    except AssertionError as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ TEST SUITE FAILURE: {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ UNEXPECTED TEST ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_all_tests()
