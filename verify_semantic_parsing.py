import sys
import os
import cv2
import torch
import numpy as np

# Ensure root folder is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.parsing.schp.labels import LIP_LABELS, LIP_LABEL_TO_ID
from pipeline.parsing.schp.preprocess import preprocess_image
from pipeline.parsing.schp.postprocess import postprocess_logits
from pipeline.parsing.semantic_mapper import semantic_mapper
from pipeline.parsing.mask_cleanup import MaskCleanup
from pipeline.parsing.garment_refiner import garment_refiner
from pipeline.detectors.schp_parser import human_clothing_parser


def test_labels_and_mappings():
    print("🧪 [1/6] Testing LIP Label Definitions & Mapping Integrity...")
    assert LIP_LABELS[0] == "background"
    assert LIP_LABELS[5] == "upper-clothes"
    assert LIP_LABELS[6] == "dress"
    assert LIP_LABELS[7] == "coat"
    assert LIP_LABELS[9] == "pants"
    assert LIP_LABELS[12] == "skirt"
    assert LIP_LABELS[18] == "left-shoe"
    assert LIP_LABELS[19] == "right-shoe"
    
    assert LIP_LABEL_TO_ID["coat"] == 7
    assert LIP_LABEL_TO_ID["pants"] == 9
    print("  - Label maps verified successfully.")


def test_preprocess_and_postprocess():
    print("\n🧪 [2/6] Testing Preprocessing & Postprocessing Tensors...")
    
    # 1. Preprocessing check
    mock_img = np.random.randint(0, 255, (200, 150, 3), dtype=np.uint8)
    tensor = preprocess_image(mock_img, target_size=473)
    
    assert tensor.shape == (1, 3, 473, 473), "Preprocessed tensor shape mismatch"
    assert tensor.dtype == torch.float32, "Preprocessed tensor should be float32"
    
    # 2. Postprocessing check
    mock_logits = torch.randn(1, 20, 473, 473)
    seg_map, conf_map, label_masks = postprocess_logits(
        logits=mock_logits,
        original_size=(150, 200), # (width, height)
        confidence_threshold=0.55
    )
    
    assert seg_map.shape == (200, 150), "Resized segmentation map shape mismatch"
    assert conf_map.shape == (200, 150), "Resized confidence map shape mismatch"
    assert len(label_masks) == 20, "Should have 20 binary masks"
    assert "upper-clothes" in label_masks
    assert label_masks["upper-clothes"].shape == (200, 150), "Individual mask shape mismatch"
    print("  - Preprocess and postprocess tensor layers verified successfully.")


def test_semantic_mapper_rules():
    print("\n🧪 [3/6] Testing Semantic Mapper & Advanced Dress/Layering Rules...")
    h, w = 100, 100
    
    # Create empty mock label masks
    mock_label_masks = {name: np.zeros((h, w), dtype=np.uint8) for name in LIP_LABELS.values()}
    
    # Test case A: Pants and Shoes mapping
    mock_label_masks["pants"][30:70, 20:80] = 255
    mock_label_masks["left-shoe"][80:95, 20:45] = 255
    mock_label_masks["right-shoe"][80:95, 55:80] = 255
    
    mapped = semantic_mapper.map_to_garments(mock_label_masks)
    
    assert "bottom" in mapped, "Pants should map to bottom"
    assert "shoes" in mapped, "Left + right shoes should merge into shoes"
    assert np.all(mapped["bottom"][30:70, 20:80] == 255)
    assert np.all(mapped["shoes"][80:95, 20:45] == 255)
    
    # Test case B: Dress integrity suppression
    # A dress is active, overlapping with a tiny top mask (noise)
    mock_label_masks = {name: np.zeros((h, w), dtype=np.uint8) for name in LIP_LABELS.values()}
    mock_label_masks["dress"][10:80, 20:80] = 255 # Dress is large (70x60 = 4200 pixels > 5% of 10000)
    mock_label_masks["upper-clothes"][15:40, 25:75] = 255 # Overlapping top
    
    mapped = semantic_mapper.map_to_garments(mock_label_masks)
    assert "dress" in mapped
    assert "top" not in mapped, "Overlapping top inside dress should be suppressed by integrity rule"
    print("  - Dress suppression and footwear merger rules verified successfully.")


def test_mask_cleanup():
    print("\n🧪 [4/6] Testing Mask Cleanup Morphology & Contour Smoothing...")
    h, w = 100, 100
    
    # Create a noisy mask: a square with a small hole, and a tiny stray component
    noisy_mask = np.zeros((h, w), dtype=np.uint8)
    noisy_mask[20:60, 20:60] = 255
    
    # Add a hole
    noisy_mask[35:45, 35:45] = 0
    
    # Add tiny noise (isolated pixel cluster)
    noisy_mask[80:85, 80:85] = 255
    
    # Clean the mask
    cleaned, poly, bbox = MaskCleanup.clean_mask(noisy_mask)
    
    # Assert hole is filled
    assert np.all(cleaned[35:45, 35:45] == 255), "Hole filling failed"
    # Assert stray tiny noise is filtered out
    assert not np.any(cleaned[80:85, 80:85] == 255), "Stray component filtering failed"
    # Assert simplified polygon and bounding box are correct
    assert bbox == [20.0, 20.0, 60.0, 60.0], f"Bounding box incorrect: {bbox}"
    assert len(poly) >= 4, "Simplified polygon should have at least 4 points"
    print("  - Hole filling, morphological smoothing, and tiny noise removal verified successfully.")


def test_garment_refiner_grabcut():
    print("\n🧪 [5/6] Testing SAM2-style GrabCut Garment Edge Refiner...")
    h, w = 100, 100
    
    # Generate synthetic image crop: black background with a grey shirt rectangle
    synthetic_img = np.zeros((h, w, 3), dtype=np.uint8)
    synthetic_img[20:70, 20:80] = [120, 120, 120]  # grey shirt
    
    # A slightly off/coarse semantic mask
    coarse_mask = np.zeros((h, w), dtype=np.uint8)
    coarse_mask[25:65, 25:75] = 255  # slightly smaller and shifted
    
    # Refine mask using GrabCut
    refined = garment_refiner.refine_garment_mask(
        img=synthetic_img,
        semantic_mask=coarse_mask,
        bbox=[25.0, 25.0, 75.0, 65.0],
        iterations=3
    )
    
    assert refined.shape == (h, w)
    # The GrabCut edge snapping should align closer to the synthetic shirt boundaries [20:70, 20:80]
    refined_area = np.sum(refined > 0)
    coarse_area = np.sum(coarse_mask > 0)
    
    print(f"  - Coarse semantic mask area: {coarse_area} pixels")
    print(f"  - GrabCut refined mask area: {refined_area} pixels (Snapped closer to grey shirt)")
    
    # Snapped mask should have expanded to capture the grey shirt edges
    assert refined_area > coarse_area, "GrabCut edge refinement should snap and expand to true edges"
    print("  - SAM2/GrabCut hybrid boundary snapping verified successfully.")


def test_high_availability_fallback():
    print("\n🧪 [6/6] Testing High-Availability Automatic Fallback Engine...")
    
    # Load a real human crop BGR image
    h, w = 400, 300
    mock_human_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Add a synthetic human block
    mock_human_img[50:350, 80:220] = [200, 200, 200]
    
    # Create person mask and bounding box
    person_mask = np.zeros((h, w), dtype=np.uint8)
    person_mask[50:350, 100:200] = 255
    person_box = [100.0, 50.0, 200.0, 350.0]
    
    # We call HumanClothingParser. Since we don't have checkpoint weights loaded yet,
    # the PyTorch model loader will throw a download failure or load failure (unless we download it),
    # which should instantly activate the geometry-based anatomical fallback parser.
    # Assert that this returns three layers (upper, lower, shoes) with zero exceptions raised!
    from pipeline.parsing.garment_mask_builder import garment_mask_builder
    
    # Force semantic parser to return empty to verify geometric fallback path
    original_parse = garment_mask_builder.parse_garments
    garment_mask_builder.parse_garments = lambda img, mask, box: []
    
    try:
        layers = human_clothing_parser.parse_clothing_layers(
            img=mock_human_img,
            person_mask=person_mask,
            person_box=person_box
        )
        
        assert len(layers) > 0, "Fallback parser should return layers"
        
        types = [layer["layer_type"] for layer in layers]
        print(f"  - Fallback parser successfully extracted layer types: {types}")
        
        assert "upper" in types, "Should extract upper clothes"
        assert "lower" in types, "Should extract lower clothes"
        assert "shoes" in types, "Should extract shoes"
        
        for layer in layers:
            assert "box" in layer
            assert "polygon" in layer
            assert "mask" in layer
            assert layer["mask"].shape == (h, w), "Layer mask should be full size"
            
        print("  - Zero-exception high-availability fallback verified successfully.")
    except Exception as e:
        assert False, f"Automatic fallback crashed with exception: {e}"
    finally:
        garment_mask_builder.parse_garments = original_parse


def run_all_tests():
    print("==========================================================")
    print("🔮 Vestir AI Semantic Ingestion Parsing Verification Suite")
    print("==========================================================")
    
    try:
        test_labels_and_mappings()
        test_preprocess_and_postprocess()
        test_semantic_mapper_rules()
        test_mask_cleanup()
        test_garment_refiner_grabcut()
        test_high_availability_fallback()
        print("\n🎉 ALL SEMANTIC PARSING MODULE TESTS PASSED SUCCESSFULLY!")
        print("==========================================================")
    except AssertionError as e:
        print(f"\n❌ VERIFICATION SUITE FAILURE: {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ UNEXPECTED SUITE ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_all_tests()
