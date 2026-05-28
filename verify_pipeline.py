import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.detectors.duplicate_filter import suppress_duplicates, calculate_iou
from pipeline.analysis.oklch_scorer import rgb_to_oklch, score_color_harmony
from pipeline.detectors.scene_classifier import scene_classifier

def test_iou_calculations():
    print("🧪 Testing Bounding Box IoU Duplicate Suppression...")
    box1 = [100.0, 100.0, 200.0, 200.0]
    box2 = [120.0, 120.0, 220.0, 220.0] # High overlap
    box3 = [500.0, 500.0, 600.0, 600.0] # Far away (no overlap)
    
    iou_12 = calculate_iou(box1, box2)
    iou_13 = calculate_iou(box1, box3)
    
    print(f"  - IoU between overlapping boxes: {iou_12:.4f} (Expected > 0.4)")
    print(f"  - IoU between non-overlapping boxes: {iou_13:.4f} (Expected 0.0)")
    
    assert iou_12 > 0.4, "Overlap IoU calculation failed"
    assert iou_13 == 0.0, "Non-overlap IoU calculation failed"
    
    # Test suppression
    detections = [
        {"box": box1, "confidence": 0.90},
        {"box": box2, "confidence": 0.85},
        {"box": box3, "confidence": 0.95}
    ]
    
    suppressed = suppress_duplicates(detections, iou_threshold=0.4)
    print(f"  - Suppressed list length: {len(suppressed)} (Expected: 2, kept highest conf box1 and box3)")
    
    assert len(suppressed) == 2, "IoU NMS duplicate suppression failed"
    assert suppressed[0]["confidence"] == 0.95, "Failed to keep highest confidence item"
    print("✅ IoU tests passed successfully!")

def test_oklch_color_harmony():
    print("\n🧪 Testing OKLCH Color Space & Harmony Recommendations...")
    
    # Test RGB to OKLCH conversion
    red_oklch = rgb_to_oklch((255, 0, 0))
    white_oklch = rgb_to_oklch((255, 255, 255))
    
    print(f"  - Red in OKLCH: Lightness={red_oklch[0]:.2f}, Chroma={red_oklch[1]:.2f}, Hue={red_oklch[2]:.1f}°")
    print(f"  - White in OKLCH: Lightness={white_oklch[0]:.2f}, Chroma={white_oklch[1]:.2f}, Hue={white_oklch[2]:.1f}°")
    
    assert red_oklch[0] > 0.3, "OKLCH red lightness conversion error"
    assert white_oklch[1] < 0.01, "OKLCH white chroma should be extremely near 0 (neutral)"
    
    # Test harmony matching
    red_colors = [{"rgb": [255, 0, 0], "weight": 1.0, "oklch": red_oklch}]
    white_colors = [{"rgb": [255, 255, 255], "weight": 1.0, "oklch": white_oklch}]
    
    # White is neutral and matches anything
    score_red_white = score_color_harmony(red_colors, white_colors)
    print(f"  - Harmony score (Red + White Neutral): {score_red_white:.2f} (Expected high ~0.95)")
    
    assert score_red_white >= 0.90, "Neutral harmony matching error"
    
    # Green complementary to red (~180 deg)
    green_oklch = rgb_to_oklch((0, 255, 0))
    green_colors = [{"rgb": [0, 255, 0], "weight": 1.0, "oklch": green_oklch}]
    
    score_red_green = score_color_harmony(red_colors, green_colors)
    print(f"  - Harmony score (Red + Green Complementary): {score_red_green:.2f} (Expected high >= 0.8)")
    
    assert score_red_green >= 0.75, "Complementary color harmony matching error"
    print("✅ OKLCH harmony tests passed successfully!")

def run_all_tests():
    print("==========================================================")
    print("🔮 Vestir AI Upgraded Pipeline Algorithm Verification Suite")
    print("==========================================================")
    
    try:
        test_iou_calculations()
        test_oklch_color_harmony()
        print("\n🎉 ALL CORE PIPELINE TESTS PASSED SUCCESSFULY!")
        print("==========================================================")
    except AssertionError as e:
        print(f"\n❌ TEST SUITE FAILURE: {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_all_tests()
