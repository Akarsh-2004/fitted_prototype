import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.composer.alignment import _alpha_bbox, _trim_alpha_components
from pipeline.parsing.mask_cleanup import MaskCleanup
from pipeline.parsing.occlusion_repair import build_blocked_mask, occlusion_repair
from pipeline.parsing.segformer_parser import segformer_parser


def test_neighbor_blocker_subtraction():
    shape = (100, 100)
    people = [
        {"polygon": [[10, 10], [70, 10], [70, 90], [10, 90]]},
        {"polygon": [[50, 20], [90, 20], [90, 80], [50, 80]]},
    ]
    blocked = build_blocked_mask(shape, people, selected_index=0, dilate_px=1)

    raw = np.zeros(shape, dtype=np.uint8)
    raw[20:80, 20:85] = 255
    allowed = np.zeros(shape, dtype=np.uint8)
    allowed[10:90, 10:70] = 255

    result = occlusion_repair.repair("top_garment", raw, allowed_mask=allowed, blocked_mask=blocked)
    assert np.sum(result.mask[:, 72:] > 0) == 0, "blocked neighbour region leaked into repaired top"
    assert result.metadata["blocked_pixels_removed"] > 0


def test_footwear_keeps_two_components():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[72:90, 15:35] = 255
    mask[72:90, 60:80] = 255
    cleaned, _, bbox = MaskCleanup.clean_mask(mask, fill_holes=False, preserve_components=2, min_area=20)
    assert np.sum(cleaned[72:90, 15:35] > 0) > 0
    assert np.sum(cleaned[72:90, 60:80] > 0) > 0
    assert bbox[0] <= 15 and bbox[2] >= 80


def test_composer_trims_stray_alpha_component():
    rgba = np.zeros((100, 100, 4), dtype=np.uint8)
    rgba[20:70, 20:70, :] = [100, 100, 100, 255]
    rgba[5:10, 90:98, :] = [100, 100, 100, 255]

    trimmed = _trim_alpha_components(rgba, "top")
    bbox = _alpha_bbox(trimmed)
    assert bbox == (20, 20, 70, 70), f"unexpected trimmed bbox: {bbox}"


def test_skin_like_bag_is_suppressed():
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    # BGR skin-ish patch.
    crop[45:70, 35:70] = [95, 135, 205]
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[45:70, 35:70] = 255
    skin_ratio = segformer_parser._skin_like_ratio(crop, mask)
    assert skin_ratio > 0.3
    assert segformer_parser._should_suppress_part("bag", mask, [35, 45, 70, 70], 100, 100, skin_ratio)


def test_flat_tiny_bottom_is_suppressed():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[55:66, 10:90] = 255
    assert segformer_parser._should_suppress_part("bottom_garment", mask, [10, 55, 90, 66], 100, 100, 0.0)


def test_top_amodal_fill_repairs_arm_gap():
    visible = np.zeros((100, 100), dtype=np.uint8)
    visible[25:75, 20:45] = 255
    visible[25:75, 60:85] = 255
    allowed = np.zeros((100, 100), dtype=np.uint8)
    allowed[15:85, 10:90] = 255

    result = occlusion_repair.complete_occluded_garment("top_garment", visible, allowed_mask=allowed)
    assert result.metadata["amodal_fill_applied"]
    assert np.sum(result.mask[35:65, 46:59] > 0) > 0, "arm-sized shirt gap was not filled"


def test_top_amodal_fill_connects_upper_lower_fragments():
    visible = np.zeros((120, 100), dtype=np.uint8)
    visible[20:45, 25:75] = 255
    visible[75:100, 25:75] = 255
    allowed = np.zeros((120, 100), dtype=np.uint8)
    allowed[10:110, 15:85] = 255

    result = occlusion_repair.complete_occluded_garment("top_garment", visible, allowed_mask=allowed)
    assert result.metadata["amodal_fill_applied"]
    assert np.sum(result.mask[48:72, 35:65] > 0) > 0, "upper/lower shirt fragments were not bridged"


def main():
    test_neighbor_blocker_subtraction()
    test_footwear_keeps_two_components()
    test_composer_trims_stray_alpha_component()
    test_skin_like_bag_is_suppressed()
    test_flat_tiny_bottom_is_suppressed()
    test_top_amodal_fill_repairs_arm_gap()
    test_top_amodal_fill_connects_upper_lower_fragments()
    print("Occlusion repair verification passed.")


if __name__ == "__main__":
    main()
