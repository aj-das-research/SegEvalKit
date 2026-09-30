import numpy as np
import pytest

from segevalkit.metrics import PairContext, compute_metrics, lesion_table
from conftest import cube

S = (40, 40, 40)


def lesions(*boxes):
    m = np.zeros(S, bool)
    for lo, size in boxes:
        m |= cube(S, lo, size)
    return m


def test_detect_miss_and_fp():
    g = lesions(((2, 2, 2), (5, 5, 5)), ((20, 20, 20), (6, 6, 6)), ((30, 2, 2), (3, 3, 3)))
    p = lesions(((2, 2, 2), (5, 5, 5)), ((21, 21, 21), (6, 6, 6)), ((2, 30, 30), (4, 4, 4)))
    r = compute_metrics(p, g, "detection")
    assert r["lesion_recall"] == pytest.approx(2 / 3)
    assert r["lesion_precision"] == pytest.approx(2 / 3)
    assert r["lesion_f1"] == pytest.approx(2 / 3)
    assert r["false_positive_lesions"] == 1 and r["false_negative_lesions"] == 1
    assert r["lesion_count_difference"] == 0
    # lesion-wise Dice: (1 + dice_shifted + 0) / (3 refs + 1 FP)
    d2 = 2 * 5 ** 3 / (2 * 6 ** 3)
    assert r["lesionwise_dice"] == pytest.approx((1 + d2) / 4)


def test_split_and_merge():
    g = lesions(((5, 5, 5), (10, 4, 4)))
    p = lesions(((5, 5, 5), (4, 4, 4)), ((11, 5, 5), (4, 4, 4)))
    r = compute_metrics(p, g, ["split_count", "merge_count", "lesion_recall"])
    assert r["split_count"] == 1 and r["merge_count"] == 0 and r["lesion_recall"] == 1
    r = compute_metrics(g, p, ["split_count", "merge_count"])
    assert r["merge_count"] == 1


def test_panoptic_quality():
    g = lesions(((2, 2, 2), (6, 6, 6)), ((20, 20, 20), (6, 6, 6)))
    r = compute_metrics(g, g, ["panoptic_quality"])
    assert r["panoptic_quality"] == pytest.approx(1.0)
    p = lesions(((2, 2, 2), (6, 6, 6)))
    # one TP with IoU 1, one FN: PQ = 1 / (1 + 0.5)
    assert compute_metrics(p, g, ["panoptic_quality"])["panoptic_quality"] == pytest.approx(2 / 3)


def test_lesion_table_rows():
    g = lesions(((2, 2, 2), (5, 5, 5)))
    p = lesions(((2, 2, 2), (5, 5, 5)), ((20, 20, 20), (3, 3, 3)))
    rows = lesion_table(PairContext(p, g, spacing=(1, 1, 2)))
    kinds = sorted(r["kind"] for r in rows)
    assert kinds == ["pred_fp", "ref"]
    ref = [r for r in rows if r["kind"] == "ref"][0]
    assert ref["volume_ml"] == pytest.approx(125 * 2 / 1000)
    assert ref["detected"] and ref["dice"] == pytest.approx(1.0)


def test_min_component_voxels_filters_noise():
    g = lesions(((2, 2, 2), (5, 5, 5)))
    p = g.copy()
    p[30, 30, 30] = True
    assert compute_metrics(p, g, ["false_positive_lesions"])["false_positive_lesions"] == 1
    assert compute_metrics(p, g, ["false_positive_lesions"], min_component_voxels=2)["false_positive_lesions"] == 0


def test_min_overlap_grazing_component_is_false_positive():
    g = lesions(((5, 5, 5), (10, 10, 10)))
    p = lesions(((14, 5, 5), (6, 6, 6)))          # overlaps 1 slab of 10x... only a sliver of the lesion
    r = compute_metrics(p, g, ["lesion_recall", "lesion_precision", "lesion_f1"],
                        params={k: {"min_overlap": 0.5} for k in ("lesion_recall", "lesion_precision", "lesion_f1")})
    assert r["lesion_recall"] == 0.0 and r["lesion_precision"] == 0.0 and r["lesion_f1"] == 0.0
