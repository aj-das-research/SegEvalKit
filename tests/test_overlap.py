import math

import numpy as np
import pytest

from segevalkit.metrics import EmptyPolicy, PairContext, compute_metrics
from conftest import cube


def test_identical_is_perfect():
    g = cube()
    r = compute_metrics(g, g, "overlap")
    for k in ("dice", "iou", "precision", "recall", "specificity", "mcc", "cohen_kappa", "balanced_accuracy"):
        assert r[k] == pytest.approx(1.0)
    assert r["voe"] == 0.0


def test_shift_known_values():
    g = cube()
    p = np.roll(g, 4, axis=0)  # overlap 12/16 of the cube
    r = compute_metrics(p, g, ["dice", "iou", "precision", "recall", "fbeta", "tversky"])
    assert r["dice"] == pytest.approx(12 / 16)
    assert r["iou"] == pytest.approx(12 / 20)
    assert r["precision"] == pytest.approx(0.75)
    assert r["recall"] == pytest.approx(0.75)
    assert r["fbeta"] == pytest.approx(0.75)  # P == R -> every F-beta equals it
    assert r["tversky"] == pytest.approx(0.75)


def test_dice_iou_relation(pair):
    p, g = pair
    r = compute_metrics(p, g, ["dice", "iou"])
    assert r["iou"] == pytest.approx(r["dice"] / (2 - r["dice"]))


def test_empty_policies():
    z = np.zeros((10, 10, 10), bool)
    g = cube((10, 10, 10), (2, 2, 2), (4, 4, 4))
    assert compute_metrics(z, z, ["dice"])["dice"] == 1.0
    assert math.isnan(compute_metrics(z, z, ["dice"], empty=EmptyPolicy("nan"))["dice"])
    r = compute_metrics(z, g, ["dice", "recall", "precision"])
    assert r["dice"] == 0.0 and r["recall"] == 0.0
    assert math.isnan(r["precision"])  # undefined, never a silent 0


def test_mcc_kappa_bounds(pair):
    p, g = pair
    r = compute_metrics(p, g, ["mcc", "cohen_kappa"])
    assert -1 <= r["mcc"] <= 1 and -1 <= r["cohen_kappa"] <= 1


def test_volume_metrics_physical_units():
    g = cube((20, 20, 20), (0, 0, 0), (10, 10, 10))
    p = cube((20, 20, 20), (0, 0, 0), (10, 10, 5))
    r = compute_metrics(p, g, "volume", spacing=(1, 1, 2))
    assert r["ref_volume"] == pytest.approx(2.0)   # 1000 voxels x 2 mm3 = 2 mL
    assert r["pred_volume"] == pytest.approx(1.0)
    assert r["relative_volume_difference"] == pytest.approx(-0.5)
    assert r["absolute_volume_difference"] == pytest.approx(1.0)
    assert r["volumetric_similarity"] == pytest.approx(1 - 500 / 1500)


def test_context_caches_counts(pair):
    p, g = pair
    ctx = PairContext(p, g)
    a = ctx.counts
    assert ctx.counts is a
    assert sum(a) == g.size
