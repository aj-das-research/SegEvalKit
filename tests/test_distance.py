import numpy as np
import pytest

from segevalkit.metrics import EmptyPolicy, compute_metrics
from conftest import cube, sphere


@pytest.mark.parametrize("spacing", [(1, 1, 1), (0.7, 0.7, 2.5)])
def test_shift_hausdorff_equals_shift(spacing):
    g = cube()
    p = np.roll(g, 3, axis=0)
    r = compute_metrics(p, g, ["hd", "hd95"], spacing=spacing)
    assert r["hd"] == pytest.approx(3 * spacing[0])


def test_identical_zero_distance(pair):
    _, g = pair
    r = compute_metrics(g, g, "distance", spacing=(0.8, 0.8, 1.5))
    assert r["hd"] == r["hd95"] == r["assd"] == r["masd"] == 0
    assert r["nsd"] == 1.0 and r["boundary_iou"] == 1.0


def test_concentric_spheres():
    g = sphere(radius=12)
    p = sphere(radius=9)
    r = compute_metrics(p, g, ["hd", "assd", "nsd"], params={"nsd": {"tolerance_mm": 1.0}})
    assert r["hd"] == pytest.approx(3.0, abs=1.0)
    assert r["nsd"] < 0.2
    r2 = compute_metrics(p, g, ["nsd"], params={"nsd": {"tolerance_mm": 4.0}})
    assert r2["nsd"] == pytest.approx(1.0)


def test_hd95_pooled_not_larger(pair):
    p, g = pair
    d = compute_metrics(p, g, ["hd_percentile"], params={"hd_percentile": {"mode": "directed"}})["hd_percentile"]
    q = compute_metrics(p, g, ["hd_percentile"], params={"hd_percentile": {"mode": "pooled"}})["hd_percentile"]
    assert q <= d + 1e-9


def test_one_empty_penalty():
    z = np.zeros((20, 20, 20), bool)
    g = cube((20, 20, 20), (5, 5, 5), (4, 4, 4))
    r = compute_metrics(z, g, ["hd95", "nsd"], spacing=(1, 1, 2))
    assert r["hd95"] == pytest.approx(np.sqrt(20 ** 2 + 20 ** 2 + 40 ** 2))
    assert r["nsd"] == 0.0
    r = compute_metrics(z, g, ["hd95"], empty=EmptyPolicy.preset("brats2023"))
    assert r["hd95"] == 374.0


def test_distance_monotone_in_error():
    g = sphere(radius=10)
    vals = [compute_metrics(np.roll(g, k, 0), g, ["assd"])["assd"] for k in (0, 1, 2, 4)]
    assert vals == sorted(vals)
