import numpy as np
import pytest

from segevalkit.metrics import betti_numbers, compute_metrics
from conftest import cube, sphere


def torus(shape=(50, 50, 30), R=14, r=5):
    x, y, z = np.indices(shape).astype(float)
    x -= shape[0] / 2
    y -= shape[1] / 2
    z -= shape[2] / 2
    return (np.sqrt(x ** 2 + y ** 2) - R) ** 2 + z ** 2 <= r ** 2


def test_betti_solid():
    assert betti_numbers(cube()) == (1, 0, 0)


def test_betti_two_components():
    m = cube((40, 40, 40), (2, 2, 2), (6, 6, 6)) | cube((40, 40, 40), (20, 20, 20), (6, 6, 6))
    assert betti_numbers(m) == (2, 0, 0)


def test_betti_torus():
    assert betti_numbers(torus()) == (1, 1, 0)


def test_betti_hollow_sphere():
    m = sphere(radius=12) & ~sphere(radius=7)
    assert betti_numbers(m) == (1, 0, 1)


def test_topology_errors():
    t = torus()
    cut = t.copy()
    cut[:, 22:28, :] = False  # breaks the ring into two arcs
    r = compute_metrics(cut, t, "topology")
    assert r["betti0_error"] == 1
    assert r["betti1_error"] == 1


def test_cldice_tube():
    g = np.zeros((60, 20, 20), bool)
    g[5:55, 7:12, 7:12] = True
    assert compute_metrics(g, g, ["cldice"])["cldice"] == pytest.approx(1.0)
    broken = g.copy()
    broken[28:34] = False
    v = compute_metrics(broken, g, ["cldice"])["cldice"]
    assert 0.8 < v < 1.0


def test_skeleton_even_width_not_empty():
    # Lee thinning deletes perfectly symmetric even-width bars; the fallback keeps a voxel.
    g = np.zeros((30, 12, 12), bool)
    g[2:28, 4:8, 4:8] = True
    assert compute_metrics(g, g, ["cldice"])["cldice"] == pytest.approx(1.0)
