"""Conformance with widely used third-party implementations.

Where conventions differ between tools, the test pins down *which* tool
SegEvalKit agrees with under which setting (see docs: "Conventions").
"""

import numpy as np
import pytest

from segevalkit.metrics import compute_metrics

pytestmark = pytest.mark.reference

SPACINGS = [(1.0, 1.0, 1.0), (0.8, 0.8, 2.5)]


@pytest.mark.parametrize("spacing", SPACINGS)
def test_against_medpy(pair, spacing):
    mb = pytest.importorskip("medpy.metric.binary")
    p, g = pair
    r = compute_metrics(p, g, ["dice", "iou", "hd", "hd_percentile", "assd", "precision", "recall",
                               "relative_volume_difference"],
                        spacing=spacing, params={"hd_percentile": {"mode": "pooled"}})
    assert r["dice"] == pytest.approx(mb.dc(p, g))
    assert r["iou"] == pytest.approx(mb.jc(p, g))
    assert r["precision"] == pytest.approx(mb.precision(p, g))
    assert r["recall"] == pytest.approx(mb.recall(p, g))
    assert r["hd"] == pytest.approx(mb.hd(p, g, voxelspacing=spacing))
    # MedPy's hd95 is the percentile of the pooled distances ...
    assert r["hd_percentile"] == pytest.approx(mb.hd95(p, g, voxelspacing=spacing))
    # ... and since MedPy 0.5.2 its "assd" is the mean of the pooled distances
    # (our ASSD); MedPy <= 0.5.1 averaged the two directed means (our MASD).
    assert r["assd"] == pytest.approx(mb.assd(p, g, voxelspacing=spacing))
    assert r["relative_volume_difference"] == pytest.approx(mb.ravd(p, g))


@pytest.mark.parametrize("spacing", SPACINGS)
def test_against_monai(pair, spacing):
    torch = pytest.importorskip("torch")
    mm = pytest.importorskip("monai.metrics")
    p, g = pair
    P = torch.from_numpy(p[None, None].astype(np.float32))
    G = torch.from_numpy(g[None, None].astype(np.float32))
    r = compute_metrics(p, g, ["dice", "hd", "hd95", "assd", "nsd"], spacing=spacing,
                        params={"nsd": {"tolerance_mm": 2.0}})
    assert r["dice"] == pytest.approx(float(mm.compute_dice(P, G, include_background=True)), rel=1e-6)
    hd = float(mm.compute_hausdorff_distance(P, G, include_background=True, spacing=list(spacing)))
    hd95 = float(mm.compute_hausdorff_distance(P, G, include_background=True, percentile=95,
                                               spacing=list(spacing)))
    assd = float(mm.compute_average_surface_distance(P, G, include_background=True, symmetric=True,
                                                     spacing=list(spacing)))
    nsd = float(mm.compute_surface_dice(P, G, class_thresholds=[2.0], include_background=True,
                                        spacing=list(spacing)))
    assert r["hd"] == pytest.approx(hd, rel=1e-5)
    assert r["hd95"] == pytest.approx(hd95, rel=1e-5)
    assert r["assd"] == pytest.approx(assd, rel=1e-5)
    assert r["nsd"] == pytest.approx(nsd, rel=1e-5)


def test_against_deepmind_surface_distance(pair):
    sd = pytest.importorskip("surface_distance")
    from conftest import sphere

    p, g = pair
    spacing = (1.0, 1.0, 1.0)
    # The maximum distance does not depend on how surface elements are weighted.
    dists = sd.compute_surface_distances(g, p, spacing)
    assert compute_metrics(p, g, ["hd"], spacing=spacing)["hd"] == pytest.approx(
        sd.compute_robust_hausdorff(dists, 100))
    # Percentiles and NSD do: DeepMind weights each surface element by its area,
    # so isolated speckle voxels (6 exposed faces) weigh more than in the
    # voxel-counting convention of MONAI / MetricsReloaded / SegEvalKit. On a
    # smooth pair the two conventions stay close (documented in "Conventions").
    g = sphere(radius=11)
    p = np.roll(sphere(radius=10), 2, axis=0)
    dists = sd.compute_surface_distances(g, p, spacing)
    r = compute_metrics(p, g, ["hd95", "nsd"], spacing=spacing, params={"nsd": {"tolerance_mm": 1.0}})
    assert r["hd95"] == pytest.approx(sd.compute_robust_hausdorff(dists, 95), abs=1.0)
    assert r["nsd"] == pytest.approx(sd.compute_surface_dice_at_tolerance(dists, 1.0), abs=0.06)
