import numpy as np
import pytest

from segevalkit.metrics import compute_metrics
from conftest import sphere


def test_perfect_probabilities():
    g = sphere(radius=8)
    prob = g.astype(np.float32)
    r = compute_metrics(g, g, "calibration", prob=prob)
    assert r["auroc"] == pytest.approx(1.0)
    assert r["auprc"] == pytest.approx(1.0)
    assert r["brier"] == pytest.approx(0.0)
    assert r["ece"] == pytest.approx(0.0, abs=1e-9)
    assert r["soft_dice"] == pytest.approx(1.0)


def test_auroc_matches_sklearn_definition():
    rng = np.random.default_rng(1)
    g = sphere(shape=(24, 24, 24), radius=7)
    prob = np.clip(g * 0.6 + rng.random(g.shape) * 0.5, 0, 1).astype(np.float32)
    r = compute_metrics(g, g, ["auroc"], prob=prob)
    pos, neg = prob[g], prob[~g]
    # Mann-Whitney U on a subsample (exact AUROC up to 1/1024 bin ties)
    u = (pos[:, None] > neg[None, ::7]).mean() + 0.5 * (pos[:, None] == neg[None, ::7]).mean()
    assert r["auroc"] == pytest.approx(u, abs=5e-3)


def test_prob_metrics_skipped_without_prob():
    g = sphere(radius=5)
    assert "ece" not in compute_metrics(g, g, ["dice", "ece"])


def test_overconfident_has_higher_ece():
    rng = np.random.default_rng(2)
    g = sphere(shape=(24, 24, 24), radius=7)
    noisy = g ^ (rng.random(g.shape) < 0.1)  # 10 % label noise
    calibrated = np.where(noisy, 0.9, 0.1).astype(np.float32)
    overconf = np.where(noisy, 0.999, 0.001).astype(np.float32)
    e1 = compute_metrics(g, g, ["ece"], prob=calibrated)["ece"]
    e2 = compute_metrics(g, g, ["ece"], prob=overconf)["ece"]
    assert e2 > e1
