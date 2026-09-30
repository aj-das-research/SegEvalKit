import numpy as np
import pytest

from segevalkit import EvaluationResult, synthetic
from segevalkit.guide import Fingerprint, recommend
from segevalkit.metrics import compute_metrics
from segevalkit.stats import presence_detection, adjust_pvalues, bland_altman, bootstrap_ci, compare, icc, rank_methods, ranking_stability
from conftest import sphere


def _res(name, vals):
    rows = [{"case_id": f"c{i}", "label": "x", "metric": "dice", "value": v} for i, v in enumerate(vals)]
    return EvaluationResult.from_rows(rows, meta={"name": name})


def test_adjust_pvalues():
    p = [0.01, 0.04, 0.03, 0.5]
    np.testing.assert_allclose(adjust_pvalues(p, "holm"), [0.04, 0.09, 0.09, 0.5])
    np.testing.assert_allclose(adjust_pvalues(p, "bh"), [0.04, 0.05333333, 0.05333333, 0.5])


def test_bootstrap_ci_contains_mean():
    x = np.random.default_rng(0).normal(0.8, 0.05, 200)
    lo, hi = bootstrap_ci(x)
    assert lo < x.mean() < hi


def test_compare_detects_difference():
    rng = np.random.default_rng(0)
    base = rng.uniform(0.6, 0.9, 40)
    df = compare(_res("A", base + 0.05), _res("B", base))
    assert df["significant"].item()
    assert df["mean_diff"].item() == pytest.approx(0.05)
    assert df["frac_a_better"].item() == 1.0


def test_ranking_and_stability():
    rng = np.random.default_rng(0)
    base = rng.uniform(0.6, 0.9, 30)
    res = {"good": _res("good", base + 0.1), "mid": _res("mid", base), "bad": _res("bad", base - 0.1)}
    for scheme in ("aggregate-then-rank", "rank-then-aggregate"):
        r = rank_methods(res, "dice", scheme=scheme)
        assert list(r["method"]) == ["good", "mid", "bad"]
    s = ranking_stability(res, "dice", n_boot=50)
    assert np.all(s["kendall_tau"] == 1.0)


def test_icc_and_bland_altman():
    x = np.linspace(10, 100, 30)
    assert icc(x, x) == pytest.approx(1.0)
    ba = bland_altman(x + 2, x)
    assert ba["bias"] == pytest.approx(2.0)
    assert ba["sd"] == pytest.approx(0.0, abs=1e-9)
    assert icc(x + 20, x) < icc(x + 20, x, kind="ICC(3,1)")  # absolute agreement penalises bias


def test_recommend_tubular_and_lesions():
    r = recommend(Fingerprint(structure="tubular"))
    assert r.metrics[0] == "cldice" and "betti0_error" in r.metrics
    r = recommend(structure="small_lesion", multi_instance=True, probabilistic=True)
    assert {"lesion_f1", "ece", "nsd"} <= set(r.metrics)
    assert r.params["ece"]["roi"] == "band"
    assert "|" in r.to_markdown()


@pytest.mark.parametrize("name", list(synthetic.PERTURBATIONS))
def test_perturbations_zero_is_identity(name):
    g = sphere(radius=9)
    fn = synthetic.PERTURBATIONS[name][0]
    np.testing.assert_array_equal(fn(g, 0, (1, 1, 1), rng=0), g)


def test_dilation_distance_is_physical():
    g = sphere(radius=9)
    p = synthetic.dilate(g, 2.0, (1, 1, 1))
    r = compute_metrics(p, g, ["hd", "nsd"], params={"nsd": {"tolerance_mm": 2.0}})
    assert r["hd"] == pytest.approx(2.0, abs=0.5)
    assert r["nsd"] == pytest.approx(1.0)


def test_sensitivity_study_shape():
    g = sphere(radius=8)
    df = synthetic.sensitivity_study([("s", g, (1, 1, 1))], ["dice", "hd95"],
                                     perturbations={"erode": [0, 1, 2], "islands": [0, 2]})
    assert set(df.columns) >= {"perturbation", "magnitude", "metric", "value"}
    d = df[(df.perturbation == "erode") & (df.metric == "dice")].sort_values("magnitude")["value"].to_numpy()
    assert d[0] == 1.0 and np.all(np.diff(d) < 0)


def test_presence_detection():
    rows = []
    for i in range(20):
        pos = i < 10
        rows += [{"case_id": f"c{i}", "label": "t", "metric": "_ref_volume_ml", "value": 1.0 if pos else 0.0},
                 {"case_id": f"c{i}", "label": "t", "metric": "_pred_volume_ml", "value": (2.0 + i) if pos else i * 0.1},
                 {"case_id": f"c{i}", "label": "t", "metric": "dice", "value": 0.5}]
    r = presence_detection(EvaluationResult.from_rows(rows), "t")
    assert r["auc"] == pytest.approx(1.0)
    assert r["sensitivity"] == 1.0 and r["specificity"] >= 0.9
    assert r["n_pos"] == 10 and r["n_neg"] == 10
