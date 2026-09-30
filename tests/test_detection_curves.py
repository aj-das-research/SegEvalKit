"""Lesion scores, lesion-wise HD95 / NSD, lesion AP, FROC / CPM, PR curves and localized presence."""

import numpy as np
import pandas as pd
import pytest

from segevalkit import EvaluationResult, Evaluator
from segevalkit.metrics import EmptyPolicy, PairContext, compute_metrics, get_metric, lesion_table
from segevalkit.stats import froc, lesion_pr, localized_presence, patient_pr, presence_detection
from conftest import cube

S = (40, 40, 40)


def boxes(*bs):
    m = np.zeros(S, bool)
    for lo, size in bs:
        m |= cube(S, lo, size)
    return m


A = ((2, 2, 2), (5, 5, 5))
B = ((20, 20, 20), (6, 6, 6))
B_SHIFT = ((21, 20, 20), (6, 6, 6))
C = ((30, 2, 2), (3, 3, 3))
D = ((2, 30, 30), (4, 4, 4))


# ------------------------------------------------------------------ scores
def test_scores_from_probability_and_volume():
    g = boxes(A, B)
    p = boxes(A, D)
    prob = np.zeros(S, np.float32)
    prob[boxes(A)] = 0.9
    prob[2, 2, 2] = 0.95
    prob[boxes(D)] = 0.4
    rows = lesion_table(PairContext(p, g, prob=prob))
    ref = [r for r in rows if r["kind"] == "ref"]
    fp = [r for r in rows if r["kind"] == "pred_fp"]
    assert ref[0]["score"] == pytest.approx(0.95) and ref[0]["score_type"] == "max_prob"
    assert np.isnan(ref[1]["score"])  # missed lesion: never detected at any threshold
    assert fp[0]["score"] == pytest.approx(0.4)
    mean_rows = lesion_table(PairContext(p, g, prob=prob), score="mean")
    assert mean_rows[0]["score"] == pytest.approx((0.9 * 124 + 0.95) / 125)
    vol = lesion_table(PairContext(p, g, spacing=(1, 1, 2)))
    assert vol[0]["score_type"] == "volume_ml" and vol[0]["score"] == pytest.approx(125 * 2 / 1000)


def test_evaluator_writes_scores():
    g = boxes(A, B).astype(np.uint8)
    p = boxes(A, D).astype(np.uint8)
    prob = np.where(p > 0, 0.7, 0.0).astype(np.float32)
    res = Evaluator(labels={"t": 1}, metrics=["lesion_f1"]).evaluate_arrays(p, g, prob={"t": prob})
    assert {"score", "score_type"} <= set(res.lesions.columns)
    assert res.lesions["score_type"].iloc[0] == "max_prob"


# ----------------------------------------------------------- lesion-wise
def test_lesionwise_hd95_and_nsd():
    g = boxes(A, B, C)
    p = boxes(A, B_SHIFT, D)
    ctx = PairContext(p, g, empty=EmptyPolicy.preset("brats2023"))
    shifted = compute_metrics(boxes(B_SHIFT), boxes(B), ["hd95", "nsd"])
    # A perfect, B shifted, C missed (374 mm, NSD 0), D false positive (374 mm, NSD 0)
    assert get_metric("lesionwise_hd95")(ctx) == pytest.approx((0 + shifted["hd95"] + 374 + 374) / 4)
    assert get_metric("lesionwise_nsd")(ctx) == pytest.approx((1 + shifted["nsd"] + 0 + 0) / 4)
    # no lesions anywhere: ideal values
    empty = PairContext(np.zeros(S, bool), np.zeros(S, bool))
    assert get_metric("lesionwise_hd95")(empty) == 0.0 and get_metric("lesionwise_nsd")(empty) == 1.0


def test_lesion_ap_ranked_by_probability():
    g = boxes(A, B)
    p = boxes(A, B, D)
    prob = np.zeros(S, np.float32)
    prob[boxes(A)], prob[boxes(D)], prob[boxes(B)] = 0.9, 0.8, 0.3
    # ranks: A (R .5, P 1), D (R .5, P 1/2), B (R 1, P 2/3) -> AP = .5 * 1 + .5 * 2/3
    assert compute_metrics(p, g, ["lesion_ap"], prob=prob)["lesion_ap"] == pytest.approx(0.5 + 0.5 * 2 / 3)
    assert compute_metrics(boxes(A, B), g, ["lesion_ap"])["lesion_ap"] == pytest.approx(1.0)


# ---------------------------------------------------------------- FROC
def _result_with_lesions(ref_scores, fp_scores, n_cases=4):
    rows = [{"case_id": f"c{i}", "label": "t", "metric": "dice", "value": 0.5} for i in range(n_cases)]
    les = [{"case_id": f"c{i % n_cases}", "label": "t", "kind": "ref", "score": s, "score_type": "max_prob"}
           for i, s in enumerate(ref_scores)]
    les += [{"case_id": f"c{i % n_cases}", "label": "t", "kind": "pred_fp", "score": s, "score_type": "max_prob"}
            for i, s in enumerate(fp_scores)]
    return EvaluationResult.from_rows(rows, les)


def test_froc_and_cpm_by_hand():
    r = _result_with_lesions([0.9, 0.7, np.nan, 0.4], [0.8, 0.5, 0.3, 0.2])
    f = froc(r, "t", n_boot=200)
    # thresholds .9 .8 .7 .5 .4 .3 .2 -> (fp/scan, sens): (0,.25) (.25,.25) (.25,.5) (.5,.5) (.5,.75) (.75,.75) (1,.75)
    np.testing.assert_allclose(f["fps"], [0, 0, .25, .25, .5, .5, .75, 1.0])
    np.testing.assert_allclose(f["sensitivity"], [0, .25, .25, .5, .5, .75, .75, .75])
    # best sensitivity per FP rate: 0 -> .25, .25 -> .5, .5 -> .75, then .75
    assert f["sensitivity_at"][0.125] == pytest.approx(0.375)
    assert f["sensitivity_at"][0.25] == pytest.approx(0.5)
    assert f["sensitivity_at"][8.0] == pytest.approx(0.75)
    assert f["cpm"] == pytest.approx((0.375 + 0.5 + 0.75 * 5) / 7)
    assert f["max_sensitivity"] == 0.75 and f["n_ref"] == 4 and f["n_cases"] == 4
    lo, hi = f["cpm_ci"]
    assert lo <= f["cpm"] <= hi


def test_lesion_pr_ap():
    r = _result_with_lesions([0.9, 0.7], [0.8], n_cases=2)
    d = lesion_pr(r, "t")
    # .9: R .5 P 1; .8: R .5 P .5; .7: R 1 P 2/3 -> AP = .5 + .5 * 2/3
    assert d["ap"] == pytest.approx(0.5 + 0.5 * 2 / 3)


def test_missing_scores_raise():
    rows = [{"case_id": "c0", "label": "t", "metric": "dice", "value": 1.0}]
    r = EvaluationResult.from_rows(rows, [{"case_id": "c0", "label": "t", "kind": "ref"}])
    with pytest.raises(ValueError):
        froc(r, "t")


# ------------------------------------------------------------ presence
def _patients():
    # P0 flagged and correctly localized; P1 flagged by a blob elsewhere; P2, P3 tumour-free
    spec = [("p0", 1.0, 5.0, 1, 1), ("p1", 1.0, 3.0, 0, 1), ("p2", 0.0, 0.0, 0, 0), ("p3", 0.0, 0.1, 0, 0)]
    rows = []
    for cid, ref_v, pred_v, tp_ref, n_ref in spec:
        rows += [{"case_id": cid, "label": "t", "metric": m, "value": v} for m, v in
                 (("_ref_volume_ml", ref_v), ("_pred_volume_ml", pred_v), ("_tp_ref_lesions", tp_ref),
                  ("_n_ref_lesions", n_ref), ("dice", 0.5))]
    return EvaluationResult.from_rows(rows)


def test_localized_presence():
    r = _patients()
    plain = presence_detection(r, "t")
    loc = localized_presence(r, "t")
    assert loc["sensitivity"] == plain["sensitivity"] == 1.0
    assert loc["sensitivity_localized"] == 0.5
    assert loc["specificity"] == plain["specificity"] == 1.0
    assert loc["auc"] == pytest.approx(plain["auc"])
    assert loc["auc_localized"] < loc["auc"]
    assert loc["lesion_sensitivity"] == pytest.approx(0.5)


def test_patient_pr():
    r = _patients()
    assert patient_pr(r, "t")["ap"] == pytest.approx(1.0)
    # localized: P1 is a false call; ranking p0 (TP), p1 (FP), p3 (FP) -> recall stops at .5
    assert patient_pr(r, "t", localized=True)["ap"] == pytest.approx(0.5)


def test_plots_render():
    import matplotlib

    matplotlib.use("Agg")
    from segevalkit import plotting as P

    r = _result_with_lesions([0.9, 0.7, np.nan, 0.4], [0.8, 0.5, 0.3, 0.2])
    assert P.froc_plot(r, "t", n_boot=50) is not None
    assert P.pr_plot(r, "t") is not None
    p = _patients()
    assert P.roc_plot(p, "t", n_boot=50, reference_point={"sensitivity": .76, "specificity": .91}) is not None
    assert P.pr_plot(p, "t", level="patient") is not None


def test_lesion_score_roc_reaches_corner():
    # three positives with scores, one positive and two negatives without any predicted lesion
    rows, les = [], []
    for cid, pos, sc in (("a", 1, 0.9), ("b", 1, 0.8), ("c", 1, 0.7), ("d", 1, None), ("e", 0, None), ("f", 0, 0.75)):
        rows += [{"case_id": cid, "label": "t", "metric": m, "value": v} for m, v in
                 (("_ref_volume_ml", float(pos)), ("_pred_volume_ml", 0.0 if sc is None else 1.0), ("dice", 0.5))]
        if sc is not None:
            les.append({"case_id": cid, "label": "t", "kind": "ref" if pos else "pred_fp", "score": sc,
                        "score_type": "max_prob"})
        elif pos:
            les.append({"case_id": cid, "label": "t", "kind": "ref", "score": np.nan, "score_type": "max_prob"})
    r = EvaluationResult.from_rows(rows, les)
    d = localized_presence(r, "t", score="lesion")
    # ranking a .9, b .8, f .75 (neg), c .7, then d and e tied lowest -> AUC = (2*2 + 1*1 + 0.5*1*1) / (4*2)
    assert d["auc"] == pytest.approx((2 + 2 + 1 + 0.5) / 8)
