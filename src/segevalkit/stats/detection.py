r"""Dataset-level detection analysis: FROC / CPM, lesion and patient PR curves, localized presence.

These functions pool over every case of an evaluation. They read the lesion
table (``result.lesions``), where every reference lesion carries the highest
confidence among the predicted components touching it and every
false-positive component carries its own confidence (see
`segevalkit.metrics.detection.component_scores`). Thresholding that score
reproduces what a detector would report at any operating point, without
re-reading the images.

Conventions (as in `segevalkit.metrics.detection`, ``"overlap"`` matching):
a reference lesion is *detected* at threshold \(t\) if a predicted component
with score \(\ge t\) overlaps it; a component with score \(\ge t\) that overlaps
no reference lesion is a false positive. Lesion-level precision counts
detected reference lesions as true positives,
\(TP_{\mathrm{les}}/(TP_{\mathrm{les}} + FP_{\mathrm{les}})\), the definition
behind the library's lesion F1.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["froc", "lesion_pr", "patient_pr", "localized_presence", "roc_band"]

#: FP-per-scan operating points of the Competition Performance Metric (LUNA16).
CPM_RATES = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def _lesion_scores(result, label: str):
    les = result.lesions
    if not len(les) or "score" not in les:
        raise ValueError("the result has no lesion scores: evaluate with detection metrics "
                         "(SegEvalKit >= 0.1.1 writes a 'score' column to lesions.csv)")
    les = les[les["label"] == label]
    w = result.wide()
    cases = w.loc[w["label"] == label, "case_id"].astype(str).unique()
    ref = les[les["kind"] == "ref"]
    fp = les[les["kind"] == "pred_fp"]
    stype = les["score_type"].dropna().iloc[0] if "score_type" in les and les["score_type"].notna().any() else "score"
    return cases, ref, fp, str(stype)


def _curve(ref_s: np.ndarray, fp_s: np.ndarray, n_cases: int, n_ref: int):
    """Sensitivity and FP per scan at every distinct score threshold, from strict to lenient."""
    ref_s = np.sort(np.where(np.isfinite(ref_s), ref_s, -np.inf))
    fp_s = np.sort(fp_s[np.isfinite(fp_s)])
    thr = np.unique(np.concatenate([ref_s[np.isfinite(ref_s)], fp_s]))[::-1]
    thr = np.concatenate([[np.inf], thr])
    # counts of scores >= t via searchsorted on the ascending arrays
    n_det = ref_s.size - np.searchsorted(ref_s, thr, side="left")
    n_fp = fp_s.size - np.searchsorted(fp_s, thr, side="left")
    sens = n_det / n_ref if n_ref else np.full(thr.shape, np.nan)
    return thr, sens, n_fp / max(n_cases, 1), n_det, n_fp


def _cpm(fps: np.ndarray, sens: np.ndarray, rates: Sequence[float]) -> np.ndarray:
    # LUNA16 (Setio et al. 2017): linear interpolation of the FROC curve at the
    # given FP rates. Each distinct FP rate keeps the highest sensitivity reached
    # there (the curve is a staircase), and beyond the largest FP rate reached the
    # final sensitivity holds.
    f, idx = np.unique(fps, return_inverse=True)
    best = np.full(f.shape, -np.inf)
    np.maximum.at(best, idx, sens)
    return np.interp(np.asarray(rates, float), f, np.maximum.accumulate(best))


def froc(result, label: str, fp_rates: Sequence[float] = CPM_RATES, n_boot: int = 1000, ci: float = 0.95,
         seed: int = 0) -> Dict:
    r"""Free-response ROC curve and Competition Performance Metric (CPM).

    Sweeps the lesion score threshold over all cases: at each threshold the
    lesion sensitivity (detected reference lesions / all reference lesions)
    is plotted against the mean number of false-positive lesions per scan
    (Chakraborty & Berbaum 2004). The CPM is the mean sensitivity at
    ``fp_rates`` false positives per scan, by default
    \(\{1/8, 1/4, 1/2, 1, 2, 4, 8\}\) (LUNA16, Setio et al. 2017), read off the
    curve by linear interpolation.

    Every evaluated case counts as a scan, including tumour-free ones. The
    confidence interval of the CPM resamples cases with replacement.

    Returns:
        Dict with ``fps``, ``sensitivity``, ``thresholds`` (the curve),
        ``cpm``, ``cpm_ci``, ``sensitivity_at`` (``{rate: sensitivity}``),
        ``max_sensitivity``, ``n_ref``, ``n_cases`` and ``score_type``.
    """
    cases, ref, fp, stype = _lesion_scores(result, label)
    n_cases, n_ref = len(cases), len(ref)
    thr, sens, fps, _, _ = _curve(ref["score"].to_numpy(float), fp["score"].to_numpy(float), n_cases, n_ref)
    at = _cpm(fps, sens, fp_rates) if n_ref else np.full(len(fp_rates), np.nan)
    out = {"fps": fps, "sensitivity": sens, "thresholds": thr, "cpm": float(np.mean(at)),
           "sensitivity_at": {float(r): float(v) for r, v in zip(fp_rates, at)},
           "max_sensitivity": float(sens[-1]) if n_ref else float("nan"),
           "n_ref": n_ref, "n_cases": n_cases, "score_type": stype, "cpm_ci": (np.nan, np.nan)}
    if n_boot and n_ref and n_cases > 1:
        idx = {c: i for i, c in enumerate(cases)}
        rc = ref["case_id"].astype(str).map(idx).to_numpy()
        fc = fp["case_id"].astype(str).map(idx).to_numpy()
        rs, fs = ref["score"].to_numpy(float), fp["score"].to_numpy(float)
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(n_boot):
            w = np.bincount(rng.integers(0, n_cases, n_cases), minlength=n_cases)  # copies of each case
            r_rep, f_rep = np.repeat(rs, w[rc]), np.repeat(fs, w[fc])
            if r_rep.size == 0:
                continue
            _, s_b, f_b, _, _ = _curve(r_rep, f_rep, n_cases, r_rep.size)
            boots.append(np.mean(_cpm(f_b, s_b, fp_rates)))
        a = (1 - ci) / 2
        out["cpm_ci"] = (float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a)))
    return out


def _pr_from_counts(tp: np.ndarray, fp: np.ndarray, n_pos: int):
    recall = tp / n_pos
    precision = np.divide(tp, tp + fp, out=np.ones_like(tp, dtype=float), where=(tp + fp) > 0)
    return recall, precision


def _ap(recall: np.ndarray, precision: np.ndarray) -> float:
    # all-point interpolated AP (PASCAL VOC 2010+ / COCO): precision made monotone in recall
    r = np.concatenate([[0.0], recall])
    p = np.concatenate([[1.0], precision])
    p = np.maximum.accumulate(p[::-1])[::-1]
    return float(np.sum(np.diff(r) * p[1:]))


def lesion_pr(result, label: str) -> Dict:
    """Lesion-level precision-recall curve over all cases and its average precision.

    Recall is lesion sensitivity; precision is
    \\(TP_{\\mathrm{les}}/(TP_{\\mathrm{les}} + FP_{\\mathrm{les}})\\) with detected
    reference lesions as true positives. AP is the all-point interpolated area.

    Returns:
        Dict with ``recall``, ``precision``, ``thresholds``, ``ap``, ``n_ref``,
        ``n_fp`` (at the most lenient threshold) and ``score_type``.
    """
    cases, ref, fp, stype = _lesion_scores(result, label)
    thr, sens, _, n_det, n_fp = _curve(ref["score"].to_numpy(float), fp["score"].to_numpy(float), len(cases), len(ref))
    if not len(ref):
        return {"recall": sens, "precision": np.full(sens.shape, np.nan), "thresholds": thr, "ap": float("nan"),
                "n_ref": 0, "n_fp": int(n_fp[-1]), "score_type": stype}
    recall, precision = _pr_from_counts(n_det.astype(float), n_fp.astype(float), len(ref))
    return {"recall": recall[1:], "precision": precision[1:], "thresholds": thr[1:],
            "ap": _ap(recall[1:], precision[1:]), "n_ref": len(ref), "n_fp": int(n_fp[-1]), "score_type": stype}


def _patient_scores(result, label: str, score: str, min_ref_ml: float):
    """Per-patient truth, score and localized score.

    ``score="pred_volume_ml"`` uses the predicted volume, and a flagged positive
    patient is localized if any predicted lesion overlaps a reference lesion.
    ``score="lesion"`` uses the highest lesion confidence of the patient, and
    the localized score is the highest confidence among components that touch
    a reference lesion, so the patient is localized at threshold t only if a
    *true* lesion is still called at t.
    """
    w = result.wide()
    w = w[w["label"] == label].copy()
    w["case_id"] = w["case_id"].astype(str)
    y = (w["ref_volume_ml"] > min_ref_ml).to_numpy()
    if score == "lesion":
        _, ref, fp, _ = _lesion_scores(result, label)
        ref_max = ref.groupby(ref["case_id"].astype(str))["score"].max()
        fp_max = fp.groupby(fp["case_id"].astype(str))["score"].max()
        loc = w["case_id"].map(ref_max).fillna(-np.inf).to_numpy(float)
        s = np.maximum(loc, w["case_id"].map(fp_max).fillna(-np.inf).to_numpy(float))
    else:
        s = w[score].to_numpy(float)
        hit = w["tp_ref_lesions"].to_numpy(float) > 0 if "tp_ref_lesions" in w else np.zeros(len(w), bool)
        loc = np.where(hit, s, -np.inf)
    # Patients without any predicted lesion rank below every scored one, so that the
    # most lenient threshold calls every patient positive and the ROC ends at (1, 1).
    # The localized score keeps -inf: calling such a patient never localizes a lesion.
    fin = s[np.isfinite(s)]
    s = np.where(np.isfinite(s), s, (fin.min() - 1.0) if fin.size else 0.0)
    return y, s, loc


def _frac_above(values: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """Fraction of ``values`` strictly above each threshold (NaN if ``values`` is empty)."""
    if values.size == 0:
        return np.full(thr.shape, np.nan)
    v = np.sort(values)
    return (v.size - np.searchsorted(v, thr, side="right")) / v.size


def _roc(y: np.ndarray, s: np.ndarray, s_pos: Optional[np.ndarray] = None):
    s_pos = s if s_pos is None else s_pos
    finite = np.concatenate([s[np.isfinite(s)], s_pos[np.isfinite(s_pos)]])
    thr = np.unique(np.concatenate([[-np.inf], finite, [np.inf]]))
    tpr = _frac_above(s_pos[y], thr)
    fpr = _frac_above(s[~y], thr)
    order = np.lexsort((tpr, fpr))
    trapz = getattr(np, "trapezoid", None) or np.trapz
    auc = float(trapz(tpr[order], fpr[order])) if y.any() and (~y).any() else float("nan")
    return thr, fpr, tpr, auc


def localized_presence(result, label: str, score: str = "pred_volume_ml", target_specificity: float = 0.9,
                       min_ref_ml: float = 0.0) -> Dict:
    """Patient-level detection with and without localization.

    Plain patient-level detection (`segevalkit.stats.presence_detection`)
    counts a patient with a lesion as found when the model flags the patient,
    even if the flagged blob is somewhere else. The *localized* variant
    additionally requires that at least one predicted lesion overlaps a
    reference lesion. Specificity is the same for both; sensitivity and AUC can
    only drop.

    Args:
        score: ``"pred_volume_ml"`` (predicted volume, the PanTS / R-Super
            protocol) or ``"lesion"`` (highest lesion confidence).

    Returns:
        Dict with ``auc``, ``auc_localized``, ``threshold``, ``specificity``,
        ``sensitivity``, ``sensitivity_localized``, ``lesion_sensitivity``
        (pooled over lesions at the same threshold when ``score="lesion"``,
        otherwise over all predictions), ``n_pos``, ``n_neg`` and the ROC arrays
        ``fpr``, ``tpr``, ``tpr_localized``, ``thresholds``.
    """
    y, s, loc = _patient_scores(result, label, score, min_ref_ml)
    thr, fpr, tpr, auc = _roc(y, s)
    tpr_loc = _frac_above(loc[y], thr)
    order = np.lexsort((tpr_loc, fpr))
    trapz = getattr(np, "trapezoid", None) or np.trapz
    auc_loc = float(trapz(tpr_loc[order], fpr[order])) if y.any() and (~y).any() else float("nan")
    spec = 1 - fpr
    ok = np.nonzero(spec >= target_specificity)[0]
    k = ok[np.argmin(thr[ok])] if ok.size else len(thr) - 1
    if score == "lesion":
        _, ref, _, _ = _lesion_scores(result, label)
        rs = ref["score"].to_numpy(float)
        les_sens = float((rs > thr[k]).mean()) if rs.size else float("nan")
    else:
        w = result.wide()
        w = w[w["label"] == label]
        les_sens = float(w["tp_ref_lesions"].sum() / w["n_ref_lesions"].sum()) if "n_ref_lesions" in w and \
            w["n_ref_lesions"].sum() else float("nan")
    return {"auc": auc, "auc_localized": auc_loc, "threshold": float(thr[k]), "specificity": float(spec[k]),
            "sensitivity": float(tpr[k]), "sensitivity_localized": float(tpr_loc[k]),
            "lesion_sensitivity": les_sens, "n_pos": int(y.sum()), "n_neg": int((~y).sum()),
            "fpr": fpr, "tpr": tpr, "tpr_localized": tpr_loc, "thresholds": thr}


def patient_pr(result, label: str, score: str = "pred_volume_ml", localized: bool = False,
               min_ref_ml: float = 0.0) -> Dict:
    """Patient-level precision-recall curve and average precision.

    A patient is called positive when its score exceeds the threshold. With
    ``localized=True`` a called positive patient only counts as a true
    positive if a predicted lesion overlaps a reference lesion (see
    `localized_presence`); mislocalized calls then count as false positives.
    """
    y, s, loc = _patient_scores(result, label, score, min_ref_ml)
    thr = np.unique(np.concatenate([s[np.isfinite(s)], [np.inf]]))[::-1]
    pos = loc if localized else s
    # thresholds are the observed scores: include each score itself (>= t) by nudging below it
    thr_eff = np.nextafter(thr, -np.inf)
    tp = _frac_above(pos[y], thr_eff) * y.sum() if y.any() else np.zeros(thr_eff.shape)
    called = _frac_above(s, thr_eff) * s.size
    n_pos = int(y.sum())
    if not n_pos:
        return {"recall": np.array([]), "precision": np.array([]), "thresholds": thr, "ap": float("nan"),
                "n_pos": 0, "n_neg": int((~y).sum()), "prevalence": 0.0}
    recall, precision = _pr_from_counts(tp, called - tp, n_pos)
    keep = called > 0
    return {"recall": recall[keep], "precision": precision[keep], "thresholds": thr[keep],
            "ap": _ap(recall[keep], precision[keep]), "n_pos": n_pos, "n_neg": int((~y).sum()),
            "prevalence": n_pos / len(y)}


def roc_band(y: np.ndarray, s: np.ndarray, n_boot: int = 500, ci: float = 0.95, seed: int = 0,
             grid: Optional[np.ndarray] = None):
    """Pointwise bootstrap band of an ROC curve (patients resampled), on a fixed FPR grid.

    Returns ``(grid, tpr_low, tpr_high, auc_ci)``.
    """
    y, s = np.asarray(y, bool), np.asarray(s, float)
    grid = np.linspace(0, 1, 101) if grid is None else grid
    rng = np.random.default_rng(seed)
    curves, aucs = [], []
    for _ in range(n_boot):
        i = rng.integers(0, y.size, y.size)
        yb, sb = y[i], s[i]
        if yb.all() or not yb.any():
            continue
        _, fpr, tpr, auc = _roc(yb, sb)
        order = np.lexsort((tpr, fpr))
        f, t = fpr[order], tpr[order]
        # upper envelope at each FPR: the step ROC takes its highest TPR at a given FPR
        curves.append(np.interp(grid, *_monotone(f, t)))
        aucs.append(auc)
    a = (1 - ci) / 2
    c = np.array(curves)
    return grid, np.quantile(c, a, axis=0), np.quantile(c, 1 - a, axis=0), \
        (float(np.quantile(aucs, a)), float(np.quantile(aucs, 1 - a)))


def _monotone(fpr: np.ndarray, tpr: np.ndarray):
    df = pd.DataFrame({"f": fpr, "t": tpr}).groupby("f")["t"].max()
    return df.index.to_numpy(float), np.maximum.accumulate(df.to_numpy(float))
