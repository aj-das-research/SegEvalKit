"""Dataset- and cohort-level analysis of an evaluation.

A benchmark is evaluated on a whole test set, and the numbers worth reporting
live at several levels. SegEvalKit keeps all of them:

=====================  =======================================================  ====================================
Level                  Question                                                 Where
=====================  =======================================================  ====================================
Instance               Was this lesion found, and how well?                     ``result.lesions``
Sample (case)          How good is this patient's segmentation?                 ``result.wide()`` / ``per_case.csv``
Dataset, macro         What is the typical per-case score?                      ``result.summary()``
Dataset, micro         What is the score of all voxels / lesions pooled?        :func:`pooled_metrics`
Cohort (subgroup)      Does performance differ by site, phase, sex, size ...?   :func:`cohort_summary`
Patient (presence)     Does this patient have the structure (tumour) at all?    :func:`segevalkit.stats.presence_detection`
=====================  =======================================================  ====================================

Macro and micro averages answer different questions. The macro mean weighs
every *patient* equally; the pooled (micro) value weighs every *voxel* or
*lesion* equally, so large structures and lesion-rich patients dominate it.
Challenges differ here: HECKTOR 2022 ranks by aggregated Dice, most others by
the mean per-case Dice. Report the one that matches your question, and say which.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats as st

from .stats import adjust_pvalues, bootstrap_ci

__all__ = ["pooled_metrics", "attach_metadata", "cohort_summary", "cohort_tests"]


def _counts(result) -> pd.DataFrame:
    w = result.wide()
    need = {"tp", "fp", "fn"}
    if not need <= set(w.columns):
        raise ValueError("this result has no raw counts; re-run the evaluation with SegEvalKit >= 0.1.1")
    return w


def pooled_metrics(result, labels: Optional[Sequence[str]] = None, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Dataset-level pooled ("micro") metrics per structure, with case-bootstrap CIs.

    Voxel level (from the summed confusion counts over all cases):

    * aggregated Dice :math:`2\\sum TP / (2\\sum TP + \\sum FP + \\sum FN)` (HECKTOR 2022),
    * pooled IoU, precision and recall.

    Lesion level (when detection metrics were computed):

    * pooled lesion sensitivity :math:`\\sum TP_{ref} / \\sum N_{ref}` over all reference lesions,
    * pooled lesion precision :math:`\\sum TP_{pred} / \\sum N_{pred}`,
    * pooled lesion F1 and false-positive lesions per scan.

    CIs resample *cases* (the independent unit), not voxels or lesions.
    """
    w = _counts(result)
    if labels is not None:
        w = w[w["label"].isin(labels)]
    rng = np.random.default_rng(seed)
    rows = []

    def voxel(d):
        tp, fp, fn = d["tp"].sum(), d["fp"].sum(), d["fn"].sum()
        return {
            "pooled_dice": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan,
            "pooled_iou": tp / (tp + fp + fn) if (tp + fp + fn) else np.nan,
            "pooled_precision": tp / (tp + fp) if (tp + fp) else np.nan,
            "pooled_recall": tp / (tp + fn) if (tp + fn) else np.nan,
        }

    def lesion(d):
        if "n_ref_lesions" not in d or d["n_ref_lesions"].isna().all():
            return {}
        nr, npred = d["n_ref_lesions"].sum(), d["n_pred_lesions"].sum()
        tr, tpp = d["tp_ref_lesions"].sum(), d["tp_pred_lesions"].sum()
        fp = npred - tpp
        fn = nr - tr
        return {
            "pooled_lesion_sensitivity": tr / nr if nr else np.nan,
            "pooled_lesion_precision": tpp / npred if npred else np.nan,
            "pooled_lesion_f1": 2 * tr / (2 * tr + fp + fn) if (2 * tr + fp + fn) else np.nan,
            "fp_lesions_per_scan": fp / len(d) if len(d) else np.nan,
            "n_ref_lesions": nr,
        }

    for label, d in w.groupby("label", sort=False):
        d = d.dropna(subset=["tp", "fp", "fn"])
        point = {**voxel(d), **lesion(d)}
        idx = np.arange(len(d))
        boots = []
        for _ in range(n_boot if len(d) > 1 else 0):
            b = d.iloc[rng.choice(idx, size=len(idx), replace=True)]
            boots.append({**voxel(b), **lesion(b)})
        bdf = pd.DataFrame(boots)
        for k, v in point.items():
            if k == "n_ref_lesions":
                continue
            lo, hi = (np.nanquantile(bdf[k], [0.025, 0.975]) if len(bdf) and k in bdf else (np.nan, np.nan))
            rows.append({"label": label, "metric": k, "n_cases": len(d), "value": float(v),
                         "ci_low": float(lo), "ci_high": float(hi)})
    return pd.DataFrame(rows)


def attach_metadata(result, metadata: Union[pd.DataFrame, str], id_column: Optional[str] = None,
                    columns: Optional[Sequence[str]] = None, normalise: bool = True) -> pd.DataFrame:
    """Join case metadata (site, scanner, phase, sex, age, ...) onto the per-case table.

    Args:
        metadata: DataFrame or a CSV / Excel path, one row per case.
        id_column: Column holding the case id (default: the first column).
        columns: Metadata columns to keep (default: all).
        normalise: Strip whitespace and unify case of string categories
            ("M " and "M" become one group) so cohorts are not split by typos.

    Returns:
        ``result.wide()`` with the metadata columns joined (cases without
        metadata keep NaN and are reported as "(missing)" in cohort summaries).
    """
    if isinstance(metadata, str):
        metadata = pd.read_excel(metadata) if metadata.endswith((".xlsx", ".xls")) else pd.read_csv(metadata)
    md = metadata.copy()
    id_column = id_column or md.columns[0]
    md = md.rename(columns={id_column: "case_id"})
    md["case_id"] = md["case_id"].astype(str).str.strip()
    if columns is not None:
        md = md[["case_id", *columns]]
    if normalise:
        for c in md.columns:
            if c != "case_id" and md[c].dtype == object:
                md[c] = md[c].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})
    return result.wide().merge(md, on="case_id", how="left")


def _group_series(df: pd.DataFrame, by: str, bins: Optional[Sequence[float]]) -> pd.Series:
    g = df[by]
    if bins is not None:
        g = pd.cut(pd.to_numeric(g, errors="coerce"), bins=list(bins), include_lowest=True).astype(str)
    return g.fillna("(missing)").astype(str)


def cohort_summary(result, metadata: Union[pd.DataFrame, str], by: Union[str, Sequence[str]],
                   metrics: Iterable[str] = ("dice", "nsd", "hd95"), labels: Optional[Sequence[str]] = None,
                   id_column: Optional[str] = None, bins: Optional[Mapping[str, Sequence[float]]] = None,
                   min_n: int = 5, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Per-subgroup statistics of each metric (macro over the cases in the subgroup).

    Args:
        by: One or more metadata columns defining the subgroups (e.g.
            ``"ct phase"``, ``["site nationality", "sex"]``). Several columns
            are analysed one at a time, not crossed.
        bins: Optional bin edges for numeric columns, e.g. ``{"age": [0, 40, 60, 80, 120]}``.
        min_n: Subgroups with fewer cases are reported but flagged ``small``.

    Returns:
        One row per (factor, group, label, metric): n, mean, median, IQR,
        bootstrap CI of the mean, and the ``small`` flag.
    """
    df = attach_metadata(result, metadata, id_column)
    if labels is not None:
        df = df[df["label"].isin(labels)]
    factors = [by] if isinstance(by, str) else list(by)
    bins = dict(bins or {})
    rows = []
    for f in factors:
        grp = _group_series(df, f, bins.get(f))
        for (label, g), d in df.assign(_g=grp).groupby(["label", "_g"], sort=True):
            for m in metrics:
                if m not in d:
                    continue
                v = d[m].to_numpy(float)
                v = v[~np.isnan(v)]
                lo, hi = bootstrap_ci(v, n_boot=n_boot, seed=seed)
                rows.append({"factor": f, "group": g, "label": label, "metric": m, "n": int(v.size),
                             "mean": float(v.mean()) if v.size else np.nan,
                             "median": float(np.median(v)) if v.size else np.nan,
                             "q1": float(np.quantile(v, .25)) if v.size else np.nan,
                             "q3": float(np.quantile(v, .75)) if v.size else np.nan,
                             "ci_low": lo, "ci_high": hi, "small": bool(v.size < min_n)})
    return pd.DataFrame(rows)


def cohort_tests(result, metadata: Union[pd.DataFrame, str], by: Union[str, Sequence[str]],
                 metrics: Iterable[str] = ("dice", "nsd", "hd95"), labels: Optional[Sequence[str]] = None,
                 id_column: Optional[str] = None, bins: Optional[Mapping[str, Sequence[float]]] = None,
                 min_n: int = 5, correction: str = "holm") -> pd.DataFrame:
    """Do subgroups differ? Kruskal–Wallis (≥ 3 groups) or Mann–Whitney U (2 groups) per factor.

    Subgroups smaller than ``min_n`` and the "(missing)" group are excluded from
    the test. p-values are corrected across all (factor, label, metric) tests.
    An effect size is reported as epsilon² (Kruskal–Wallis) or the rank-biserial
    correlation (Mann–Whitney).
    """
    df = attach_metadata(result, metadata, id_column)
    if labels is not None:
        df = df[df["label"].isin(labels)]
    factors = [by] if isinstance(by, str) else list(by)
    bins = dict(bins or {})
    rows = []
    for f in factors:
        grp = _group_series(df, f, bins.get(f))
        d0 = df.assign(_g=grp)
        for label, dl in d0.groupby("label", sort=False):
            for m in metrics:
                if m not in dl:
                    continue
                samples = {g: x[m].dropna().to_numpy(float) for g, x in dl.groupby("_g")
                           if g != "(missing)" and x[m].notna().sum() >= min_n}
                if len(samples) < 2:
                    continue
                vals = list(samples.values())
                n = sum(len(v) for v in vals)
                if len(vals) == 2:
                    res = st.mannwhitneyu(vals[0], vals[1], alternative="two-sided")
                    eff = 1 - 2 * res.statistic / (len(vals[0]) * len(vals[1]))
                    test, p = "mann-whitney", float(res.pvalue)
                else:
                    res = st.kruskal(*vals)
                    eff = float((res.statistic - len(vals) + 1) / (n - len(vals)))
                    test, p = "kruskal-wallis", float(res.pvalue)
                rows.append({"factor": f, "label": label, "metric": m, "groups": len(vals), "n": n,
                             "test": test, "effect": float(eff), "p_value": p})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_adjusted"] = adjust_pvalues(out["p_value"], correction)
        out["significant"] = out["p_adjusted"] < 0.05
    return out
