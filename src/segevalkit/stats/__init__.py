"""Statistics for segmentation evaluation: aggregation, comparison, ranking, agreement.

Per-case metric values are the unit of analysis: a mean Dice without its
spread, its confidence interval and a paired test against the competitor says
very little (Maier-Hein et al. 2018, *Nat Commun* 9:5217; Wiesenfarth et al.
2021, *Sci Rep* 11:2369). This module provides:

* :func:`summarize` / :func:`bootstrap_ci`: descriptive statistics with
  percentile-bootstrap confidence intervals.
* :func:`compare`: paired tests between two methods (Wilcoxon signed-rank,
  paired t, sign-flip permutation) with Holm or Benjamini-Hochberg correction
  across labels and metrics, plus effect sizes.
* :func:`rank_methods` and :func:`ranking_stability`: challenge-style ranking
  (aggregate-then-rank, rank-then-aggregate) and its robustness under
  bootstrap resampling of cases (Kendall's tau to the full-data ranking).
* :func:`bland_altman` and :func:`icc`: volumetric agreement.
* :func:`stratify`: summaries within bins of a covariate such as structure size.
* :func:`presence_detection`: case-level ("does this patient have a tumour?")
  sensitivity / specificity / ROC-AUC from predicted volume, the PanTS
  patient-wise protocol.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as st

_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy >= 2 renamed trapz

__all__ = [
    "bootstrap_ci",
    "summarize",
    "compare",
    "adjust_pvalues",
    "rank_methods",
    "ranking_stability",
    "bland_altman",
    "icc",
    "stratify",
    "presence_detection",
]


def bootstrap_ci(x: Sequence[float], stat=np.mean, ci: float = 0.95, n_boot: int = 2000,
                 seed: int = 0) -> tuple:
    """Percentile bootstrap confidence interval of ``stat`` (NaNs dropped)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return (np.nan, np.nan)
    if x.size == 1:
        return (float(x[0]), float(x[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boots = stat(x[idx], axis=1)
    a = (1 - ci) / 2
    return float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))


def summarize(df: pd.DataFrame, by: Sequence[str] = ("label", "metric"), ci: float = 0.95,
              n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Descriptive statistics of a long table (``..., metric, value``).

    Returns columns ``n, n_nan, mean, std, median, q1, q3, min, max,
    ci_low, ci_high`` for each group in ``by``.
    """
    out = []
    for key, g in df.groupby(list(by), sort=False):
        v = g["value"].to_numpy(dtype=float)
        ok = v[~np.isnan(v)]
        lo, hi = bootstrap_ci(ok, ci=ci, n_boot=n_boot, seed=seed)
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        row.update(
            n=int(ok.size), n_nan=int(v.size - ok.size),
            mean=float(ok.mean()) if ok.size else np.nan,
            std=float(ok.std(ddof=1)) if ok.size > 1 else np.nan,
            median=float(np.median(ok)) if ok.size else np.nan,
            q1=float(np.quantile(ok, 0.25)) if ok.size else np.nan,
            q3=float(np.quantile(ok, 0.75)) if ok.size else np.nan,
            min=float(ok.min()) if ok.size else np.nan,
            max=float(ok.max()) if ok.size else np.nan,
            ci_low=lo, ci_high=hi,
        )
        out.append(row)
    return pd.DataFrame(out)


def adjust_pvalues(p: Sequence[float], method: str = "holm") -> np.ndarray:
    """Multiple-comparison correction: ``"holm"``, ``"bh"`` (Benjamini-Hochberg) or ``"none"``."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    q = p[ok]
    m = q.size
    if m == 0 or method == "none":
        return p.copy()
    order = np.argsort(q)
    if method == "holm":
        adj = np.maximum.accumulate((m - np.arange(m)) * q[order])
    elif method == "bh":
        adj = np.minimum.accumulate((m / np.arange(m, 0, -1)) * q[order][::-1])[::-1]
    else:
        raise ValueError("method must be 'holm', 'bh' or 'none'")
    res = np.empty(m)
    res[order] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


def _paired(a: np.ndarray, b: np.ndarray, test: str, n_perm: int, seed: int) -> float:
    d = a - b
    if np.allclose(d, 0):
        return 1.0
    if test == "wilcoxon":
        return float(st.wilcoxon(a, b, zero_method="zsplit").pvalue)
    if test == "ttest":
        return float(st.ttest_rel(a, b).pvalue)
    if test == "permutation":
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1.0, 1.0], size=(n_perm, d.size))
        null = np.abs((signs * d).mean(axis=1))
        return float((1 + np.sum(null >= abs(d.mean()))) / (n_perm + 1))
    raise ValueError("test must be 'wilcoxon', 'ttest' or 'permutation'")


def compare(a, b, *, metrics: Optional[Sequence[str]] = None, labels: Optional[Sequence[str]] = None,
            test: str = "wilcoxon", correction: str = "holm", n_perm: int = 10000,
            seed: int = 0) -> pd.DataFrame:
    """Paired comparison of two evaluation results on their common cases.

    Args:
        a, b: :class:`~segevalkit.results.EvaluationResult` objects (or long tables).
        test: ``"wilcoxon"`` (default; no normality assumption), ``"ttest"`` or
            ``"permutation"`` (sign-flip test of the mean difference).
        correction: Correction across all (label, metric) tests.

    Returns:
        One row per (label, metric): means, mean difference ``a − b`` with a
        bootstrap CI, fraction of cases where *a* is better, the matched-pairs
        rank-biserial effect size, raw and adjusted p-values.
    """
    from ..metrics import get_metric

    la = a.per_case() if hasattr(a, "per_case") else a
    lb = b.per_case() if hasattr(b, "per_case") else b
    m = la.merge(lb, on=["case_id", "label", "metric"], suffixes=("_a", "_b"))
    if metrics is not None:
        m = m[m["metric"].isin(metrics)]
    if labels is not None:
        m = m[m["label"].isin(labels)]
    rows = []
    for (label, metric), g in m.groupby(["label", "metric"], sort=False):
        g = g.dropna(subset=["value_a", "value_b"])
        if len(g) < 2:
            continue
        va, vb = g["value_a"].to_numpy(float), g["value_b"].to_numpy(float)
        d = va - vb
        info = get_metric(metric)
        sign = {"higher": 1.0, "lower": -1.0}.get(info.better)
        if sign is None:  # signed metric: compare magnitudes
            d_better = np.abs(vb) - np.abs(va)
        else:
            d_better = sign * d
        nz = d_better[d_better != 0]
        rb = np.nan
        if nz.size:
            r = st.rankdata(np.abs(nz))
            rb = float((r[nz > 0].sum() - r[nz < 0].sum()) / r.sum())
        lo, hi = bootstrap_ci(d, seed=seed)
        rows.append({
            "label": label, "metric": metric, "n": len(g),
            "mean_a": va.mean(), "mean_b": vb.mean(), "mean_diff": d.mean(),
            "diff_ci_low": lo, "diff_ci_high": hi,
            "frac_a_better": float(np.mean(d_better > 0)),
            "effect_rank_biserial": rb,
            "p_value": _paired(va, vb, test, n_perm, seed),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["p_adjusted"] = adjust_pvalues(out["p_value"], correction)
        out["significant"] = out["p_adjusted"] < 0.05
    return out


def _direction(metric: str) -> float:
    from ..metrics import get_metric

    return {"higher": -1.0, "lower": 1.0}.get(get_metric(metric).better, 1.0)


def rank_methods(results: Mapping[str, object], metric: str, label: Optional[str] = None,
                 scheme: str = "aggregate-then-rank", agg: str = "mean") -> pd.DataFrame:
    """Rank methods on one metric.

    Args:
        results: ``{method_name: EvaluationResult}``.
        scheme: ``"aggregate-then-rank"`` (rank the per-method mean/median) or
            ``"rank-then-aggregate"`` (rank methods within every case, then
            average ranks; robust to a few catastrophic cases). Both are used
            by major challenges and can disagree (Maier-Hein et al. 2018).
        agg: ``"mean"`` or ``"median"``.
    """
    tables = []
    for name, r in results.items():
        d = r.per_case() if hasattr(r, "per_case") else r
        d = d[d["metric"] == metric]
        if label is not None:
            d = d[d["label"] == label]
        tables.append(d.assign(method=name))
    d = pd.concat(tables)
    sgn = _direction(metric)
    key = ["case_id", "label"]
    if scheme == "aggregate-then-rank":
        s = d.groupby("method")["value"].agg(agg)
        score = s * sgn if _is_monotone(metric) else s.abs()
        rk = score.rank(method="min")
        return pd.DataFrame({"method": s.index, agg: s.values, "rank": rk.values}).sort_values("rank").reset_index(drop=True)
    if scheme == "rank-then-aggregate":
        piv = d.pivot_table(index=key, columns="method", values="value", aggfunc="first")
        vals = piv * sgn if _is_monotone(metric) else piv.abs()
        # Missing values rank last (a method that failed a case must not gain from it).
        ranks = vals.fillna(np.inf).rank(axis=1, method="min")
        mean_rank = ranks.mean(axis=0)
        return (pd.DataFrame({"method": mean_rank.index, "mean_rank": mean_rank.values,
                              "rank": mean_rank.rank(method="min").values})
                .sort_values("rank").reset_index(drop=True))
    raise ValueError("scheme must be 'aggregate-then-rank' or 'rank-then-aggregate'")


def _is_monotone(metric: str) -> bool:
    from ..metrics import get_metric

    return get_metric(metric).better in ("higher", "lower")


def ranking_stability(results: Mapping[str, object], metric: str, label: Optional[str] = None,
                      scheme: str = "aggregate-then-rank", n_boot: int = 1000, seed: int = 0) -> Dict:
    """Bootstrap the case set and re-rank (Wiesenfarth et al. 2021).

    Returns a dict with ``ranks`` (DataFrame ``n_boot x methods``), the
    full-data ranking ``reference`` and ``kendall_tau`` per bootstrap sample.
    """
    rng = np.random.default_rng(seed)
    full = rank_methods(results, metric, label, scheme).set_index("method")["rank"]
    methods = list(full.index)
    per = {}
    for name, r in results.items():
        d = r.per_case() if hasattr(r, "per_case") else r
        d = d[d["metric"] == metric]
        if label is not None:
            d = d[d["label"] == label]
        per[name] = d
    cases = sorted(set.intersection(*[set(v["case_id"]) for v in per.values()]))
    all_ranks, taus = [], []
    for _ in range(n_boot):
        pick = rng.choice(cases, size=len(cases), replace=True)
        # Re-key duplicated draws so rank-then-aggregate counts each draw.
        sub = {}
        for n, d in per.items():
            s = pd.DataFrame({"case_id": pick})
            s["draw"] = np.arange(len(pick))
            j = s.merge(d, on="case_id")
            j["case_id"] = j["case_id"].astype(str) + "#" + j["draw"].astype(str)
            sub[n] = j.drop(columns="draw")
        rk = rank_methods(sub, metric, label, scheme).set_index("method")["rank"].reindex(methods)
        all_ranks.append(rk.to_numpy())
        taus.append(st.kendalltau(full.to_numpy(), rk.to_numpy()).statistic)
    return {"reference": full, "ranks": pd.DataFrame(all_ranks, columns=methods),
            "kendall_tau": np.asarray(taus)}


def bland_altman(pred: Sequence[float], ref: Sequence[float]) -> Dict[str, float]:
    """Bland-Altman agreement: bias and 95 % limits of agreement of ``pred − ref``.

    Bland & Altman 1986, *Lancet* 327(8476).
    """
    p = np.asarray(pred, dtype=float)
    r = np.asarray(ref, dtype=float)
    ok = ~(np.isnan(p) | np.isnan(r))
    d = p[ok] - r[ok]
    bias = float(d.mean())
    sd = float(d.std(ddof=1)) if d.size > 1 else np.nan
    return {"n": int(d.size), "bias": bias, "sd": sd,
            "loa_low": bias - 1.96 * sd, "loa_high": bias + 1.96 * sd,
            "mean": ((p[ok] + r[ok]) / 2).tolist(), "diff": d.tolist()}


def icc(pred: Sequence[float], ref: Sequence[float], kind: str = "ICC(2,1)") -> float:
    """Intraclass correlation between two raters (prediction and reference).

    ``"ICC(2,1)"``: two-way random effects, absolute agreement, single rater;
    ``"ICC(3,1)"``: two-way mixed, consistency (Shrout & Fleiss 1979;
    Koo & Li 2016 for interpretation).
    """
    x = np.column_stack([np.asarray(pred, float), np.asarray(ref, float)])
    x = x[~np.isnan(x).any(axis=1)]
    n, k = x.shape
    if n < 2:
        return float("nan")
    gm = x.mean()
    msr = k * ((x.mean(axis=1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((x.mean(axis=0) - gm) ** 2).sum() / (k - 1)
    sse = ((x - x.mean(axis=1, keepdims=True) - x.mean(axis=0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    if kind == "ICC(2,1)":
        return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))
    if kind == "ICC(3,1)":
        return float((msr - mse) / (msr + (k - 1) * mse))
    raise ValueError("kind must be 'ICC(2,1)' or 'ICC(3,1)'")


def stratify(result, metric: str, by: str = "ref_volume_ml", bins: Optional[Sequence[float]] = None,
             q: int = 4, label: Optional[str] = None) -> pd.DataFrame:
    """Summarise ``metric`` within bins of a per-case covariate (default: reference volume).

    Reveals size dependence, e.g. Dice collapsing on small structures
    (Reinke et al. 2024, pitfall "small structures").
    """
    w = result.wide()
    if label is not None:
        w = w[w["label"] == label]
    w = w.dropna(subset=[metric, by])
    if bins is None:
        w["bin"] = pd.qcut(w[by], q=q, duplicates="drop")
    else:
        w["bin"] = pd.cut(w[by], bins=list(bins), include_lowest=True)
    g = w.groupby("bin", observed=True)[metric]
    return pd.DataFrame({"n": g.size(), "mean": g.mean(), "median": g.median(),
                         "q1": g.quantile(0.25), "q3": g.quantile(0.75)}).reset_index()


def presence_detection(result, label: str, score: str = "pred_volume_ml", target_specificity: float = 0.9,
                       min_ref_ml: float = 0.0) -> Dict:
    """Case-level presence detection from a per-case score (default: predicted volume).

    A case is *positive* if its reference contains the structure (volume >
    ``min_ref_ml``); the model calls it positive if ``score`` exceeds a
    threshold. Sweeping the threshold gives an ROC curve; the operating point
    is the lowest threshold whose specificity reaches ``target_specificity``.
    This is the patient-wise protocol of the PanTS benchmark
    (Li et al. 2025) and the image-level view recommended by Metrics Reloaded
    whenever references can be empty.

    Returns:
        dict with ``auc``, ``threshold``, ``sensitivity``, ``specificity``,
        ``n_pos``, ``n_neg`` and the ROC arrays ``fpr``, ``tpr``, ``thresholds``.
    """
    w = result.wide()
    w = w[w["label"] == label].dropna(subset=[score, "ref_volume_ml"])
    y = (w["ref_volume_ml"] > min_ref_ml).to_numpy()
    s = w[score].to_numpy(float)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    thr = np.unique(np.concatenate([[-np.inf], s, [np.inf]]))
    tpr = np.array([(s[y] > t).mean() if n_pos else np.nan for t in thr])
    fpr = np.array([(s[~y] > t).mean() if n_neg else np.nan for t in thr])
    order = np.argsort(fpr, kind="stable")
    auc = float(_trapz(tpr[order], fpr[order])) if n_pos and n_neg else float("nan")
    spec = 1 - fpr
    ok = np.nonzero(spec >= target_specificity)[0]
    k = ok[np.argmin(thr[ok])] if ok.size else len(thr) - 1
    return {"auc": auc, "threshold": float(thr[k]), "sensitivity": float(tpr[k]), "specificity": float(spec[k]),
            "n_pos": n_pos, "n_neg": n_neg, "fpr": fpr, "tpr": tpr, "thresholds": thr}
