r"""Probabilistic and calibration metrics (need a soft prediction).

A thresholded mask throws away the model's confidence. When a probability map
\(p\in[0,1]^N\) is available these metrics ask two different questions:

* **Discrimination** (AUROC, AUPRC): does the model rank foreground voxels
  above background voxels?
* **Calibration** (ECE, Brier, NLL): when the model says 80 %, is it right
  80 % of the time? This matters whenever probabilities are shown to a
  clinician or propagated into downstream decisions.

Because 3D volumes are overwhelmingly background, whole-volume calibration is
dominated by easy background voxels. Every metric therefore takes
``roi="all"`` (the whole image, the literal definition) or ``roi="band"`` (the
union of reference and predicted foreground grown by ``margin_mm``; the region
where the model is actually uncertain), following Mehrtash et al. 2020.

Rank-based metrics use a 1024-bin histogram of the scores, which is exact up to
the bin width and runs in O(N) on 10^8 voxels.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy import ndimage

from .base import register_metric
from .context import PairContext

__all__ = ["reliability_curve"]

_EPS = 1e-7


def _roi(ctx: PairContext, roi: str, margin_mm: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return flattened ``(p, y)`` inside the requested region."""
    if ctx.prob is None:
        raise ValueError("this metric needs a probability map: pass prob=... to PairContext/evaluate")
    p, y = ctx.prob, ctx.ref
    if roi == "all":
        return p.ravel(), y.ravel()
    if roi != "band":
        raise ValueError("roi must be 'all' or 'band'")

    def build():
        fg = y | (p >= 0.5)
        if not fg.any():
            return np.zeros_like(fg)
        dist = ndimage.distance_transform_edt(~fg, sampling=ctx.spacing)
        return dist <= margin_mm

    region = ctx.memo(("roi_band", float(margin_mm)), build)
    return p[region], y[region]


_ROI = {"roi": "all", "margin_mm": 10.0}


@register_metric(
    "soft_dice", display="Soft Dice", abbr="sDSC", family="calibration", better="higher",
    requires=("probabilities",),
    summary="Dice computed with probabilities instead of a thresholded mask; rewards confident correct voxels.",
    reference="Milletari et al. 2016, 3DV (V-Net)",
)
def soft_dice(ctx: PairContext) -> float:
    r"""$$\mathrm{sDSC} = \frac{2\sum_i p_i g_i}{\sum_i p_i + \sum_i g_i}$$"""
    p = ctx.prob
    if p is None:
        raise ValueError("soft_dice needs a probability map")
    y = ctx.ref
    den = float(p.sum(dtype=np.float64) + y.sum())
    if den == 0:
        return ctx.best_or_nan(1.0)
    return 2.0 * float(p[y].sum(dtype=np.float64)) / den


def _hist(p: np.ndarray, y: np.ndarray, bins: int = 1024):
    idx = np.minimum((p * bins).astype(np.int64), bins - 1)
    pos = np.bincount(idx[y], minlength=bins).astype(np.float64)
    neg = np.bincount(idx[~y], minlength=bins).astype(np.float64)
    return pos, neg


@register_metric(
    "auroc", display="Area under the ROC curve", abbr="AUROC", family="calibration", better="higher",
    requires=("probabilities",),
    summary="Probability that a random foreground voxel scores higher than a random background voxel.",
    reference="Hanley & McNeil 1982, Radiology 143(1)",
    defaults=_ROI,
)
def auroc(ctx: PairContext, roi: str = "all", margin_mm: float = 10.0) -> float:
    r"""$$\mathrm{AUROC} = \int_0^1 \mathrm{TPR}\; d\,\mathrm{FPR}
    = P\big(p_i > p_j \mid g_i = 1, g_j = 0\big)$$"""
    p, y = _roi(ctx, roi, margin_mm)
    pos, neg = _hist(p, y)
    n_pos, n_neg = pos.sum(), neg.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Mann-Whitney U with ties inside a bin counted as one half.
    neg_below = np.cumsum(neg) - neg
    return float((pos * (neg_below + 0.5 * neg)).sum() / (n_pos * n_neg))


@register_metric(
    "auprc", display="Area under the precision-recall curve (average precision)", abbr="AP",
    family="calibration", better="higher", requires=("probabilities",),
    summary="Average precision over all thresholds; unlike AUROC it stays informative under heavy class imbalance.",
    reference="Saito & Rehmsmeier 2015, PLoS ONE 10(3)",
    defaults=_ROI,
)
def auprc(ctx: PairContext, roi: str = "all", margin_mm: float = 10.0) -> float:
    r"""$$\mathrm{AP} = \sum_k (R_k - R_{k-1})\,P_k$$ over descending thresholds."""
    p, y = _roi(ctx, roi, margin_mm)
    pos, neg = _hist(p, y)
    n_pos = pos.sum()
    if n_pos == 0:
        return float("nan")
    tp = np.cumsum(pos[::-1])
    fp = np.cumsum(neg[::-1])
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / n_pos
    return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))


@register_metric(
    "brier", display="Brier score", abbr="BS", family="calibration", better="lower",
    requires=("probabilities",),
    summary="Mean squared difference between predicted probability and the true label; mixes calibration and sharpness.",
    reference="Brier 1950, Monthly Weather Review 78(1)",
    defaults=_ROI,
)
def brier(ctx: PairContext, roi: str = "all", margin_mm: float = 10.0) -> float:
    r"""$$\mathrm{BS} = \frac1N\sum_i (p_i - g_i)^2$$"""
    p, y = _roi(ctx, roi, margin_mm)
    if p.size == 0:
        return float("nan")
    return float(np.mean((p.astype(np.float64) - y) ** 2))


@register_metric(
    "nll", display="Negative log-likelihood", abbr="NLL", family="calibration", better="lower",
    value_range=(0.0, float("inf")), requires=("probabilities",),
    summary="Average surprise of the true label under the predicted probabilities; punishes confident mistakes hard.",
    reference="Guo et al. 2017, ICML (On Calibration of Modern Neural Networks)",
    defaults=_ROI,
)
def nll(ctx: PairContext, roi: str = "all", margin_mm: float = 10.0) -> float:
    r"""$$\mathrm{NLL} = -\frac1N\sum_i \big[g_i\log p_i + (1-g_i)\log(1-p_i)\big]$$"""
    p, y = _roi(ctx, roi, margin_mm)
    if p.size == 0:
        return float("nan")
    p = np.clip(p.astype(np.float64), _EPS, 1 - _EPS)
    return float(-np.mean(np.where(y, np.log(p), np.log1p(-p))))


def reliability_curve(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> Dict[str, np.ndarray]:
    """Top-label reliability diagram data.

    Returns a dict with per-bin ``confidence`` (mean), ``accuracy``, ``count``
    and the bin ``edges``. Confidence is \\(\\max(p, 1-p)\\) and the
    predicted label is \\(p\\ge 0.5\\) (Guo et al. 2017).
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=bool).ravel()
    conf = np.maximum(p, 1.0 - p)
    correct = (p >= 0.5) == y
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    count = np.bincount(idx, minlength=n_bins).astype(np.float64)
    s_conf = np.bincount(idx, weights=conf, minlength=n_bins)
    s_acc = np.bincount(idx, weights=correct, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"edges": edges, "count": count,
                "confidence": s_conf / count, "accuracy": s_acc / count}


@register_metric(
    "ece", display="Expected calibration error", abbr="ECE", family="calibration", better="lower",
    requires=("probabilities",),
    summary="Average gap between confidence and accuracy across confidence bins; 0 means perfectly calibrated.",
    reference="Naeini et al. 2015, AAAI; Guo et al. 2017, ICML; Mehrtash et al. 2020, IEEE TMI 39(12)",
    defaults={**_ROI, "n_bins": 15},
)
def ece(ctx: PairContext, roi: str = "all", margin_mm: float = 10.0, n_bins: int = 15) -> float:
    r"""$$\mathrm{ECE} = \sum_{b=1}^{B}\frac{|B_b|}{N}\,\big|\mathrm{acc}(B_b) - \mathrm{conf}(B_b)\big|$$"""
    p, y = _roi(ctx, roi, margin_mm)
    if p.size == 0:
        return float("nan")
    c = reliability_curve(p, y, n_bins)
    w = c["count"] / c["count"].sum()
    gap = np.abs(np.nan_to_num(c["accuracy"]) - np.nan_to_num(c["confidence"]))
    return float(np.sum(w * gap))
