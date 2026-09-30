r"""Agreement and information-theoretic metrics.

These treat prediction and reference as two *partitions* of the image into
foreground and background and compare the partitions as a whole, background
included. They come from clustering and inter-rater-agreement literature and
are reported mostly for completeness and comparability with Taha & Hanbury
(2015); for segmentation quality they are strongly correlated with Dice but
are affected by the size of the background.

With :math:`N` voxels and the 2x2 contingency table
:math:`n_{11}=TP, n_{10}=FN, n_{01}=FP, n_{00}=TN` (rows: reference).
"""

from __future__ import annotations

import math

from .base import register_metric
from .context import PairContext


def _entropy(*counts: float) -> float:
    n = sum(counts)
    return -sum((c / n) * math.log(c / n) for c in counts if c > 0)


def _mi(ctx: PairContext) -> float:
    tp, fp, fn, tn = (float(x) for x in ctx.counts)
    hg = _entropy(tp + fn, fp + tn)
    hp = _entropy(tp + fp, fn + tn)
    hj = _entropy(tp, fp, fn, tn)
    return hg + hp - hj, hg, hp


@register_metric(
    "mutual_information", display="Mutual information", abbr="MI", family="agreement", better="higher",
    value_range=(0.0, math.log(2)), unit="nats",
    summary="How much knowing the prediction reduces uncertainty about the reference label of a voxel.",
    reference="Russakoff et al. 2004, ECCV; Taha & Hanbury 2015",
)
def mutual_information(ctx: PairContext) -> float:
    r"""$$\mathrm{MI}(P,G) = H(P) + H(G) - H(P,G)$$"""
    return float(_mi(ctx)[0])


@register_metric(
    "variation_of_information", display="Variation of information", abbr="VI", family="agreement",
    better="lower", value_range=(0.0, 2 * math.log(2)), unit="nats",
    summary="Information lost and gained between the two partitions; a true metric on partitions.",
    reference="Meilă 2007, J Multivariate Anal 98(5); Taha & Hanbury 2015",
)
def variation_of_information(ctx: PairContext) -> float:
    r"""$$\mathrm{VI}(P,G) = H(P) + H(G) - 2\,\mathrm{MI}(P,G)$$"""
    mi, hg, hp = _mi(ctx)
    return float(max(hg + hp - 2 * mi, 0.0))


@register_metric(
    "adjusted_rand_index", display="Adjusted Rand index", abbr="ARI", family="agreement", better="higher",
    value_range=(-1.0, 1.0),
    summary="Pairwise agreement of voxel labels, corrected for chance.",
    reference="Hubert & Arabie 1985, J Classification 2; Taha & Hanbury 2015",
)
def adjusted_rand_index(ctx: PairContext) -> float:
    r"""$$\mathrm{ARI} = \frac{\sum_{ij}\binom{n_{ij}}{2} - \big[\sum_i\binom{a_i}{2}\sum_j\binom{b_j}{2}\big]/\binom{N}{2}}
    {\tfrac12\big[\sum_i\binom{a_i}{2}+\sum_j\binom{b_j}{2}\big] - \big[\sum_i\binom{a_i}{2}\sum_j\binom{b_j}{2}\big]/\binom{N}{2}}$$"""
    tp, fp, fn, tn = (float(x) for x in ctx.counts)
    c2 = lambda x: x * (x - 1) / 2.0  # noqa: E731
    n = tp + fp + fn + tn
    index = c2(tp) + c2(fp) + c2(fn) + c2(tn)
    sa = c2(tp + fn) + c2(fp + tn)
    sb = c2(tp + fp) + c2(fn + tn)
    expected = sa * sb / c2(n)
    max_index = 0.5 * (sa + sb)
    if max_index == expected:
        return ctx.best_or_nan(1.0)
    return float((index - expected) / (max_index - expected))


@register_metric(
    "global_consistency_error", display="Global consistency error", abbr="GCE", family="agreement",
    better="lower",
    summary="Degree to which one segmentation is a refinement of the other; 0 when identical.",
    reference="Martin et al. 2001, ICCV; Taha & Hanbury 2015",
)
def global_consistency_error(ctx: PairContext) -> float:
    r"""$$\mathrm{GCE} = \frac1N\min\Big\{\tfrac{FN(FN+2TP)}{TP+FN} + \tfrac{FP(FP+2TN)}{TN+FP},\;
    \tfrac{FP(FP+2TP)}{TP+FP} + \tfrac{FN(FN+2TN)}{TN+FN}\Big\}$$"""
    tp, fp, fn, tn = (float(x) for x in ctx.counts)
    n = tp + fp + fn + tn
    d = lambda a, b: a / b if b else 0.0  # noqa: E731
    e1 = d(fn * (fn + 2 * tp), tp + fn) + d(fp * (fp + 2 * tn), tn + fp)
    e2 = d(fp * (fp + 2 * tp), tp + fp) + d(fn * (fn + 2 * tn), tn + fn)
    return float(min(e1, e2) / n)
