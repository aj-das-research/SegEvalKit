r"""Overlap metrics computed from the voxel confusion matrix.

With :math:`P` the predicted and :math:`G` the reference foreground,
:math:`TP = |P \cap G|`, :math:`FP = |P \setminus G|`,
:math:`FN = |G \setminus P|` and :math:`TN` the remaining voxels.

Metrics that ignore :math:`TN` (Dice, IoU, precision, recall...) are
*invariant to the size of the image*, which is why they are preferred for
segmentation. Metrics that use :math:`TN` (specificity, accuracy, kappa, MCC)
are dominated by the background in 3D volumes and should be read with that
in mind (Metrics Reloaded pitfall "class imbalance").
"""

from __future__ import annotations

import math

from .base import register_metric
from .context import PairContext


def _ratio(num: float, den: float, ctx: PairContext, best: float) -> float:
    # A zero denominator means the ratio is undefined. When both masks are
    # empty that is a correct "absent" call and the EmptyPolicy decides; when
    # only one is (precision with an empty prediction, recall with an empty
    # reference) the value is genuinely undefined and NaN, never a silent 0.
    if den == 0:
        return ctx.best_or_nan(best) if ctx.both_empty else float("nan")
    return num / den


@register_metric(
    "dice", display="Dice similarity coefficient", abbr="DSC", family="overlap", better="higher",
    summary="Twice the overlap divided by the total size of both masks; the most common segmentation score.",
    reference="Dice 1945, Ecology 26(3); Sørensen 1948",
)
def dice(ctx: PairContext) -> float:
    r"""$$\mathrm{DSC} = \frac{2|P\cap G|}{|P|+|G|} = \frac{2TP}{2TP+FP+FN}$$"""
    return _ratio(2 * ctx.tp, 2 * ctx.tp + ctx.fp + ctx.fn, ctx, 1.0)


@register_metric(
    "iou", display="Intersection over union (Jaccard index)", abbr="IoU", family="overlap", better="higher",
    summary="Overlap divided by the union of both masks; a stricter monotone transform of Dice.",
    reference="Jaccard 1912, New Phytologist 11(2)",
)
def iou(ctx: PairContext) -> float:
    r"""$$\mathrm{IoU} = \frac{|P\cap G|}{|P\cup G|} = \frac{TP}{TP+FP+FN} = \frac{\mathrm{DSC}}{2-\mathrm{DSC}}$$"""
    return _ratio(ctx.tp, ctx.tp + ctx.fp + ctx.fn, ctx, 1.0)


@register_metric(
    "voe", display="Volumetric overlap error", abbr="VOE", family="overlap", better="lower",
    summary="One minus IoU: the fraction of the union that is not shared (used in LiTS / SLIVER07).",
    reference="Heimann et al. 2009, IEEE TMI 28(8)",
)
def voe(ctx: PairContext) -> float:
    r"""$$\mathrm{VOE} = 1 - \mathrm{IoU}$$"""
    v = iou(ctx)
    return 1.0 - v if not math.isnan(v) else v


@register_metric(
    "precision", display="Precision (positive predictive value)", abbr="PPV", family="overlap", better="higher",
    summary="Fraction of predicted voxels that are truly foreground; penalises over-segmentation.",
    reference="Taha & Hanbury 2015, BMC Med Imaging 15:29",
)
def precision(ctx: PairContext) -> float:
    r"""$$\mathrm{PPV} = \frac{TP}{TP+FP}$$"""
    return _ratio(ctx.tp, ctx.tp + ctx.fp, ctx, 1.0)


@register_metric(
    "recall", display="Recall (sensitivity, true-positive rate)", abbr="TPR", family="overlap", better="higher",
    summary="Fraction of the reference that was found; penalises under-segmentation.",
    reference="Taha & Hanbury 2015, BMC Med Imaging 15:29",
)
def recall(ctx: PairContext) -> float:
    r"""$$\mathrm{TPR} = \frac{TP}{TP+FN}$$"""
    return _ratio(ctx.tp, ctx.tp + ctx.fn, ctx, 1.0)


@register_metric(
    "specificity", display="Specificity (true-negative rate)", abbr="TNR", family="overlap", better="higher",
    summary="Fraction of background left as background; near 1 for almost any 3D prediction because background dominates.",
    reference="Taha & Hanbury 2015, BMC Med Imaging 15:29",
)
def specificity(ctx: PairContext) -> float:
    r"""$$\mathrm{TNR} = \frac{TN}{TN+FP}$$"""
    den = ctx.tn + ctx.fp
    return ctx.tn / den if den else float("nan")


@register_metric(
    "fpr", display="False-positive rate (fall-out)", abbr="FPR", family="overlap", better="lower",
    summary="Fraction of background wrongly labelled foreground (1 − specificity).",
    reference="Taha & Hanbury 2015, BMC Med Imaging 15:29",
)
def fpr(ctx: PairContext) -> float:
    r"""$$\mathrm{FPR} = \frac{FP}{FP+TN}$$"""
    den = ctx.fp + ctx.tn
    return ctx.fp / den if den else float("nan")


@register_metric(
    "fnr", display="False-negative rate (miss rate)", abbr="FNR", family="overlap", better="lower",
    summary="Fraction of the reference that was missed (1 − recall).",
    reference="Taha & Hanbury 2015, BMC Med Imaging 15:29",
)
def fnr(ctx: PairContext) -> float:
    r"""$$\mathrm{FNR} = \frac{FN}{FN+TP}$$"""
    v = recall(ctx)
    return 1.0 - v if not math.isnan(v) else v


@register_metric(
    "fbeta", display="F-beta score", abbr="Fβ", family="overlap", better="higher",
    summary="Weighted harmonic mean of precision and recall; beta > 1 favours recall (missing tissue is worse).",
    reference="van Rijsbergen 1979, Information Retrieval",
    defaults={"beta": 2.0},
)
def fbeta(ctx: PairContext, beta: float = 2.0) -> float:
    r"""$$F_\beta = \frac{(1+\beta^2)\,TP}{(1+\beta^2)\,TP + \beta^2 FN + FP}$$"""
    b2 = beta * beta
    return _ratio((1 + b2) * ctx.tp, (1 + b2) * ctx.tp + b2 * ctx.fn + ctx.fp, ctx, 1.0)


@register_metric(
    "tversky", display="Tversky index", abbr="TI", family="overlap", better="higher",
    summary="Asymmetric Dice that weights false positives (alpha) and false negatives (beta) differently.",
    reference="Tversky 1977, Psychological Review 84(4); Salehi et al. 2017, MLMI",
    defaults={"alpha": 0.3, "beta": 0.7},
)
def tversky(ctx: PairContext, alpha: float = 0.3, beta: float = 0.7) -> float:
    r"""$$\mathrm{TI}_{\alpha,\beta} = \frac{TP}{TP + \alpha FP + \beta FN}$$"""
    return _ratio(ctx.tp, ctx.tp + alpha * ctx.fp + beta * ctx.fn, ctx, 1.0)


@register_metric(
    "accuracy", display="Voxel accuracy", abbr="Acc", family="overlap", better="higher",
    summary="Fraction of all voxels labelled correctly; uninformative in 3D because background dominates.",
    reference="Taha & Hanbury 2015, BMC Med Imaging 15:29",
)
def accuracy(ctx: PairContext) -> float:
    r"""$$\mathrm{Acc} = \frac{TP+TN}{TP+FP+FN+TN}$$"""
    return (ctx.tp + ctx.tn) / ctx.n_voxels


@register_metric(
    "balanced_accuracy", display="Balanced accuracy", abbr="BAcc", family="overlap", better="higher",
    summary="Mean of sensitivity and specificity; equals the ROC AUC of a binary prediction.",
    reference="Brodersen et al. 2010, ICPR; Taha & Hanbury 2015",
)
def balanced_accuracy(ctx: PairContext) -> float:
    r"""$$\mathrm{BAcc} = \tfrac{1}{2}\left(\frac{TP}{TP+FN} + \frac{TN}{TN+FP}\right)$$"""
    if ctx.ref_empty:
        return ctx.best_or_nan(1.0) if ctx.pred_empty else float("nan")
    return 0.5 * (recall(ctx) + specificity(ctx))


@register_metric(
    "mcc", display="Matthews correlation coefficient", abbr="MCC", family="agreement", better="higher",
    value_range=(-1.0, 1.0),
    summary="Correlation between prediction and reference over all voxels; robust to class imbalance among TN-aware metrics.",
    reference="Matthews 1975, BBA 405(2); Chicco & Jurman 2020, BMC Genomics",
)
def mcc(ctx: PairContext) -> float:
    r"""$$\mathrm{MCC} = \frac{TP\cdot TN - FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$"""
    tp, fp, fn, tn = (float(x) for x in ctx.counts)
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if den == 0:
        return ctx.best_or_nan(1.0) if ctx.both_empty else 0.0
    return (tp * tn - fp * fn) / den


@register_metric(
    "cohen_kappa", display="Cohen's kappa", abbr="κ", family="agreement", better="higher",
    value_range=(-1.0, 1.0),
    summary="Voxel agreement corrected for the agreement expected by chance.",
    reference="Cohen 1960, Educ Psychol Meas 20(1); Taha & Hanbury 2015",
)
def cohen_kappa(ctx: PairContext) -> float:
    r"""$$\kappa = \frac{p_o - p_e}{1 - p_e},\quad p_o = \frac{TP+TN}{N},\;
    p_e = \frac{(TP+FP)(TP+FN) + (TN+FN)(TN+FP)}{N^2}$$"""
    tp, fp, fn, tn = (float(x) for x in ctx.counts)
    n = tp + fp + fn + tn
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (n * n)
    if pe == 1.0:
        return ctx.best_or_nan(1.0)
    return (po - pe) / (1.0 - pe)
