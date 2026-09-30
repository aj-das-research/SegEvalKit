r"""Detection and instance (lesion-wise) metrics.

A voxel-level Dice of 0.8 can hide a model that finds the one large tumour and
misses every small metastasis. When the clinical question is *"were the
lesions found?"* the unit of analysis must be the lesion, not the voxel.

Instances are the connected components of each mask (26-connected by default;
see :class:`~segevalkit.metrics.context.PairContext`). Components are then
matched between prediction and reference with one of two criteria:

``"overlap"`` (default)
    A reference lesion is *detected* if any predicted component overlaps it by
    at least ``min_overlap`` of its volume; a predicted component is a false
    positive if it overlaps no reference lesion. Many-to-many, as in ISLES,
    autoPET and BraTS lesion-wise evaluation.

``"iou"``
    One-to-one matching by maximum-IoU assignment (Hungarian), keeping pairs
    with IoU > ``iou_threshold``. With the default threshold of 0.5 each match
    is unique, which is the definition behind panoptic quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .base import register_metric
from .context import PairContext

__all__ = ["InstanceMatch", "match_instances", "lesion_table"]


@dataclass
class InstanceMatch:
    """Result of matching reference and predicted components.

    Attributes:
        n_ref, n_pred: Number of reference / predicted components.
        ref_sizes, pred_sizes: Voxel counts of each component (index 0 unused).
        inter: ``(n_ref+1, n_pred+1)`` voxel-overlap matrix; row/column 0 is background.
        ref_detected: Boolean per reference component (index ``i-1``).
        pred_matched: Boolean per predicted component (index ``j-1``).
        pairs: Matched ``(ref_id, pred_id, iou)`` triples (``"iou"`` criterion) or
            every overlapping pair (``"overlap"`` criterion).
    """

    n_ref: int
    n_pred: int
    ref_sizes: np.ndarray
    pred_sizes: np.ndarray
    inter: np.ndarray
    ref_detected: np.ndarray
    pred_matched: np.ndarray
    pairs: List[tuple]

    @property
    def tp_ref(self) -> int:
        return int(self.ref_detected.sum())

    @property
    def fn(self) -> int:
        return self.n_ref - self.tp_ref

    @property
    def tp_pred(self) -> int:
        return int(self.pred_matched.sum())

    @property
    def fp(self) -> int:
        return self.n_pred - self.tp_pred


def _overlap_matrix(ctx: PairContext):
    lg, ng = ctx.ref_components
    lp, npred = ctx.pred_components
    fg = (lg > 0) | (lp > 0)
    flat = lg[fg].astype(np.int64) * (npred + 1) + lp[fg].astype(np.int64)
    inter = np.bincount(flat, minlength=(ng + 1) * (npred + 1)).reshape(ng + 1, npred + 1)
    ref_sizes = inter.sum(axis=1)
    pred_sizes = inter.sum(axis=0)
    return ng, npred, inter, ref_sizes, pred_sizes


def match_instances(
    ctx: PairContext,
    criterion: str = "overlap",
    iou_threshold: float = 0.5,
    min_overlap: float = 0.0,
) -> InstanceMatch:
    """Match connected components of prediction and reference (cached per parameters)."""
    key = ("match", criterion, float(iou_threshold), float(min_overlap))
    return ctx.memo(key, lambda: _match(ctx, criterion, iou_threshold, min_overlap))


def _match(ctx, criterion, iou_threshold, min_overlap) -> InstanceMatch:
    ng, npred, inter, rs, ps = ctx.memo("overlap_matrix", lambda: _overlap_matrix(ctx))
    core = inter[1:, 1:].astype(np.float64)
    ref_detected = np.zeros(ng, dtype=bool)
    pred_matched = np.zeros(npred, dtype=bool)
    pairs: List[tuple] = []
    if ng and npred:
        union = rs[1:, None] + ps[None, 1:] - core
        iou = np.divide(core, union, out=np.zeros_like(core), where=union > 0)
        if criterion == "overlap":
            frac = core / rs[1:, None]
            hit = (core > 0) & (frac >= min_overlap) if min_overlap > 0 else core > 0
            ref_detected = hit.any(axis=1)
            # A predicted component is a true positive only if it takes part in
            # a qualifying overlap; a component that merely grazes a lesion below
            # ``min_overlap`` counts as a false positive.
            pred_matched = hit.any(axis=0)
            pairs = [(int(i + 1), int(j + 1), float(iou[i, j])) for i, j in zip(*np.nonzero(hit))]
        elif criterion == "iou":
            from scipy.optimize import linear_sum_assignment

            rows, cols = linear_sum_assignment(-iou)
            for i, j in zip(rows, cols):
                if iou[i, j] > iou_threshold:
                    ref_detected[i] = pred_matched[j] = True
                    pairs.append((int(i + 1), int(j + 1), float(iou[i, j])))
        else:
            raise ValueError("criterion must be 'overlap' or 'iou'")
    return InstanceMatch(ng, npred, rs, ps, inter, ref_detected, pred_matched, pairs)


# ------------------------------------------------------------------ metrics
_MATCH_DOC = dict(criterion="overlap", iou_threshold=0.5, min_overlap=0.0)


@register_metric(
    "lesion_recall", display="Lesion-wise recall (detection sensitivity)", abbr="L-TPR", family="detection",
    better="higher",
    summary="Fraction of reference lesions that were detected, regardless of their size.",
    reference="Hernandez Petzsche et al. 2022 (ISLES'22); Maier-Hein et al. 2024 (Metrics Reloaded)",
    defaults=_MATCH_DOC,
)
def lesion_recall(ctx: PairContext, **kw) -> float:
    r"""$$\mathrm{L\text{-}TPR} = \frac{TP_{les}}{TP_{les} + FN_{les}}$$"""
    m = match_instances(ctx, **kw)
    if m.n_ref == 0:
        return ctx.best_or_nan(1.0) if m.n_pred == 0 else float("nan")
    return m.tp_ref / m.n_ref


@register_metric(
    "lesion_precision", display="Lesion-wise precision", abbr="L-PPV", family="detection", better="higher",
    summary="Fraction of predicted lesions that correspond to a real lesion (1 − false-discovery rate).",
    reference="Maier-Hein et al. 2024 (Metrics Reloaded)",
    defaults=_MATCH_DOC,
)
def lesion_precision(ctx: PairContext, **kw) -> float:
    r"""$$\mathrm{L\text{-}PPV} = \frac{TP_{les}}{TP_{les} + FP_{les}}$$"""
    m = match_instances(ctx, **kw)
    if m.n_pred == 0:
        return ctx.best_or_nan(1.0) if m.n_ref == 0 else float("nan")
    return m.tp_pred / m.n_pred


@register_metric(
    "lesion_f1", display="Lesion-wise F1 score", abbr="L-F1", family="detection", better="higher",
    summary="Harmonic mean of lesion precision and recall: one number for 'found the lesions, and only them'.",
    reference="Hernandez Petzsche et al. 2022, Sci Data 9:762 (ISLES'22)",
    defaults=_MATCH_DOC,
)
def lesion_f1(ctx: PairContext, **kw) -> float:
    r"""$$\mathrm{L\text{-}F1} = \frac{2\,TP_{les}}{2\,TP_{les} + FP_{les} + FN_{les}}$$

    Under the ``"overlap"`` criterion :math:`TP_{les}` is the number of detected
    reference lesions (as in the ISLES'22 evaluation code).
    """
    m = match_instances(ctx, **kw)
    den = 2 * m.tp_ref + m.fp + m.fn
    if den == 0:
        return ctx.best_or_nan(1.0)
    return 2 * m.tp_ref / den


@register_metric(
    "lesion_count_difference", display="Absolute lesion count difference", abbr="|ΔN|", family="detection",
    better="lower", value_range=(0.0, float("inf")),
    summary="How many more or fewer lesions were predicted than exist (ISLES'22 secondary metric).",
    reference="Hernandez Petzsche et al. 2022, Sci Data 9:762 (ISLES'22)",
)
def lesion_count_difference(ctx: PairContext) -> float:
    r"""$$|\Delta N| = |N_P - N_G|$$"""
    return float(abs(ctx.pred_components[1] - ctx.ref_components[1]))


@register_metric(
    "false_positive_lesions", display="False-positive lesion count", abbr="FP_les", family="detection",
    better="lower", value_range=(0.0, float("inf")),
    summary="Number of predicted components that overlap no reference lesion.",
    reference="Maier-Hein et al. 2024 (Metrics Reloaded); Chakraborty & Berbaum 2004, Med Phys 31(8) (FROC)",
    defaults=_MATCH_DOC,
)
def false_positive_lesions(ctx: PairContext, **kw) -> float:
    return float(match_instances(ctx, **kw).fp)


@register_metric(
    "false_negative_lesions", display="Missed lesion count", abbr="FN_les", family="detection",
    better="lower", value_range=(0.0, float("inf")),
    summary="Number of reference lesions that were not detected.",
    reference="Maier-Hein et al. 2024 (Metrics Reloaded)",
    defaults=_MATCH_DOC,
)
def false_negative_lesions(ctx: PairContext, **kw) -> float:
    return float(match_instances(ctx, **kw).fn)


@register_metric(
    "split_count", display="Split lesions", abbr="Splits", family="detection", better="lower",
    value_range=(0.0, float("inf")),
    summary="Number of reference lesions covered by more than one predicted component (over-fragmentation).",
    reference="Maier-Hein et al. 2024 (Metrics Reloaded); Carass et al. 2020, Sci Rep 10:8242",
)
def split_count(ctx: PairContext) -> float:
    r"""$$\#\{i : |\{j : P_j\cap G_i\neq\emptyset\}| > 1\}$$"""
    m = match_instances(ctx, criterion="overlap")
    return float(np.count_nonzero((m.inter[1:, 1:] > 0).sum(axis=1) > 1)) if m.n_ref and m.n_pred else 0.0


@register_metric(
    "merge_count", display="Merged lesions", abbr="Merges", family="detection", better="lower",
    value_range=(0.0, float("inf")),
    summary="Number of predicted components that cover more than one reference lesion (under-separation).",
    reference="Maier-Hein et al. 2024 (Metrics Reloaded); Carass et al. 2020, Sci Rep 10:8242",
)
def merge_count(ctx: PairContext) -> float:
    r"""$$\#\{j : |\{i : P_j\cap G_i\neq\emptyset\}| > 1\}$$"""
    m = match_instances(ctx, criterion="overlap")
    return float(np.count_nonzero((m.inter[1:, 1:] > 0).sum(axis=0) > 1)) if m.n_ref and m.n_pred else 0.0


@register_metric(
    "panoptic_quality", display="Panoptic quality", abbr="PQ", family="detection", better="higher",
    summary="Detection quality × segmentation quality of matched lesions: F1 of IoU>0.5 matches times their mean IoU.",
    reference="Kirillov et al. 2019, CVPR (Panoptic Segmentation)",
    defaults={"iou_threshold": 0.5},
)
def panoptic_quality(ctx: PairContext, iou_threshold: float = 0.5) -> float:
    r"""$$\mathrm{PQ} = \underbrace{\frac{\sum_{(p,g)\in TP}\mathrm{IoU}(p,g)}{|TP|}}_{SQ}\times
    \underbrace{\frac{|TP|}{|TP| + \tfrac12|FP| + \tfrac12|FN|}}_{RQ}$$"""
    m = match_instances(ctx, criterion="iou", iou_threshold=iou_threshold)
    den = m.tp_ref + 0.5 * m.fp + 0.5 * m.fn
    if den == 0:
        return ctx.best_or_nan(1.0)
    return float(sum(p[2] for p in m.pairs) / den)


@register_metric(
    "lesionwise_dice", display="Lesion-wise Dice (BraTS 2023)", abbr="LW-DSC", family="detection",
    better="higher",
    summary="Dice averaged over lesions, with missed and spurious lesions each scoring 0; small lesions count as much as large.",
    reference="Kazerooni et al. 2023, arXiv:2305.17033; BraTS-2023-Metrics code (github.com/rachitsaluja/BraTS-2023-Metrics)",
)
def lesionwise_dice(ctx: PairContext) -> float:
    r"""$$\mathrm{LW\text{-}DSC} = \frac{1}{N_G + FP_{les}}\sum_{i=1}^{N_G}
    \mathrm{DSC}\Big(G_i,\ \textstyle\bigcup_{j: P_j\cap G_i\neq\emptyset} P_j\Big)$$

    Every reference lesion is compared with the union of the predicted
    components that touch it; undetected lesions contribute 0 and each
    false-positive component adds 0 to the numerator and 1 to the denominator.
    (BraTS additionally dilates the reference before matching; pass a
    dilated reference if you need that exact behaviour.)
    """
    m = match_instances(ctx, criterion="overlap")
    den = m.n_ref + m.fp
    if den == 0:
        return ctx.best_or_nan(1.0)
    total = 0.0
    core = m.inter[1:, 1:]
    for i in range(m.n_ref):
        touching = np.nonzero(core[i] > 0)[0]
        if touching.size == 0:
            continue
        inter_i = core[i, touching].sum()
        pred_vol = m.pred_sizes[touching + 1].sum()
        total += 2.0 * inter_i / (m.ref_sizes[i + 1] + pred_vol)
    return float(total / den)


def lesion_table(ctx: PairContext, criterion: str = "overlap", **kw) -> List[Dict[str, float]]:
    """One row per reference lesion and per unmatched predicted component.

    Columns: ``kind`` (``"ref"``/``"pred_fp"``), ``component``, ``volume_ml``,
    ``detected``, ``dice``, ``iou``, ``n_touching``. Used for lesion-size
    stratified analysis and the FROC-style plots.
    """
    m = match_instances(ctx, criterion=criterion, **kw)
    vox_ml = ctx.voxel_volume_mm3 / 1000.0
    core = m.inter[1:, 1:]
    rows: List[Dict[str, float]] = []
    for i in range(m.n_ref):
        touching = np.nonzero(core[i] > 0)[0]
        inter_i = core[i, touching].sum() if touching.size else 0
        pred_vol = m.pred_sizes[touching + 1].sum() if touching.size else 0
        size = m.ref_sizes[i + 1]
        rows.append({
            "kind": "ref", "component": i + 1, "volume_ml": size * vox_ml,
            "detected": bool(m.ref_detected[i]),
            "dice": 2.0 * inter_i / (size + pred_vol) if size + pred_vol else 0.0,
            "iou": inter_i / (size + pred_vol - inter_i) if size + pred_vol - inter_i else 0.0,
            "n_touching": int(touching.size),
        })
    for j in np.nonzero(~m.pred_matched)[0]:
        rows.append({
            "kind": "pred_fp", "component": int(j + 1), "volume_ml": m.pred_sizes[j + 1] * vox_ml,
            "detected": False, "dice": 0.0, "iou": 0.0, "n_touching": 0,
        })
    return rows
