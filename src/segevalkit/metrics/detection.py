r"""Detection and instance (lesion-wise) metrics.

A voxel-level Dice of 0.8 can hide a model that finds the one large tumour and
misses every small metastasis. When the clinical question is *"were the
lesions found?"* the unit of analysis must be the lesion, not the voxel.

Instances are the connected components of each mask (26-connected by default;
see `PairContext`). Components are then
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
from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage

from .base import register_metric
from .context import PairContext

__all__ = ["InstanceMatch", "match_instances", "lesion_table", "component_scores"]


@dataclass
class InstanceMatch:
    """Result of matching reference and predicted components.

    Attributes:
        n_ref: Number of reference components.
        n_pred: Number of predicted components.
        ref_sizes: Voxel counts of each reference component (index 0 unused).
        pred_sizes: Voxel counts of each predicted component (index 0 unused).
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
    r"""$$\mathrm{L\text{-}TPR} = \frac{TP_{\mathrm{les}}}{TP_{\mathrm{les}} + FN_{\mathrm{les}}}$$"""
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
    r"""$$\mathrm{L\text{-}PPV} = \frac{TP_{\mathrm{les}}}{TP_{\mathrm{les}} + FP_{\mathrm{les}}}$$"""
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
    r"""$$\mathrm{L\text{-}F1} = \frac{2\,TP_{\mathrm{les}}}{2\,TP_{\mathrm{les}} + FP_{\mathrm{les}} + FN_{\mathrm{les}}}$$

    Under the ``"overlap"`` criterion \(TP_{\mathrm{les}}\) is the number of detected
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
    "false_positive_lesions", display="False-positive lesion count", abbr="FP lesions", family="detection",
    better="lower", value_range=(0.0, float("inf")),
    summary="Number of predicted components that overlap no reference lesion.",
    reference="Maier-Hein et al. 2024 (Metrics Reloaded); Chakraborty & Berbaum 2004, Med Phys 31(8) (FROC)",
    defaults=_MATCH_DOC,
)
def false_positive_lesions(ctx: PairContext, **kw) -> float:
    return float(match_instances(ctx, **kw).fp)


@register_metric(
    "false_negative_lesions", display="Missed lesion count", abbr="FN lesions", family="detection",
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
    r"""$$\mathrm{LW\text{-}DSC} = \frac{1}{N_G + FP_{\mathrm{les}}}\sum_{i=1}^{N_G}
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


def component_scores(ctx: PairContext, reduce: str = "max") -> Tuple[np.ndarray, str]:
    """Confidence score of every predicted component (cached).

    With a probability map the score is the ``"max"`` (default) or ``"mean"``
    foreground probability inside the component, the usual lesion confidence
    for FROC analysis (LUNA16, autoPET). Without one, the component volume in
    mL is used, so larger predicted lesions rank as more confident.

    Returns:
        ``(scores, score_type)``: one score per predicted component (index
        ``j-1`` for component ``j``) and ``"max_prob"``, ``"mean_prob"`` or
        ``"volume_ml"``.
    """
    if reduce not in ("max", "mean"):
        raise ValueError("reduce must be 'max' or 'mean'")

    def compute():
        lp, npred = ctx.pred_components
        if ctx.prob is None:
            sizes = np.bincount(lp.ravel(), minlength=npred + 1)[1:]
            return sizes * (ctx.voxel_volume_mm3 / 1000.0), "volume_ml"
        if npred == 0:
            return np.zeros(0), f"{reduce}_prob"
        fn = ndimage.maximum if reduce == "max" else ndimage.mean
        return np.asarray(fn(ctx.prob, lp, index=np.arange(1, npred + 1)), dtype=float), f"{reduce}_prob"

    return ctx.memo(("component_scores", reduce), compute)


def _lesion_pairs(ctx: PairContext):
    """For each reference lesion: a PairContext of the lesion against the predicted components touching it."""
    m = match_instances(ctx, criterion="overlap")
    lg, _ = ctx.ref_components
    lp, _ = ctx.pred_components
    core = m.inter[1:, 1:]
    out = []
    for i in range(m.n_ref):
        touching = np.nonzero(core[i] > 0)[0] + 1
        if touching.size == 0:
            out.append(None)
            continue
        g = lg == (i + 1)
        p = np.isin(lp, touching)
        crop = tuple(slice(max(sl.start - 1, 0), sl.stop + 1) for sl in ndimage.find_objects((g | p).astype(np.uint8))[0])
        out.append(PairContext(p[crop], g[crop], ctx.spacing, device=ctx.device, empty=ctx.empty,
                               connectivity=ctx.connectivity))
    return m, out


def _lesionwise(ctx: PairContext, per_lesion, missed_value: float, fp_value: float, both_empty_value: float):
    m, subs = ctx.memo("lesion_pairs", lambda: _lesion_pairs(ctx))
    den = m.n_ref + m.fp
    if den == 0:
        return ctx.best_or_nan(both_empty_value)
    total = sum(missed_value if sub is None else per_lesion(sub) for sub in subs) + m.fp * fp_value
    return float(total / den)


@register_metric(
    "lesionwise_hd95", display="Lesion-wise HD95 (BraTS 2023)", abbr="LW-HD95", family="detection",
    better="lower", value_range=(0.0, float("inf")), unit="mm", requires=("spacing",),
    summary="HD95 averaged over lesions; a missed or a spurious lesion scores the empty-mask distance penalty.",
    reference="Kazerooni et al. 2023, arXiv:2305.17033; BraTS-2023-Metrics code (github.com/rachitsaluja/BraTS-2023-Metrics)",
)
def lesionwise_hd95(ctx: PairContext) -> float:
    r"""$$\mathrm{LW\text{-}HD95} = \frac{1}{N_G + FP_{\mathrm{les}}}\Big(\sum_{i=1}^{N_G}
    \mathrm{HD95}\big(G_i,\ \textstyle\bigcup_{j: P_j\cap G_i\neq\emptyset} P_j\big) + FP_{\mathrm{les}}\cdot d_{\max}\Big)$$

    An undetected reference lesion scores \(d_{\max}\), the one-mask-empty
    distance of the `EmptyPolicy` (374 mm under the ``brats2023`` preset, the
    image diagonal by default), and so does every false-positive component.
    """
    from .distance import hd95

    pen = ctx.distance_penalty()
    return _lesionwise(ctx, lambda sub: hd95(sub), pen, pen, 0.0)


@register_metric(
    "lesionwise_nsd", display="Lesion-wise normalized surface Dice", abbr="LW-NSD", family="detection",
    better="higher", requires=("spacing",),
    summary="NSD averaged over lesions; a missed or a spurious lesion scores 0, so small lesions count as much as large.",
    reference="Kazerooni et al. 2023, arXiv:2305.17033 (lesion-wise protocol); Nikolov et al. 2021 (NSD)",
    defaults={"tolerance_mm": 2.0},
)
def lesionwise_nsd(ctx: PairContext, tolerance_mm: float = 2.0) -> float:
    r"""$$\mathrm{LW\text{-}NSD}_\tau = \frac{1}{N_G + FP_{\mathrm{les}}}\sum_{i=1}^{N_G}
    \mathrm{NSD}_\tau\big(G_i,\ \textstyle\bigcup_{j: P_j\cap G_i\neq\emptyset} P_j\big)$$"""
    from .distance import nsd

    return _lesionwise(ctx, lambda sub: nsd(sub, tolerance_mm=tolerance_mm), 0.0, 0.0, 1.0)


@register_metric(
    "lesion_ap", display="Lesion-level average precision", abbr="L-AP", family="detection", better="higher",
    summary="Area under the lesion precision-recall curve obtained by ranking predicted lesions by confidence.",
    reference="Everingham et al. 2010, IJCV 88 (PASCAL VOC AP); Maier-Hein et al. 2024 (Metrics Reloaded)",
    defaults={"reduce": "max"},
)
def lesion_ap(ctx: PairContext, reduce: str = "max") -> float:
    r"""$$\mathrm{AP} = \sum_k (R_k - R_{k-1})\,\max_{k'\ge k} P_{k'}$$

    Predicted components are ranked by `component_scores` (probability, or
    volume without a probability map). The \(k\)-th threshold keeps the top
    components; \(R_k\) is the fraction of reference lesions touched by a kept
    component and \(P_k = TP_{\mathrm{les}}/(TP_{\mathrm{les}} + FP_{\mathrm{les}})\)
    with detected reference lesions as true positives, the definition of
    `lesion_f1`. A
    case with no reference lesion has no AP (NaN, or 1 if nothing is
    predicted, per the `EmptyPolicy`). The dataset-level AP over all cases is
    `segevalkit.stats.lesion_pr`.
    """
    m = match_instances(ctx, criterion="overlap")
    if m.n_ref == 0:
        return ctx.best_or_nan(1.0) if m.n_pred == 0 else float("nan")
    scores, _ = component_scores(ctx, reduce)
    # recall at a threshold = reference lesions touched by any kept component
    core = m.inter[1:, 1:] > 0
    order = np.argsort(-scores, kind="stable")
    detected = np.zeros(m.n_ref, bool)
    rec, prec, kept_fp = [0.0], [1.0], 0
    for rank, j in enumerate(order, start=1):
        detected |= core[:, j]
        kept_fp += not bool(m.pred_matched[j])
        if rank == len(order) or scores[order[rank]] != scores[j]:
            tp = int(detected.sum())
            rec.append(tp / m.n_ref)
            prec.append(tp / (tp + kept_fp) if tp + kept_fp else 1.0)
    rec, prec = np.array(rec), np.maximum.accumulate(np.array(prec)[::-1])[::-1]
    return float(np.sum(np.diff(rec) * prec[1:]))


def lesion_table(ctx: PairContext, criterion: str = "overlap", score: str = "max", **kw) -> List[Dict[str, float]]:
    """One row per reference lesion and per unmatched predicted component.

    Columns: ``kind`` (``"ref"``/``"pred_fp"``), ``component``, ``volume_ml``,
    ``detected``, ``dice``, ``iou``, ``n_touching``, ``score`` and
    ``score_type``. For a reference lesion, ``score`` is the highest
    confidence among the predicted components touching it (NaN if none), i.e.
    the threshold up to which the lesion stays detected; for a false-positive
    component it is its own confidence (see `component_scores`). These two
    columns are all that FROC, CPM and lesion-level PR curves need
    (`segevalkit.stats.froc`, `segevalkit.stats.lesion_pr`).
    """
    m = match_instances(ctx, criterion=criterion, **kw)
    scores, score_type = component_scores(ctx, score)
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
            "score": float(scores[touching].max()) if touching.size else float("nan"), "score_type": score_type,
        })
    for j in np.nonzero(~m.pred_matched)[0]:
        rows.append({
            "kind": "pred_fp", "component": int(j + 1), "volume_ml": m.pred_sizes[j + 1] * vox_ml,
            "detected": False, "dice": 0.0, "iou": 0.0, "n_touching": 0,
            "score": float(scores[j]), "score_type": score_type,
        })
    return rows
