r"""Boundary and surface-distance metrics.

Let \(\partial P\) and \(\partial G\) be the surface voxels of the
prediction and the reference (see `segevalkit.metrics.context.surface`)
and \(d(x, S) = \min_{y\in S}\lVert x-y\rVert_2\) the Euclidean distance in
millimetres from a point to a surface. The two *directed* distance sets are

\[
D_{P\to G} = \{d(p, \partial G) : p\in\partial P\},\qquad
D_{G\to P} = \{d(g, \partial P) : g\in\partial G\}.
\]

Every metric in this module is a summary of these two sets. Overlap metrics
are blind to *where* the error is; distance metrics are blind to *how much*
volume is wrong. Report at least one of each.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .base import register_metric
from .context import PairContext


def _distance_or_empty(ctx: PairContext):
    """Return None when the directed sets are usable, else the empty-case value."""
    if ctx.both_empty:
        return ctx.best_or_nan(0.0)
    if ctx.one_empty:
        return ctx.distance_penalty()
    return None


@register_metric(
    "hd", display="Hausdorff distance", abbr="HD", family="distance", better="lower",
    value_range=(0.0, float("inf")), unit="mm", requires=("spacing",),
    summary="The single worst boundary error: the largest distance from any surface point to the other surface.",
    reference="Huttenlocher et al. 1993, IEEE TPAMI 15(9)",
)
def hd(ctx: PairContext) -> float:
    r"""$$\mathrm{HD} = \max\Big(\max D_{P\to G},\; \max D_{G\to P}\Big)$$"""
    e = _distance_or_empty(ctx)
    if e is not None:
        return e
    a, b = ctx.surface_distances
    return float(max(a.max(), b.max()))


@register_metric(
    "hd_percentile", display="Percentile Hausdorff distance", abbr="HDq", family="distance",
    better="lower", value_range=(0.0, float("inf")), unit="mm", requires=("spacing",),
    summary="Hausdorff distance with the most extreme (100−q) % of boundary points ignored, for robustness to outliers.",
    reference="Huttenlocher et al. 1993; convention of DeepMind surface-distance / MONAI",
    defaults={"q": 95.0, "mode": "directed"},
)
def hd_percentile(ctx: PairContext, q: float = 95.0, mode: str = "directed") -> float:
    r"""$$\mathrm{HD}_q = \max\Big(P_q(D_{P\to G}),\; P_q(D_{G\to P})\Big)$$

    \(P_q\) is the *q*-th percentile (linear interpolation).
    ``mode="directed"`` (default) takes the maximum of the two directed
    percentiles, the convention of MetricsReloaded, MONAI, DeepMind's
    ``surface-distance`` and the BraTS code. ``mode="pooled"`` takes the
    percentile of the pooled set \(D_{P\to G}\cup D_{G\to P}\), the
    MedPy convention; it is never larger than the directed form.
    """
    e = _distance_or_empty(ctx)
    if e is not None:
        return e
    a, b = ctx.surface_distances
    if mode == "pooled":
        return float(np.percentile(np.concatenate([a, b]), q))
    if mode != "directed":
        raise ValueError("mode must be 'directed' or 'pooled'")
    return float(max(np.percentile(a, q), np.percentile(b, q)))


@register_metric(
    "hd95", display="95th-percentile Hausdorff distance", abbr="HD95", family="distance",
    better="lower", value_range=(0.0, float("inf")), unit="mm", requires=("spacing",),
    summary="Near-worst boundary error, ignoring the 5 % most extreme surface points (BraTS, KiTS, TopCoW...).",
    reference="Huttenlocher et al. 1993; Bakas et al. 2018, arXiv:1811.02629 (BraTS)",
    defaults={"mode": "directed"},
)
def hd95(ctx: PairContext, mode: str = "directed") -> float:
    r"""$$\mathrm{HD}_{95} = \max\Big(P_{95}(D_{P\to G}),\; P_{95}(D_{G\to P})\Big)$$

    ``mode="pooled"`` gives the MedPy convention (see `hd_percentile`).
    """
    return hd_percentile(ctx, q=95.0, mode=mode)


@register_metric(
    "assd", display="Average symmetric surface distance", abbr="ASSD", family="distance",
    better="lower", value_range=(0.0, float("inf")), unit="mm", requires=("spacing",),
    summary="Mean distance over all surface points of both masks: the typical boundary error in mm.",
    reference="Heimann et al. 2009, IEEE TMI 28(8); Yeghiazaryan & Voiculescu 2018, J Med Imaging 5(1)",
)
def assd(ctx: PairContext) -> float:
    r"""$$\mathrm{ASSD} = \frac{\sum D_{P\to G} + \sum D_{G\to P}}{|\partial P| + |\partial G|}$$"""
    e = _distance_or_empty(ctx)
    if e is not None:
        return e
    a, b = ctx.surface_distances
    return float((a.sum() + b.sum()) / (a.size + b.size))


@register_metric(
    "masd", display="Mean average surface distance", abbr="MASD", family="distance",
    better="lower", value_range=(0.0, float("inf")), unit="mm", requires=("spacing",),
    summary="Average of the two directed mean distances; unlike ASSD, both surfaces weigh equally regardless of size.",
    reference="Yeghiazaryan & Voiculescu 2018, J Med Imaging 5(1); Maier-Hein et al. 2024 (Metrics Reloaded)",
)
def masd(ctx: PairContext) -> float:
    r"""$$\mathrm{MASD} = \tfrac{1}{2}\big(\overline{D_{P\to G}} + \overline{D_{G\to P}}\big)$$"""
    e = _distance_or_empty(ctx)
    if e is not None:
        return e
    a, b = ctx.surface_distances
    return float(0.5 * (a.mean() + b.mean()))


@register_metric(
    "nsd", display="Normalized surface Dice", abbr="NSD", family="distance", better="higher",
    requires=("spacing",),
    summary="Fraction of both surfaces lying within a clinically acceptable tolerance tau of the other surface.",
    reference="Nikolov et al. 2021, J Med Internet Res 23(7):e26151",
    defaults={"tolerance_mm": 2.0},
)
def nsd(ctx: PairContext, tolerance_mm: float = 2.0) -> float:
    r"""$$\mathrm{NSD}_\tau = \frac{|\{d\in D_{P\to G}: d\le\tau\}| + |\{d\in D_{G\to P}: d\le\tau\}|}{|\partial P| + |\partial G|}$$

    This is the voxel-counting form (each surface voxel has unit weight), as in
    MONAI and the FLARE / PanTS evaluation code. DeepMind's reference
    implementation weights each surface element by its area; the two agree
    closely for smooth, well-sampled surfaces. The tolerance is task-specific
    and should reflect inter-rater variability (Metrics Reloaded).
    """
    if ctx.both_empty:
        return ctx.best_or_nan(1.0)
    if ctx.one_empty:
        return 0.0
    a, b = ctx.surface_distances
    return float((np.count_nonzero(a <= tolerance_mm) + np.count_nonzero(b <= tolerance_mm)) / (a.size + b.size))


def _inner_band(mask: np.ndarray, width_mm: float, spacing) -> np.ndarray:
    """Inner boundary band: the surface voxels plus every foreground voxel within ``width_mm`` of the background.

    The surface voxels are always included, so the band is never empty for a
    non-empty mask, even when ``width_mm`` is smaller than the voxel size.
    """
    from .context import surface

    inside = ndimage.distance_transform_edt(mask, sampling=spacing)
    return surface(mask) | (mask & (inside <= width_mm))


@register_metric(
    "boundary_iou", display="Boundary IoU", abbr="BIoU", family="distance", better="higher",
    requires=("spacing",),
    summary="IoU restricted to a thin band inside each boundary; sensitive to boundary quality even for large objects.",
    reference="Cheng et al. 2021, CVPR (Boundary IoU)",
    defaults={"width_mm": 2.0},
)
def boundary_iou(ctx: PairContext, width_mm: float = 2.0) -> float:
    r"""$$\mathrm{BIoU}_d = \frac{|(P_d\cap P)\cap(G_d\cap G)|}{|(P_d\cap P)\cup(G_d\cap G)|}$$

    \(X_d\) is the set of voxels within distance *d* of the contour of *X*.
    """
    if ctx.both_empty:
        return ctx.best_or_nan(1.0)
    if ctx.one_empty:
        return 0.0
    bp = _inner_band(ctx.pred_cropped, width_mm, ctx.spacing)
    bg = _inner_band(ctx.ref_cropped, width_mm, ctx.spacing)
    union = np.count_nonzero(bp | bg)
    return float(np.count_nonzero(bp & bg) / union) if union else 1.0


@register_metric(
    "centroid_distance", display="Centroid distance", abbr="CD", family="distance", better="lower",
    value_range=(0.0, float("inf")), unit="mm", requires=("spacing",),
    summary="Distance between the centres of mass of the two masks; a localisation error insensitive to shape.",
    reference="Reinke et al. 2024, Nat Methods 21 (localisation criteria)",
)
def centroid_distance(ctx: PairContext) -> float:
    r"""$$\mathrm{CD} = \lVert \bar{x}_P - \bar{x}_G \rVert_2$$"""
    e = _distance_or_empty(ctx)
    if e is not None:
        return e
    sp = np.asarray(ctx.spacing)
    cp = np.asarray(ndimage.center_of_mass(ctx.pred_cropped)) * sp
    cg = np.asarray(ndimage.center_of_mass(ctx.ref_cropped)) * sp
    return float(np.linalg.norm(cp - cg))
