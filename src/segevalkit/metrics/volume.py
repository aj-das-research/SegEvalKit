r"""Volume agreement metrics.

These compare *how much* was segmented, not *where*: a prediction can have a
perfect volume and zero overlap. They answer volumetry questions (organ
volume, tumour burden, response assessment) and must be paired with an overlap
or distance metric for anything else.

Volumes are reported in millilitres (1 mL = 1000 mm³) using the voxel spacing.
"""

from __future__ import annotations

from .base import register_metric
from .context import PairContext


@register_metric(
    "pred_volume", display="Predicted volume", abbr="Vol(P)", family="volume", better="none",
    value_range=(0.0, float("inf")), unit="mL", requires=("spacing",),
    summary="Physical volume of the prediction (descriptive, not a score).",
    reference="—",
)
def pred_volume(ctx: PairContext) -> float:
    r"""$$V_P = |P|\cdot v_{\text{voxel}} / 1000$$"""
    return (ctx.tp + ctx.fp) * ctx.voxel_volume_mm3 / 1000.0


@register_metric(
    "ref_volume", display="Reference volume", abbr="Vol(G)", family="volume", better="none",
    value_range=(0.0, float("inf")), unit="mL", requires=("spacing",),
    summary="Physical volume of the reference (descriptive; the usual stratification variable).",
    reference="—",
)
def ref_volume(ctx: PairContext) -> float:
    r"""$$V_G = |G|\cdot v_{\text{voxel}} / 1000$$"""
    return (ctx.tp + ctx.fn) * ctx.voxel_volume_mm3 / 1000.0


@register_metric(
    "volume_difference", display="Signed volume difference", abbr="ΔV", family="volume", better="zero",
    value_range=(-float("inf"), float("inf")), unit="mL", requires=("spacing",),
    summary="Predicted minus reference volume in mL; positive means over-segmentation (bias).",
    reference="Heimann et al. 2009, IEEE TMI 28(8)",
)
def volume_difference(ctx: PairContext) -> float:
    r"""$$\Delta V = V_P - V_G$$"""
    return pred_volume(ctx) - ref_volume(ctx)


@register_metric(
    "absolute_volume_difference", display="Absolute volume difference", abbr="AVD", family="volume",
    better="lower", value_range=(0.0, float("inf")), unit="mL", requires=("spacing",),
    summary="Magnitude of the volume error in mL (used e.g. by ISLES and autoPET).",
    reference="Hernandez Petzsche et al. 2022, Sci Data 9:762 (ISLES'22)",
)
def absolute_volume_difference(ctx: PairContext) -> float:
    r"""$$\mathrm{AVD} = |V_P - V_G|$$"""
    return abs(volume_difference(ctx))


@register_metric(
    "relative_volume_difference", display="Relative volume difference", abbr="RVD", family="volume",
    better="zero", value_range=(-1.0, float("inf")),
    summary="Signed volume error relative to the reference volume; −0.2 means 20 % under-segmented.",
    reference="Heimann et al. 2009, IEEE TMI 28(8)",
)
def relative_volume_difference(ctx: PairContext) -> float:
    r"""$$\mathrm{RVD} = \frac{|P| - |G|}{|G|}$$"""
    n_ref = ctx.tp + ctx.fn
    if n_ref == 0:
        return ctx.best_or_nan(0.0) if ctx.pred_empty else float("nan")
    return (ctx.tp + ctx.fp - n_ref) / n_ref


@register_metric(
    "volumetric_similarity", display="Volumetric similarity", abbr="VS", family="volume", better="higher",
    summary="One minus the absolute volume difference normalised by the summed volumes; ignores location.",
    reference="Taha & Hanbury 2015, BMC Med Imaging 15:29",
)
def volumetric_similarity(ctx: PairContext) -> float:
    r"""$$\mathrm{VS} = 1 - \frac{\big||P|-|G|\big|}{|P|+|G|} = 1 - \frac{|FN-FP|}{2TP+FP+FN}$$"""
    den = 2 * ctx.tp + ctx.fp + ctx.fn
    if den == 0:
        return ctx.best_or_nan(1.0)
    return 1.0 - abs(ctx.fn - ctx.fp) / den
