r"""Topology and centreline metrics.

Overlap and distance metrics can be excellent for a vessel tree that is broken
into ten pieces, or for a hollow organ whose lumen is filled in. Topological
metrics count what shape *is*: connected pieces, tunnels and cavities.

For a 3D binary object the Betti numbers are

* \(\beta_0\): number of connected components (26-connected foreground),
* \(\beta_1\): number of independent tunnels / handles (loops),
* \(\beta_2\): number of enclosed cavities (6-connected background pockets),

related by the Euler characteristic \(\chi = \beta_0 - \beta_1 + \beta_2\).
The (26, 6) adjacency pair is the standard well-composed choice for 3D digital
topology (Kong & Rosenfeld 1989).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import ndimage

from .base import register_metric
from .context import FACE, FULL, PairContext


def betti_numbers(mask: np.ndarray) -> Tuple[int, int, int]:
    """Betti numbers ``(b0, b1, b2)`` of a 3D binary mask.

    The mask is padded with background so that exactly one background
    component (the outside) is unbounded.
    """
    from skimage.measure import euler_number

    m = np.pad(np.asarray(mask, dtype=bool), 1)
    if not m.any():
        return 0, 0, 0
    _, b0 = ndimage.label(m, structure=FULL)
    _, n_bg = ndimage.label(~m, structure=FACE)
    b2 = n_bg - 1
    chi = int(euler_number(m, connectivity=3))
    b1 = b0 + b2 - chi
    return int(b0), int(b1), int(b2)


def _betti(ctx: PairContext):
    return ctx.memo("betti", lambda: (betti_numbers(ctx.pred_cropped), betti_numbers(ctx.ref_cropped)))


def _betti_error(ctx: PairContext, k: int) -> float:
    if ctx.both_empty:
        return ctx.best_or_nan(0.0)
    bp, bg = _betti(ctx)
    return float(abs(bp[k] - bg[k]))


@register_metric(
    "betti0_error", display="Betti-0 error (components)", abbr="β₀ err", family="topology", better="lower",
    value_range=(0.0, float("inf")),
    summary="Difference in the number of connected pieces: counts spurious islands and fragmentation.",
    reference="Hu et al. 2019, NeurIPS (TopoLoss); Yang et al. 2023, arXiv:2312.17670 (TopCoW)",
)
def betti0_error(ctx: PairContext) -> float:
    r"""$$\varepsilon_{\beta_0} = |\beta_0(P) - \beta_0(G)|$$"""
    return _betti_error(ctx, 0)


@register_metric(
    "betti1_error", display="Betti-1 error (tunnels/loops)", abbr="β₁ err", family="topology", better="lower",
    value_range=(0.0, float("inf")),
    summary="Difference in the number of loops/handles: e.g. a vessel ring (circle of Willis) opened or falsely closed.",
    reference="Hu et al. 2019, NeurIPS; Yang et al. 2023 (TopCoW)",
)
def betti1_error(ctx: PairContext) -> float:
    r"""$$\varepsilon_{\beta_1} = |\beta_1(P) - \beta_1(G)|$$"""
    return _betti_error(ctx, 1)


@register_metric(
    "betti2_error", display="Betti-2 error (cavities)", abbr="β₂ err", family="topology", better="lower",
    value_range=(0.0, float("inf")),
    summary="Difference in the number of enclosed holes: e.g. a filled-in lumen or a spurious internal void.",
    reference="Hu et al. 2019, NeurIPS",
)
def betti2_error(ctx: PairContext) -> float:
    r"""$$\varepsilon_{\beta_2} = |\beta_2(P) - \beta_2(G)|$$"""
    return _betti_error(ctx, 2)


@register_metric(
    "euler_error", display="Euler characteristic error", abbr="χ err", family="topology", better="lower",
    value_range=(0.0, float("inf")),
    summary="Difference of the Euler characteristic (components − loops + cavities); one number for overall topology.",
    reference="Kong & Rosenfeld 1989, CVGIP 48(3)",
)
def euler_error(ctx: PairContext) -> float:
    r"""$$\varepsilon_\chi = |\chi(P) - \chi(G)|,\quad \chi = \beta_0 - \beta_1 + \beta_2$$"""
    if ctx.both_empty:
        return ctx.best_or_nan(0.0)
    bp, bg = _betti(ctx)
    return float(abs((bp[0] - bp[1] + bp[2]) - (bg[0] - bg[1] + bg[2])))


@register_metric(
    "cldice", display="Centreline Dice", abbr="clDice", family="topology", better="higher",
    summary="Dice computed on skeletons: rewards connected, complete tubular structures (vessels, airways, ducts).",
    reference="Shit et al. 2021, CVPR (clDice)",
)
def cldice(ctx: PairContext) -> float:
    r"""$$\mathrm{clDice} = 2\,\frac{T_{\mathrm{prec}}\,T_{\mathrm{sens}}}{T_{\mathrm{prec}}+T_{\mathrm{sens}}},\quad
    T_{\mathrm{prec}} = \frac{|S_P\cap G|}{|S_P|},\; T_{\mathrm{sens}} = \frac{|S_G\cap P|}{|S_G|}$$

    \(S_X\) is the topological skeleton of *X*.
    """
    if ctx.both_empty:
        return ctx.best_or_nan(1.0)
    if ctx.one_empty:
        return 0.0
    sp, sg = ctx.pred_skeleton, ctx.ref_skeleton
    n_sp, n_sg = np.count_nonzero(sp), np.count_nonzero(sg)
    if n_sp == 0 or n_sg == 0:
        return 0.0
    t_prec = np.count_nonzero(sp & ctx.ref_cropped) / n_sp
    t_sens = np.count_nonzero(sg & ctx.pred_cropped) / n_sg
    if t_prec + t_sens == 0:
        return 0.0
    return float(2 * t_prec * t_sens / (t_prec + t_sens))

