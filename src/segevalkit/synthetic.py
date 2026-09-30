"""Controlled perturbations of reference masks for metric sensitivity studies.

To understand what a metric *measures*, apply one known kind of error to a
real reference mask, increase its magnitude, and watch the metric respond.
Every perturbation here is parameterised in physical units (mm, mL, counts)
so that results transfer across datasets and voxel spacings:

=====================  =============================================================
``dilate`` / ``erode`` uniform boundary over-/under-segmentation by *d* mm
``shift``              rigid translation by *d* mm (registration-like error)
``islands``            *k* spurious false-positive blobs placed near the structure
``remove_slab``        a fraction of the structure cut away (partial miss)
``holes``              *k* internal cavities (topology change without much volume)
``cut``                a planar cut of width *d* mm (breaks connectivity, e.g. a vessel)
``boundary_noise``     smooth random boundary jitter of amplitude *d* mm
=====================  =============================================================

:func:`sensitivity_study` runs a grid of perturbations and returns a tidy
table ready for :func:`segevalkit.plotting.sensitivity_curves`.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import ndimage

__all__ = ["dilate", "erode", "shift", "islands", "remove_slab", "holes", "cut", "boundary_noise",
           "PERTURBATIONS", "sensitivity_study"]


def _sp(spacing):
    return np.asarray(spacing if spacing is not None else (1.0, 1.0, 1.0), dtype=float)


def dilate(mask: np.ndarray, mm: float, spacing=None, rng=None) -> np.ndarray:
    """Grow the mask by ``mm`` (exact Euclidean, spacing-aware)."""
    if mm <= 0:
        return mask.copy()
    return ndimage.distance_transform_edt(~mask, sampling=_sp(spacing)) <= mm


def erode(mask: np.ndarray, mm: float, spacing=None, rng=None) -> np.ndarray:
    """Shrink the mask by ``mm``."""
    if mm <= 0:
        return mask.copy()
    return ndimage.distance_transform_edt(mask, sampling=_sp(spacing)) > mm


def shift(mask: np.ndarray, mm: float, spacing=None, rng=None, axis: Optional[int] = None) -> np.ndarray:
    """Translate by ``mm`` along ``axis`` (default: a random in-plane direction), linear-interpolated."""
    sp = _sp(spacing)
    rng = np.random.default_rng(rng)
    if axis is None:
        theta = rng.uniform(0, 2 * np.pi)
        vec = np.array([np.cos(theta), np.sin(theta), 0.0]) * mm
    else:
        vec = np.zeros(3)
        vec[axis] = mm
    return ndimage.shift(mask.astype(np.float32), vec / sp, order=1, mode="constant") >= 0.5


def _ball(radius_mm: float, spacing) -> np.ndarray:
    sp = _sp(spacing)
    r = np.ceil(radius_mm / sp).astype(int)
    g = np.ogrid[tuple(slice(-k, k + 1) for k in r)]
    return sum((gi * s) ** 2 for gi, s in zip(g, sp)) <= radius_mm ** 2


def _stamp(out: np.ndarray, center, ball: np.ndarray, value: bool) -> None:
    lo = [c - b // 2 for c, b in zip(center, ball.shape)]
    sl_out, sl_b = [], []
    for ax in range(3):
        a0, a1 = max(lo[ax], 0), min(lo[ax] + ball.shape[ax], out.shape[ax])
        if a1 <= a0:
            return
        sl_out.append(slice(a0, a1))
        sl_b.append(slice(a0 - lo[ax], a1 - lo[ax]))
    region = out[tuple(sl_out)]
    b = ball[tuple(sl_b)]
    region[b] = value


def islands(mask: np.ndarray, k: int, spacing=None, rng=None, radius_mm: float = 4.0,
            max_distance_mm: float = 60.0) -> np.ndarray:
    """Add ``k`` spherical false-positive blobs (radius ``radius_mm``) outside the structure,
    within ``max_distance_mm`` of it."""
    out = mask.copy()
    k = int(k)
    if k <= 0:
        return out
    rng = np.random.default_rng(rng)
    sp = _sp(spacing)
    dist = ndimage.distance_transform_edt(~mask, sampling=sp)
    cand = np.argwhere((dist > 2 * radius_mm) & (dist < max_distance_mm))
    if cand.size == 0:
        return out
    ball = _ball(radius_mm, sp)
    for c in cand[rng.choice(len(cand), size=min(k, len(cand)), replace=False)]:
        _stamp(out, c, ball, True)
    return out


def remove_slab(mask: np.ndarray, fraction: float, spacing=None, rng=None, axis: int = 2) -> np.ndarray:
    """Remove the top ``fraction`` of the structure's voxels along ``axis`` (a partial miss)."""
    out = mask.copy()
    if fraction <= 0 or not mask.any():
        return out
    coords = np.nonzero(mask)[axis]
    cutoff = np.quantile(coords, 1 - fraction)
    idx = np.arange(mask.shape[axis]).reshape([-1 if a == axis else 1 for a in range(3)])
    out &= ~(idx > cutoff)
    return out


def holes(mask: np.ndarray, k: int, spacing=None, rng=None, radius_mm: float = 3.0) -> np.ndarray:
    """Carve ``k`` spherical internal cavities fully inside the structure."""
    out = mask.copy()
    k = int(k)
    if k <= 0:
        return out
    rng = np.random.default_rng(rng)
    sp = _sp(spacing)
    inside = ndimage.distance_transform_edt(mask, sampling=sp)
    cand = np.argwhere(inside > radius_mm + 1.5 * sp.max())
    if cand.size == 0:
        return out
    ball = _ball(radius_mm, sp)
    for c in cand[rng.choice(len(cand), size=min(k, len(cand)), replace=False)]:
        _stamp(out, c, ball, False)
    return out


def cut(mask: np.ndarray, mm: float, spacing=None, rng=None, axis: int = 2, position: float = 0.5) -> np.ndarray:
    """Remove a slab of width ``mm`` perpendicular to ``axis`` at the ``position`` quantile of the
    structure's extent: breaks connectivity with almost no volume change."""
    out = mask.copy()
    if mm <= 0 or not mask.any():
        return out
    sp = _sp(spacing)
    coords = np.nonzero(mask)[axis]
    center = np.quantile(coords, position)
    half = mm / sp[axis] / 2
    idx = np.arange(mask.shape[axis]).reshape([-1 if a == axis else 1 for a in range(3)])
    out &= ~((idx >= center - half) & (idx <= center + half))
    return out


def boundary_noise(mask: np.ndarray, mm: float, spacing=None, rng=None, smooth_mm: float = 6.0) -> np.ndarray:
    """Jitter the boundary with smooth random noise of amplitude ~``mm``.

    The signed distance to the boundary is offset by a smooth Gaussian random
    field, so the boundary moves in and out by up to about ``mm`` while the
    volume stays roughly constant: pure boundary error.
    """
    if mm <= 0:
        return mask.copy()
    rng = np.random.default_rng(rng)
    sp = _sp(spacing)
    sdt = ndimage.distance_transform_edt(mask, sampling=sp) - ndimage.distance_transform_edt(~mask, sampling=sp)
    field = ndimage.gaussian_filter(rng.standard_normal(mask.shape).astype(np.float32), smooth_mm / sp)
    field /= field.std() + 1e-8
    return sdt + mm * field > 0


#: Registry: name -> (function, unit label, default magnitudes).
PERTURBATIONS: Dict[str, tuple] = {
    "dilate": (dilate, "mm", [0, 1, 2, 3, 5, 8]),
    "erode": (erode, "mm", [0, 1, 2, 3, 5, 8]),
    "shift": (shift, "mm", [0, 1, 2, 4, 6, 10]),
    "islands": (islands, "number of false-positive blobs", [0, 1, 2, 4, 8]),
    "remove_slab": (remove_slab, "fraction removed", [0, 0.05, 0.1, 0.2, 0.4]),
    "holes": (holes, "number of cavities", [0, 1, 2, 4, 8]),
    "cut": (cut, "cut width [mm]", [0, 1, 2, 4, 8]),
    "boundary_noise": (boundary_noise, "noise amplitude [mm]", [0, 1, 2, 3, 5]),
}


def sensitivity_study(refs: Iterable, metrics: Sequence[str], *,
                      perturbations: Optional[Mapping[str, Sequence[float]]] = None,
                      params: Optional[Mapping[str, Mapping]] = None, seed: int = 0,
                      device: str = "cpu", progress: bool = False) -> pd.DataFrame:
    """Apply each perturbation at each magnitude to each reference and compute metrics.

    Args:
        refs: Iterable of ``(case_id, mask, spacing)`` tuples.
        metrics: Metric names.
        perturbations: ``{name: magnitudes}``; defaults to :data:`PERTURBATIONS`.
        params: Metric parameter overrides.

    Returns:
        Long table ``case_id, perturbation, magnitude, unit, metric, value``.
    """
    from .metrics import compute_metrics

    perturbations = perturbations or {k: v[2] for k, v in PERTURBATIONS.items()}
    rows = []
    refs = list(refs)
    it = refs
    if progress:
        from tqdm.auto import tqdm

        it = tqdm(refs, desc="sensitivity study")
    for ci, (case_id, ref, spacing) in enumerate(it):
        ref = np.asarray(ref, bool)
        for pi, (name, mags) in enumerate(perturbations.items()):
            fn: Callable = PERTURBATIONS[name][0]
            unit = PERTURBATIONS[name][1]
            for mag in mags:
                pred = fn(ref, mag, spacing, rng=seed + 1000 * ci + pi)
                vals = compute_metrics(pred, ref, metrics, spacing=spacing, params=params, device=device)
                for m, v in vals.items():
                    rows.append({"case_id": case_id, "perturbation": name, "magnitude": mag,
                                 "unit": unit, "metric": m, "value": v})
    return pd.DataFrame(rows)
