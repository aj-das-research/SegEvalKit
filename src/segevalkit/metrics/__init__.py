"""Segmentation metrics.

Importing this package registers every built-in metric. Use
:func:`compute_metrics` for one binary pair, or the higher-level
:class:`segevalkit.Evaluator` for whole datasets and multi-label volumes.

Example:
    >>> import numpy as np
    >>> from segevalkit.metrics import compute_metrics
    >>> g = np.zeros((32, 32, 32), bool); g[8:24, 8:24, 8:24] = True
    >>> p = np.roll(g, 2, axis=0)
    >>> round(compute_metrics(p, g, ["dice"], spacing=(1, 1, 1))["dice"], 3)
    0.875
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, Iterable, Mapping, Optional, Sequence, Union

import numpy as np

from . import agreement, calibration, detection, distance, overlap, topology, volume  # noqa: F401
from .base import (
    FAMILIES,
    METRIC_SETS,
    MetricInfo,
    get_metric,
    list_metrics,
    register_metric,
    resolve_metrics,
)
from .context import EmptyPolicy, PairContext
from .detection import lesion_table, match_instances
from .topology import betti_numbers

__all__ = [
    "FAMILIES",
    "METRIC_SETS",
    "MetricInfo",
    "EmptyPolicy",
    "PairContext",
    "compute_metrics",
    "get_metric",
    "list_metrics",
    "register_metric",
    "resolve_metrics",
    "lesion_table",
    "match_instances",
    "betti_numbers",
]


def compute_metrics(
    pred: Union[np.ndarray, PairContext],
    ref: Optional[np.ndarray] = None,
    metrics: Union[str, Iterable[str]] = "default",
    *,
    spacing: Optional[Sequence[float]] = None,
    prob: Optional[np.ndarray] = None,
    params: Optional[Mapping[str, Mapping]] = None,
    device: str = "cpu",
    empty: EmptyPolicy = EmptyPolicy(),
    strict: bool = False,
    **context_kwargs,
) -> Dict[str, float]:
    """Compute several metrics for one binary (prediction, reference) pair.

    Args:
        pred: Predicted mask, or an existing :class:`PairContext` (then ``ref``
            and the context arguments are ignored).
        ref: Reference mask.
        metrics: Metric names, aliases or set names (see
            :data:`~segevalkit.metrics.base.METRIC_SETS`), or ``"all"``.
        spacing: Voxel spacing in mm.
        prob: Optional foreground probability map.
        params: Per-metric keyword overrides, e.g. ``{"nsd": {"tolerance_mm": 1}}``.
        device: ``"cpu"`` or a CUDA device.
        empty: Empty-mask policy.
        strict: Raise on metric errors instead of recording NaN with a warning.

    Returns:
        ``{metric_name: value}``. Metrics that need a probability map are
        skipped (not NaN) when ``prob`` is not given.
    """
    if isinstance(pred, PairContext) and ref is not None and not isinstance(ref, np.ndarray):
        # compute_metrics(ctx, ["hd95"]): the second positional argument is the metric list.
        metrics, ref = ref, None
    ctx = pred if isinstance(pred, PairContext) else PairContext(
        pred, ref, spacing, prob, device=device, empty=empty, **context_kwargs)
    params = params or {}
    out: Dict[str, float] = {}
    for name in resolve_metrics(metrics):
        info = get_metric(name)
        if "probabilities" in info.requires and ctx.prob is None:
            continue
        try:
            out[name] = float(info(ctx, **params.get(name, {})))
        except Exception as exc:  # pragma: no cover - surfaced to the user
            if strict:
                raise
            warnings.warn(f"metric {name!r} failed: {exc}", RuntimeWarning, stacklevel=2)
            out[name] = math.nan
    return out
