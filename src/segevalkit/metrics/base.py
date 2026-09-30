"""Metric registry.

Every metric in SegEvalKit is a plain function ``fn(ctx, **params) -> float`` that
reads what it needs from a `PairContext`.
The function is registered together with a `MetricInfo` record that
carries everything a user (or the documentation, the report and the metric
recommender) needs to know about it: family, range, better direction, units,
what inputs it requires, and the primary reference.

Keeping the metadata next to the implementation is deliberate: the metric
catalogue in the docs, ``segevalkit metrics`` on the command line and the
direction arrows on every plot are all generated from this one registry, so
they can never drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

__all__ = [
    "FAMILIES",
    "MetricInfo",
    "register_metric",
    "get_metric",
    "list_metrics",
    "resolve_metrics",
    "METRIC_SETS",
]

#: Metric families, in the order they are presented in docs and reports.
FAMILIES: Dict[str, str] = {
    "overlap": "Overlap (voxel counting)",
    "volume": "Volume agreement",
    "distance": "Boundary & surface distance",
    "topology": "Topology & centreline",
    "detection": "Detection & instance (lesion-wise)",
    "calibration": "Probabilistic & calibration",
    "agreement": "Agreement & information theory",
}


@dataclass(frozen=True)
class MetricInfo:
    """Metadata describing one metric.

    Attributes:
        name: Registry key, ``snake_case`` (e.g. ``"hd95"``).
        display: Human-readable name.
        abbr: Short label used on plots and tables.
        family: One of `FAMILIES`.
        better: ``"higher"``, ``"lower"``, ``"zero"`` (signed metrics whose
            ideal value is 0, e.g. relative volume difference) or ``"none"``
            (descriptive quantities such as a volume, which are not scores).
        value_range: ``(low, high)``; ``float("inf")`` for unbounded.
        unit: Unit string (``""`` for dimensionless, ``"mm"``, ``"mL"``...).
        requires: Extra inputs: ``"spacing"`` (physical voxel size matters),
            ``"probabilities"`` (needs a soft prediction).
        summary: One-sentence plain-language description.
        reference: Primary reference, short form.
        defaults: Default keyword parameters (e.g. ``{"tolerance_mm": 1.0}``).
        fn: The implementation.
    """

    name: str
    display: str
    abbr: str
    family: str
    better: str
    value_range: Tuple[float, float]
    unit: str
    requires: Tuple[str, ...]
    summary: str
    reference: str
    defaults: Mapping[str, Any] = field(default_factory=dict)
    fn: Optional[Callable[..., float]] = field(default=None, compare=False, repr=False)

    @property
    def higher_is_better(self) -> Optional[bool]:
        """``True``/``False`` for monotone metrics, ``None`` for signed ones."""
        return {"higher": True, "lower": False}.get(self.better)

    @property
    def arrow(self) -> str:
        """Direction glyph for plot labels: ↑, ↓ or →0."""
        return {"higher": "↑", "lower": "↓", "zero": "→0", "none": ""}[self.better]

    @property
    def label(self) -> str:
        """Axis label, e.g. ``"HD95 [mm] ↓"``."""
        unit = f" [{self.unit}]" if self.unit else ""
        return f"{self.abbr}{unit} {self.arrow}".rstrip()

    def __call__(self, ctx, **params) -> float:
        kwargs = {**self.defaults, **params}
        return self.fn(ctx, **kwargs)


_REGISTRY: Dict[str, MetricInfo] = {}


def register_metric(
    name: str,
    *,
    display: str,
    abbr: str,
    family: str,
    better: str,
    value_range: Tuple[float, float] = (0.0, 1.0),
    unit: str = "",
    requires: Iterable[str] = (),
    summary: str = "",
    reference: str = "",
    defaults: Optional[Mapping[str, Any]] = None,
) -> Callable[[Callable[..., float]], Callable[..., float]]:
    """Decorator that registers a metric implementation under ``name``."""
    if family not in FAMILIES:
        raise ValueError(f"unknown metric family {family!r}")
    if better not in ("higher", "lower", "zero", "none"):
        raise ValueError(f"better must be 'higher', 'lower', 'zero' or 'none', got {better!r}")

    def decorator(fn: Callable[..., float]) -> Callable[..., float]:
        if name in _REGISTRY:
            raise ValueError(f"metric {name!r} registered twice")
        _REGISTRY[name] = MetricInfo(
            name=name,
            display=display,
            abbr=abbr,
            family=family,
            better=better,
            value_range=tuple(value_range),
            unit=unit,
            requires=tuple(requires),
            summary=summary,
            reference=reference,
            defaults=dict(defaults or {}),
            fn=fn,
        )
        return fn

    return decorator


def get_metric(name: str) -> MetricInfo:
    """Return the `MetricInfo` registered under ``name`` (or an alias)."""
    key = _ALIASES.get(name.lower(), name.lower())
    try:
        return _REGISTRY[key]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"unknown metric {name!r}. Known metrics: {known}") from None


def list_metrics(family: Optional[str] = None) -> List[MetricInfo]:
    """All registered metrics, ordered by family then registration order."""
    order = list(FAMILIES)
    items = [m for m in _REGISTRY.values() if family is None or m.family == family]
    return sorted(items, key=lambda m: order.index(m.family))


#: Common alternative spellings.
_ALIASES: Dict[str, str] = {
    "dsc": "dice",
    "f1": "dice",
    "jaccard": "iou",
    "jac": "iou",
    "sensitivity": "recall",
    "tpr": "recall",
    "ppv": "precision",
    "tnr": "specificity",
    "hausdorff": "hd",
    "hd100": "hd",
    "assd": "assd",
    "asd": "assd",
    "surface_dice": "nsd",
    "normalized_surface_dice": "nsd",
    "cldsc": "cldice",
    "kappa": "cohen_kappa",
    "rvd": "relative_volume_difference",
    "avd": "absolute_volume_difference",
    "vs": "volumetric_similarity",
    "voe": "voe",
    "pq": "panoptic_quality",
    "vi": "variation_of_information",
    "ari": "adjusted_rand_index",
}


#: Named metric bundles accepted wherever a metric list is expected.
METRIC_SETS: Dict[str, List[str]] = {
    # A compact, broadly-recommended default: one overlap, one boundary, one
    # robust distance, plus volume and the raw confusion rates.
    "default": ["dice", "iou", "nsd", "hd95", "assd", "precision", "recall",
                "relative_volume_difference"],
    "overlap": ["dice", "iou", "precision", "recall", "specificity", "voe",
                "fbeta", "tversky", "mcc", "cohen_kappa", "balanced_accuracy"],
    "distance": ["hd", "hd95", "assd", "masd", "nsd", "boundary_iou"],
    "volume": ["pred_volume", "ref_volume", "absolute_volume_difference",
               "relative_volume_difference", "volumetric_similarity"],
    "topology": ["cldice", "betti0_error", "betti1_error", "betti2_error",
                 "euler_error"],
    "detection": ["lesion_precision", "lesion_recall", "lesion_f1",
                  "panoptic_quality", "lesionwise_dice", "lesionwise_hd95", "lesionwise_nsd",
                  "lesion_ap", "lesion_count_difference", "false_positive_lesions",
                  "false_negative_lesions", "split_count", "merge_count"],
    "calibration": ["ece", "brier", "nll", "auroc", "auprc", "soft_dice"],
    "agreement": ["cohen_kappa", "mcc", "adjusted_rand_index",
                  "variation_of_information", "mutual_information",
                  "global_consistency_error"],
}


def resolve_metrics(spec: Iterable[str] | str) -> List[str]:
    """Expand a metric spec (names, aliases, set names, ``"all"``) to registry keys.

    Examples:
        >>> resolve_metrics(["default", "cldice"])[:3]
        ['dice', 'iou', 'nsd']
    """
    if isinstance(spec, str):
        spec = [s.strip() for s in spec.split(",") if s.strip()]
    out: List[str] = []
    for item in spec:
        key = item.lower()
        if key == "all":
            names = [m.name for m in list_metrics()]
        elif key == "all_binary":
            names = [m.name for m in list_metrics() if "probabilities" not in m.requires]
        elif key in METRIC_SETS:
            names = METRIC_SETS[key]
        else:
            names = [get_metric(key).name]
        out.extend(n for n in names if n not in out)
    return out
