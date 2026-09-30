"""Quantitative plotting in the SegEvalKit purple theme.

All functions return a matplotlib ``Figure``; save with
``fig.savefig("plot.png")``. See `segevalkit.plotting.theme` for the
palette and its colour-vision-deficiency validation.
"""

from .quantitative import (
    bland_altman_plot,
    comparison_forest,
    detection_by_size,
    ecdf,
    failure_quadrants,
    metric_correlation,
    metric_distribution,
    metric_heatmap,
    metric_profile,
    metric_vs_size,
    ranking_stability_plot,
    reliability_diagram,
    sensitivity_curves,
    volume_agreement,
)
from .theme import CATEGORICAL, ERROR_COLORS, PURPLE, apply_theme, theme

__all__ = [
    "apply_theme",
    "theme",
    "CATEGORICAL",
    "ERROR_COLORS",
    "PURPLE",
    "bland_altman_plot",
    "comparison_forest",
    "detection_by_size",
    "ecdf",
    "failure_quadrants",
    "metric_correlation",
    "metric_distribution",
    "metric_heatmap",
    "metric_profile",
    "metric_vs_size",
    "ranking_stability_plot",
    "reliability_diagram",
    "sensitivity_curves",
    "volume_agreement",
]
