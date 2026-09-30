"""SegEvalKit: holistic evaluation of volumetric (CT/MR) medical image segmentation.

Examples:
    Quick start:

    >>> import segevalkit as sek
    >>> ev = sek.Evaluator(labels={"liver": 1, "tumour": 2}, metrics="default")
    >>> res = ev.evaluate("predictions/", "labelsTr/")
    >>> res.summary()
    >>> sek.plotting.metric_distribution(res, "dice")

Sub-packages:

* `segevalkit.metrics`: 50+ metrics with a shared, cached computation context.
* `segevalkit.io`: NIfTI/ITK/NumPy loading with geometry checks, label specs, case discovery.
* `segevalkit.stats`: bootstrap CIs, paired tests, rankings, Bland-Altman, ICC.
* `segevalkit.plotting`: publication-ready quantitative plots.
* `segevalkit.viz`: qualitative visualisation (overlays, error maps, 3D surface distance).
* `segevalkit.guide`: which metrics to use for which question.
* `segevalkit.datasets`: presets for public benchmarks (labels + official metrics).
* `segevalkit.synthetic`: controlled perturbations for metric sensitivity studies.
* `segevalkit.report`: a self-contained HTML evaluation report.
"""

__version__ = "0.1.0"

from .metrics import (  # noqa: E402
    EmptyPolicy,
    PairContext,
    compute_metrics,
    get_metric,
    list_metrics,
    register_metric,
    resolve_metrics,
)
from .evaluator import Evaluator  # noqa: E402
from .results import EvaluationResult, load_results  # noqa: E402

__all__ = [
    "__version__",
    "Evaluator",
    "EvaluationResult",
    "load_results",
    "EmptyPolicy",
    "PairContext",
    "compute_metrics",
    "get_metric",
    "list_metrics",
    "register_metric",
    "resolve_metrics",
]


def __getattr__(name):
    # Heavy optional sub-packages (matplotlib, jinja2) load on first access.
    import importlib

    if name in {"plotting", "viz", "stats", "guide", "datasets", "synthetic", "report", "io"}:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module 'segevalkit' has no attribute {name!r}")
