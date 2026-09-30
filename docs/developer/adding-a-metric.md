# Adding a metric

A metric is a function `fn(ctx, **params) -> float` registered with metadata. This page adds **surface recall at
tolerance** end to end: the fraction of the reference surface within τ mm of the prediction (the reference-side half
of NSD, for when missing boundary costs more than extra boundary).

## 1. Implement and register

```python
import numpy as np
from segevalkit import register_metric, compute_metrics, PairContext


@register_metric(
    "surface_recall",                        # registry key (snake_case, unique)
    display="Surface recall at tolerance",   # human-readable name
    abbr="SR",                               # short label for plots and tables
    family="distance",                       # one of FAMILIES
    better="higher",                         # higher | lower | zero | none
    value_range=(0.0, 1.0),
    requires=("spacing",),                   # the value depends on voxel size
    summary="Fraction of the reference surface lying within tau mm "
            "of the predicted surface.",
    reference="Nikolov et al. 2021, J Med Internet Res 23(7):e26151 "
              "(surface overlap)",
    defaults={"tolerance_mm": 2.0},
)
def surface_recall(ctx: PairContext, tolerance_mm: float = 2.0) -> float:
    r"""$$\mathrm{SR}_\tau = \frac{|\{d \in D_{G\to P} : d \le \tau\}|}{|\partial G|}$$"""
    if ctx.both_empty:                       # structure correctly absent
        return ctx.best_or_nan(1.0)          # 1.0 or NaN, per the EmptyPolicy
    if ctx.one_empty:
        return 0.0
    _, d_ref_to_pred = ctx.surface_distances  # cached, shared with HD95/NSD/ASSD
    return float(np.mean(d_ref_to_pred <= tolerance_mm))
```

The decorator validates family and direction, stores a `MetricInfo`, and exposes the metric to `compute_metrics`,
`Evaluator`, the CLI (`--metrics surface_recall`), plot labels (`SR ↑`), the report glossary and the
[catalogue](../metrics/catalogue.md).

## 2. Use it

```python
g = np.zeros((48, 48, 48), bool); g[12:36, 12:36, 12:36] = True
p = np.zeros_like(g);             p[15:36, 12:36, 12:36] = True  # 3 voxels off one face

compute_metrics(p, g, ["surface_recall", "nsd", "hd95"], spacing=(1, 1, 1),
                params={"surface_recall": {"tolerance_mm": 2.0}})
```

```text
{'surface_recall': 0.818639798488665, 'nsd': 0.8518762343647136, 'hd95': 3.0}
```

Surface recall is below NSD because the missing face lies on the *reference* side, as intended.

## 3. Implementation rules

| Rule | How |
|---|---|
| **Reuse cached intermediates** | `ctx.counts`, `ctx.surface_distances`, `ctx.pred_components`, `ctx.pred_skeleton`, `ctx.pred_cropped`, ... never recompute an EDT. |
| **Cache your own** | `ctx.memo(("my_key", param), lambda: expensive(ctx))` shares work between your metrics. |
| **Decide empty masks explicitly** | `ctx.both_empty` → `ctx.best_or_nan(best)`; one-empty distance → `ctx.distance_penalty()`; undefined ratios → `float("nan")`. |
| **Physical units** | use `ctx.spacing` (mm) and `ctx.voxel_volume_mm3`; declare `requires=("spacing",)` and the `unit`. |
| **Probabilities** | read `ctx.prob` and declare `requires=("probabilities",)`; the evaluator skips the metric when no map is given. |
| **Pure function** | no global state, no I/O, deterministic for a fixed input. |
| **Equation in the docstring** | a `$$...$$` block; it appears in the report glossary. |

## 4. Test it

Use inputs with hand-computable answers, the empty cases and invariances:

```python title="tests/test_surface_recall.py"
import numpy as np
import pytest
from segevalkit.metrics import compute_metrics, EmptyPolicy

def test_identical_is_one():
    g = np.zeros((20, 20, 20), bool); g[5:15, 5:15, 5:15] = True
    assert compute_metrics(g, g, ["surface_recall"])["surface_recall"] == 1.0

def test_empty_cases():
    z = np.zeros((10, 10, 10), bool); g = z.copy(); g[2:5, 2:5, 2:5] = True
    assert compute_metrics(z, g, ["surface_recall"])["surface_recall"] == 0.0
    nan_policy = EmptyPolicy("nan")
    assert np.isnan(compute_metrics(z, z, ["surface_recall"], empty=nan_policy)["surface_recall"])

def test_tolerance_monotone():
    g = np.zeros((30, 30, 30), bool); g[5:25, 5:25, 5:25] = True
    p = np.roll(g, 3, 0)
    vals = [compute_metrics(p, g, ["surface_recall"],
                            params={"surface_recall": {"tolerance_mm": t}})["surface_recall"]
            for t in (0, 1, 2, 3)]
    assert vals == sorted(vals) and vals[-1] == 1.0
```

If a reference implementation exists (MONAI, MedPy, DeepMind, ...), add a conformance test to
`tests/test_reference_implementations.py` and document convention differences in
[Conventions](../guide/conventions.md).

## 5. Contribute it to the library

1. Put the function in the module of its family (`src/segevalkit/metrics/distance.py` here).
2. Add it to a [metric set](../metrics/index.md) in `METRIC_SETS` if it belongs to one.
3. Add aliases in `_ALIASES` if the metric is known under other names.
4. Write its card on the family page (`docs/metrics/distance.md`), following the existing cards.
5. If it changes a recommendation, update `segevalkit.guide.recommend` and its tests.
6. Add an entry to the [changelog](../about/changelog.md).

!!! tip "Keeping it local"
    A metric registered in your own module (imported before evaluation) behaves exactly like a built-in one; no
    fork is needed.
