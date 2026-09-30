# Metrics

SegEvalKit registers 53 metrics in seven families. Each family answers a different question about a
segmentation, and each is blind to something the others see. No single number describes segmentation quality;
Metrics Reloaded (Maier-Hein et al. 2024) recommends combining **at least one overlap metric with one boundary
metric**, and adding detection metrics whenever the task has several instances.

Every metric is a plain function registered together with its metadata: registry key, display name,
abbreviation, family, better direction, range, unit, required inputs, summary, reference and default
parameters. The [full catalogue](catalogue.md), the command-line listing (`segevalkit metrics -v`), plot axis
labels and the HTML report are all generated from this single registry, so they cannot drift apart. The family
pages below give the definition, interpretation, pitfalls and exact empty-mask behaviour of each metric.

## The seven families

[Overlap](overlap.md)
:   Counts correctly and incorrectly labelled voxels (Dice, IoU, precision, recall, F\(_\beta\), specificity...).
    Cheap, bounded and spacing-free. **Misses** where the errors are, boundary quality and topology; strongly
    size-dependent; TN-based members are meaningless in background-dominated 3D volumes.

[Volume](volume.md)
:   Compares how much was segmented, in mL or relative to the reference. The endpoint for volumetry. **Misses**
    position entirely: a mask of the right size in the wrong place scores perfectly.

[Distance & boundary](distance.md)
:   Summarises the distances between the two surfaces in mm (HD, HD95, ASSD, MASD, NSD, Boundary IoU,
    centroid distance). **Misses** how much volume is wrong; the maximum-based members are dominated by single
    outlier voxels.

[Topology](topology.md)
:   Counts connected components, loops and cavities (Betti errors, Euler error) and compares centrelines
    (clDice). **Misses** location and overlap; different errors can cancel.

[Detection & instances](detection.md)
:   Counts lesions rather than voxels: lesion recall, precision and F1, splits and merges, panoptic quality,
    lesion-wise Dice. **Misses** boundary accuracy within detected lesions; depends on the connectivity and
    matching rule you choose.

[Calibration](calibration.md)
:   Scores the probability map itself: discrimination (AUROC, AUPRC) and calibration (ECE, Brier, NLL), plus
    soft Dice. **Misses** the geometry of the final mask; whole-volume values are dominated by easy background.

[Agreement](agreement.md)
:   Treats the masks as partitions: MCC, Cohen's kappa, mutual information, variation of information, adjusted
    Rand index, global consistency error. **Misses** little that Dice does not already capture for binary
    masks, while adding dependence on the field of view.

## At a glance

| Family | Question it answers | Blind spot | Typical pick |
|---|---|---|---|
| Overlap | How many voxels are right? | Where the wrong voxels are; small structures are penalised more | [`dice`](overlap.md#dice) |
| Volume | Is the amount right? | Position | [`absolute_volume_difference`](volume.md#absolute_volume_difference), [`relative_volume_difference`](volume.md#relative_volume_difference) |
| Distance | How far off is the boundary? | Volume of the error; single outliers (HD) | [`nsd`](distance.md#nsd), [`hd95`](distance.md#hd95) |
| Topology | Is the shape connected correctly? | Location and amount of error | [`cldice`](topology.md#cldice), [`betti0_error`](topology.md#betti0_error) |
| Detection | Were the lesions found, and only them? | Boundary quality of found lesions | [`lesion_f1`](detection.md#lesion_f1), [`panoptic_quality`](detection.md#panoptic_quality) |
| Calibration | Can the probabilities be trusted? | Mask geometry; background dilution | [`ece`](calibration.md#ece) with `roi="band"`, [`brier`](calibration.md#brier) |
| Agreement | Do the partitions agree beyond chance? | Adds field-of-view dependence to Dice-like information | [`cohen_kappa`](agreement.md#cohen_kappa) (inter-rater studies) |

For a structured recommendation from the properties of your task, see the
[decision guide](../guide/choosing.md); for what can go wrong, see [Pitfalls](../guide/pitfalls.md).

## Computing metrics {#computing-metrics}

### One pair: `compute_metrics`

`compute_metrics(pred, ref, metrics, *, spacing, prob, params, device, empty, strict, **context_kwargs)` scores
one binary pair. `metrics` accepts registry keys, aliases, [set names](#metric-sets) or `"all"`, as a list or a
comma-separated string. `params` maps a metric key to keyword overrides of its defaults.

```python
import numpy as np
from segevalkit.metrics import compute_metrics

ref = np.zeros((32, 32, 32), bool)
ref[8:24, 8:24, 8:24] = True             # a 16-voxel cube
pred = np.roll(ref, 2, axis=0)           # shifted by 2 voxels along the first axis

scores = compute_metrics(
    pred, ref,
    metrics=["default", "masd", "cldice"],   # a set name plus two extra metrics
    spacing=(0.8, 0.8, 2.0),                 # mm, one value per array axis
    params={"nsd": {"tolerance_mm": 1.0}},   # per-metric keyword overrides
)
print({k: round(v, 4) for k, v in scores.items()})
```

```text
{'dice': 0.875, 'iou': 0.7778, 'nsd': 0.6864, 'hd95': 1.6, 'assd': 0.5538, 'precision': 0.875, 'recall': 0.875, 'relative_volume_difference': 0.0, 'masd': 0.5538, 'cldice': 1.0}
```

Inputs are any arrays of equal shape; non-zero means foreground and 2D arrays are treated as one slice. Without
`spacing` the voxels are taken as 1 mm isotropic, so distances are then in voxel units. Metrics that need a
probability map are silently skipped when `prob` is not given. A metric that raises is recorded as NaN with a
`RuntimeWarning`, unless `strict=True`. Extra keyword arguments (`connectivity`, `min_component_voxels`) go to
the `PairContext`.

### Caching: `PairContext`

Most metrics share expensive intermediates: confusion counts, surfaces, directed surface distances, connected
components, skeletons, Betti numbers and lesion matchings. A `PairContext` computes each of them lazily, once,
and caches it, so asking for twenty metrics costs little more than asking for the most expensive one.
`compute_metrics` creates a context internally; create one yourself to reuse it across calls (passing `metrics`
by keyword is clearest, though `compute_metrics(ctx, ["hd95"])` also works):

```python
from segevalkit.metrics import PairContext

ctx = PairContext(pred, ref, spacing=(0.8, 0.8, 2.0))
print(ctx)
first = compute_metrics(ctx, metrics=["hd95", "assd"])            # computes the surface distances
second = compute_metrics(ctx, metrics=["nsd", "masd"],             # reuses them
                         params={"nsd": {"tolerance_mm": 2.0}})
print({k: round(v, 4) for k, v in {**first, **second}.items()})
```

```text
PairContext(shape=(32, 32, 32), spacing=(0.8, 0.8, 2.0), pred_voxels=4096, ref_voxels=4096, device='cpu')
{'hd95': 1.6, 'assd': 0.5538, 'nsd': 1.0, 'masd': 0.5538}
```

`PairContext` also fixes the conventions: surfaces are 6-connected boundary voxels (the image border counts as
background), directed distances are exact Euclidean distances in mm, lesions are 26-connected components by
default, and empty masks follow an explicit [`EmptyPolicy`](#empty-masks). With `device="cuda"`, confusion counts
and surface distances run in PyTorch and give the same values.

### Datasets and multi-label volumes: `Evaluator`

`Evaluator` applies the same metrics to every label of every case, reads images and spacing from disk, checks
that prediction and reference grids agree, and returns a tidy `EvaluationResult`. Every option that changes a
number is stored in the result's provenance.

```python
import numpy as np
from segevalkit import Evaluator, EmptyPolicy

ref = np.zeros((32, 32, 32), np.uint8)
ref[4:20, 4:20, 4:20] = 1                # label 1: "organ"
ref[22:28, 22:28, 22:28] = 2             # label 2: "lesion"
pred = np.roll(ref, 1, axis=1)

ev = Evaluator(
    labels={"organ": 1, "lesion": 2},
    metrics=["dice", "nsd", "hd95", "lesion_f1"],
    params={"nsd": {"tolerance_mm": 1.0}},
    empty=EmptyPolicy.preset("segevalkit"),
    connectivity=26,
)
res = ev.evaluate_arrays(pred, ref, spacing=(1.0, 1.0, 1.0), case_id="demo")
print(res.wide()[["case_id", "label", "dice", "nsd", "hd95", "lesion_f1"]])
```

```text
  case_id   label      dice  nsd  hd95  lesion_f1
0    demo  lesion  0.833333  1.0   1.0        1.0
1    demo   organ  0.937500  1.0   1.0        1.0
```

For folders of NIfTI files, call `ev.evaluate("preds/", "labelsTr/", n_workers=8)` and then `res.summary()`
(mean, median, IQR, bootstrap CI and the number of NaN values per label and metric) or `res.save("eval_out/")`.
Per-label parameter overrides can be given in the label specification. See the
[Quickstart](../getting-started/quickstart.md) and [Configuration](../getting-started/configuration.md).

## Metric sets {#metric-sets}

Named bundles are accepted wherever a metric list is expected (`metrics=["default", "cldice"]`,
`--metrics distance,detection`). Duplicates are removed and order is preserved.

| Set | Metrics |
|---|---|
| `default` | `dice`, `iou`, `nsd`, `hd95`, `assd`, `precision`, `recall`, `relative_volume_difference` |
| `overlap` | `dice`, `iou`, `precision`, `recall`, `specificity`, `voe`, `fbeta`, `tversky`, `mcc`, `cohen_kappa`, `balanced_accuracy` |
| `distance` | `hd`, `hd95`, `assd`, `masd`, `nsd`, `boundary_iou` |
| `volume` | `pred_volume`, `ref_volume`, `absolute_volume_difference`, `relative_volume_difference`, `volumetric_similarity` |
| `topology` | `cldice`, `betti0_error`, `betti1_error`, `betti2_error`, `euler_error` |
| `detection` | `lesion_precision`, `lesion_recall`, `lesion_f1`, `panoptic_quality`, `lesionwise_dice`, `lesion_count_difference`, `false_positive_lesions`, `false_negative_lesions`, `split_count`, `merge_count` |
| `calibration` | `ece`, `brier`, `nll`, `auroc`, `auprc`, `soft_dice` |
| `agreement` | `cohen_kappa`, `mcc`, `adjusted_rand_index`, `variation_of_information`, `mutual_information`, `global_consistency_error` |
| `all` | every registered metric |
| `all_binary` | every metric that does not need a probability map |

Sets are curated bundles, not the full family: `fpr`, `fnr`, `accuracy`, `volume_difference`, `hd_percentile`
and `centroid_distance` are available by name only. Common aliases are resolved everywhere, for example `dsc`
and `f1` → `dice`, `jaccard` → `iou`, `sensitivity` / `tpr` → `recall`, `ppv` → `precision`, `hausdorff` /
`hd100` → `hd`, `asd` → `assd`, `surface_dice` → `nsd`, `kappa` → `cohen_kappa`, `rvd`, `avd`, `vs`, `pq`, `vi`,
`ari`. `segevalkit.metrics.resolve_metrics(spec)` shows what a specification expands to.

## Empty masks {#empty-masks}

Many metrics are undefined when the prediction and/or the reference is empty, and existing tools disagree on
what to return: MetricsReloaded returns NaN and advises best or worst values at aggregation, MONAI can return
NaN or `inf`, DeepMind's `surface-distance` returns `inf`, MedPy raises, and BraTS 2023 substitutes fixed
values. SegEvalKit makes the choice explicit with an `EmptyPolicy`:

`both_empty`
:   What to return when **both** masks are empty (the structure is correctly absent). `"best"` (default) returns
    each metric's ideal value (Dice 1, HD 0...), the BraTS / KiTS convention. `"nan"` returns NaN so the case is
    excluded from aggregates, the nnU-Net convention.

`one_empty_distance`
:   The value of HD, HD\(_q\), HD95, ASSD, MASD and centroid distance when **exactly one** mask is empty.
    `"worst"` (default) uses the image diagonal in mm, \(\sqrt{\sum_k (n_k s_k)^2}\) for an image of
    \(n_k\) voxels of spacing \(s_k\) along axis \(k\): a finite, size-aware upper bound on any distance in
    the image. `"nan"` excludes the case; a number is used verbatim.

| Preset | `both_empty` | `one_empty_distance` | Convention |
|---|---|---|---|
| `segevalkit` (default) | `"best"` | `"worst"` (image diagonal) | |
| `brats2023` | `"best"` | 374.0 mm | BraTS 2023 lesion-wise code (SRI-24 atlas diagonal, rounded up) |
| `metrics_reloaded` | `"best"` | `"worst"` (image diagonal) | the aggregation advice in MetricsReloaded |
| `nan` (alias `nnunet`) | `"nan"` | `"nan"` | nnU-Net / MONAI `ignore_empty`: exclude undefined cases |

```python
import numpy as np
from segevalkit.metrics import compute_metrics, EmptyPolicy

ref = np.zeros((32, 32, 32), bool)                 # structure absent from the reference
pred = np.zeros_like(ref)
pred[4:8, 4:8, 4:8] = True                         # ...but the model predicts it

for name in ["segevalkit", "brats2023", "nan"]:
    out = compute_metrics(pred, ref, ["dice", "precision", "hd95", "nsd"],
                          spacing=(1, 1, 1), empty=EmptyPolicy.preset(name))
    print(f"{name:11s}", {k: round(v, 2) for k, v in out.items()})

both = compute_metrics(np.zeros_like(ref), ref, ["dice", "hd95"])   # correctly absent
print("both empty ", both)
```

```text
segevalkit  {'dice': 0.0, 'precision': 0.0, 'hd95': 55.43, 'nsd': 0.0}
brats2023   {'dice': 0.0, 'precision': 0.0, 'hd95': 374.0, 'nsd': 0.0}
nan         {'dice': 0.0, 'precision': 0.0, 'hd95': nan, 'nsd': 0.0}
both empty  {'dice': 1.0, 'hd95': 0.0}
```

Rules that hold independently of the policy:

- A ratio whose denominator is zero only because **one** mask is empty is genuinely undefined and returns NaN,
  never a silent 0: precision with an empty prediction, recall and RVD with an empty reference, lesion recall
  with no reference lesions, lesion precision with no predicted lesions.
- Bounded overlap-type scores (Dice, IoU, F\(_\beta\), Tversky, VS, NSD, Boundary IoU, clDice, lesion F1, PQ,
  lesion-wise Dice, MCC, kappa, ARI) are 0 when exactly one mask is empty.
- Some metrics never consult the policy because they are defined for empty masks: volumes and volume
  differences, specificity, FPR, accuracy, lesion counts, and split and merge counts (specificity and FPR are NaN
  only if the reference fills the whole image).
- MI, VI and GCE return 0 when both masks are empty under `"best"`, and NaN under `"nan"`. Calibration metrics return NaN when their
  region contains no voxels (or, for AUROC and AUPRC, no foreground).

The table below summarises the behaviour; "best / NaN" means the value depends on `both_empty`.

| Metrics | Both empty | Prediction empty only | Reference empty only |
|---|---|---|---|
| `dice`, `iou`, `fbeta`, `tversky`, `volumetric_similarity` | 1 / NaN | 0 | 0 |
| `voe` | 0 / NaN | 1 | 1 |
| `precision` | 1 / NaN | NaN | 0 |
| `recall` | 1 / NaN | 0 | NaN |
| `fnr` | 0 / NaN | 1 | NaN |
| `relative_volume_difference` | 0 / NaN | −1 | NaN |
| `balanced_accuracy` | 1 / NaN | 0.5 | NaN |
| `hd`, `hd_percentile`, `hd95`, `assd`, `masd`, `centroid_distance` | 0 / NaN | penalty | penalty |
| `nsd`, `boundary_iou`, `cldice` | 1 / NaN | 0 | 0 |
| `betti0_error` ... `euler_error` | 0 / NaN | \(\beta_k(G)\) | \(\beta_k(P)\) |
| `mcc`, `cohen_kappa`, `adjusted_rand_index` | 1 / NaN | 0 | 0 |
| `mutual_information`, `variation_of_information`, `global_consistency_error` | 0 / NaN | computed | computed |
| `lesion_recall` | 1 / NaN | 0 | NaN |
| `lesion_precision` | 1 / NaN | NaN | 0 |
| `lesion_f1`, `panoptic_quality`, `lesionwise_dice` | 1 / NaN | 0 | 0 |

For detection metrics "empty" means "no connected components" after `min_component_voxels` filtering. The
`Evaluator` also records `_ref_empty` and `_pred_empty` flags for every case and label, and `summary()` reports
the number of NaN values, so excluded cases are always visible.

!!! warning "Choose the policy before looking at results"
    The policy can change a mean substantially (see [Pitfalls](../guide/pitfalls.md#empty-references)). With
    `"nan"`, a model is not penalised for hallucinating a structure that is absent, as far as distance metrics
    are concerned. Report the policy, the number of empty-reference cases, and, where references can be empty,
    presence detection at the case level (`segevalkit.stats.presence_detection`).

## Adding a custom metric {#custom-metrics}

Register a function `fn(ctx, **params) -> float` with `@register_metric`. It then works everywhere a built-in
metric does: `compute_metrics`, `Evaluator`, the command line, plots (direction arrow and unit in the axis label)
and reports. Read cached intermediates from the `PairContext` (`ctx.tp`, `ctx.fp`, `ctx.fn`, `ctx.tn`,
`ctx.surface_distances`, `ctx.pred_components`, `ctx.ref_skeleton`...), honour the `EmptyPolicy` through
`ctx.both_empty`, `ctx.one_empty`, `ctx.best_or_nan(best)` and `ctx.distance_penalty()`, and cache any expensive
intermediate of your own with `ctx.memo(key, compute)`.

```python
import numpy as np
from segevalkit.metrics import register_metric, compute_metrics, get_metric
from segevalkit.metrics.context import PairContext


@register_metric(
    "surface_recall",
    display="Surface recall at tolerance",
    abbr="SR",
    family="distance",
    better="higher",
    value_range=(0.0, 1.0),
    requires=("spacing",),
    summary="Fraction of the reference surface within tolerance_mm of the predicted surface.",
    reference="Nikolov et al. 2021, J Med Internet Res 23(7):e26151",
    defaults={"tolerance_mm": 2.0},
)
def surface_recall(ctx: PairContext, tolerance_mm: float = 2.0) -> float:
    r"""$$\mathrm{SR}_\tau = |\{d\in D_{G\to P}: d\le\tau\}| \,/\, |\partial G|$$"""
    if ctx.both_empty:
        return ctx.best_or_nan(1.0)          # honour the EmptyPolicy
    if ctx.one_empty:
        return 0.0
    _, d_ref_to_pred = ctx.surface_distances  # cached, shared with HD95/ASSD/NSD
    return float(np.mean(d_ref_to_pred <= tolerance_mm))


ref = np.zeros((32, 32, 32), bool); ref[8:24, 8:24, 8:24] = True
pred = np.roll(ref, 2, axis=0)
print(get_metric("surface_recall").label)
print(compute_metrics(pred, ref, ["surface_recall", "nsd"], spacing=(1, 1, 1),
                      params={"surface_recall": {"tolerance_mm": 1.0}, "nsd": {"tolerance_mm": 1.0}}))
```

```text
SR ↑
{'surface_recall': 0.7041420118343196, 'nsd': 0.7041420118343196}
```

Rules enforced by the registry:

- `family` must be one of `overlap`, `volume`, `distance`, `topology`, `detection`, `calibration`, `agreement`.
- `better` must be `"higher"`, `"lower"`, `"zero"` (signed metrics whose ideal is 0) or `"none"` (descriptive
  quantities).
- A key can be registered only once; re-running the registration in the same session raises `ValueError`.
- `requires=("probabilities",)` makes `compute_metrics` and `Evaluator` skip the metric when no probability map
  is given.
- `defaults` are merged with the per-metric `params` at call time.

The metric must be registered in every process that evaluates. With `Evaluator(n_workers > 1)` on a CPU the
workers are forked and inherit the registration. With a GPU device they are spawned and re-import your code, so
register the metric at module level (in your script or in a module it imports), not inside a function or an
`if __name__ == "__main__":` block.
