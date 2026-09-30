# Metric pitfalls

A metric value is only meaningful together with the properties of the task, the data and the way it was
computed and aggregated. This page catalogues the pitfalls that most often distort 3D segmentation results. It
follows the taxonomy of Reinke et al. (2024), the companion paper of Metrics Reloaded, which groups pitfalls into
poor metric selection (P2: ignoring properties of the target structures, the data set or the algorithm output)
and poor metric application (P3: implementation, aggregation, ranking and reporting).

Each entry states what goes wrong, gives a small numeric example computed with SegEvalKit, and says how to avoid
it. The examples use 1 mm isotropic voxels unless stated otherwise.

| # | Pitfall | Affects | First line of defence |
|---|---|---|---|
| 1 | [Size and small-structure bias](#size-bias) | Dice, IoU, RVD | stratify by size; add NSD or lesion-wise metrics |
| 2 | [Class imbalance and TN-based metrics](#class-imbalance) | accuracy, specificity, AUROC, kappa, MCC | use TN-free metrics |
| 3 | [Boundary, volume and topology blindness](#boundary-volume) | every single metric | one overlap + one boundary metric (+ topology) |
| 4 | [Outlier sensitivity of the Hausdorff distance](#outliers) | HD | HD95 or NSD; HD only as a safety bound |
| 5 | [Empty references](#empty-references) | Dice, HD, most ratios | explicit `EmptyPolicy`; case-level presence detection |
| 6 | [Aggregation](#aggregation) | every mean | per-case, per-structure reporting; count NaN |
| 7 | [Anisotropic spacing](#anisotropic-spacing) | distances, volumes, BIoU, clDice | always pass spacing |
| 8 | [Connectivity and lesion definition](#connectivity) | detection metrics | declare connectivity, matching and minimum size |
| 9 | [Tolerance choice for NSD](#nsd-tolerance) | NSD, BIoU | fix \(\tau\) from inter-rater variability |
| 10 | [Ranking instability](#ranking) | challenge and paper rankings | bootstrap rankings; report the scheme |
| 11 | [Calibration in background-dominated volumes](#calibration-background) | ECE, Brier, NLL, AUROC | `roi="band"` |
| 12 | [Convention mismatches between tools](#conventions) | HD95, ASSD, NSD, lesion Dice | state the convention; use conformance presets |

## 1. Size and small-structure bias {#size-bias}

**What goes wrong.** Overlap metrics normalise by the size of the structure, so the same boundary error costs a
small structure far more than a large one. Dice therefore mixes segmentation quality with structure size, and a
mean Dice over structures (or lesions) of different size is not comparable (Reinke et al. 2024, P2: small
structure sizes, high variability of structure sizes).

**Example.** A one-voxel shift of a cube:

```python
import numpy as np
from segevalkit.metrics import compute_metrics

for edge in (3, 6, 30):
    ref = np.zeros((40, 40, 40), bool)
    ref[5:5 + edge, 5:5 + edge, 5:5 + edge] = True
    pred = np.roll(ref, 1, axis=0)                  # one-voxel shift
    s = compute_metrics(pred, ref, ["dice", "hd95", "nsd"], spacing=(1, 1, 1),
                        params={"nsd": {"tolerance_mm": 1.0}})
    print(edge, {k: round(v, 3) for k, v in s.items()})
```

```text
3 {'dice': 0.667, 'hd95': 1.0, 'nsd': 1.0}
6 {'dice': 0.833, 'hd95': 1.0, 'nsd': 1.0}
30 {'dice': 0.967, 'hd95': 1.0, 'nsd': 1.0}
```

The error is identical (1 mm everywhere, well within a 1 mm tolerance), but Dice ranges from 0.667 to 0.967.
Relative volume difference behaves the same way: 3 extra voxels on a 2-voxel lesion is RVD = +1.5.

**How to avoid it.**

- Report [Dice](../metrics/overlap.md#dice) per structure, never averaged across structures of very different size.
- Stratify by reference volume: `segevalkit.stats.stratify(result, "dice", by="ref_volume_ml")`.
- Add a boundary metric in mm, such as [NSD](../metrics/distance.md#nsd) or [HD95](../metrics/distance.md#hd95),
  whose value does not depend on object size.
- For small lesions, report volume errors in mL ([AVD](../metrics/volume.md#absolute_volume_difference)) rather
  than relative, and use [lesion-wise metrics](../metrics/detection.md).

## 2. Class imbalance and TN-based metrics {#class-imbalance}

**What goes wrong.** In a 3D volume the background outnumbers the foreground by \(10^3\) to \(10^6\). Every metric
that uses true negatives is dominated by them: accuracy and specificity are close to 1 for almost any prediction,
voxel-wise AUROC is close to 1 for almost any model, and kappa, MCC, ARI, MI and GCE change when the image is
cropped (Reinke et al. 2024, P2: high class imbalance).

**Example.** A \(64^3\) volume with a 64-voxel lesion that is missed entirely, and a 64-voxel false positive
elsewhere:

| Metric | Value |
|---|---|
| [Dice](../metrics/overlap.md#dice) | 0.0000 |
| [Accuracy](../metrics/overlap.md#accuracy) | 0.9995 |
| [Specificity](../metrics/overlap.md#specificity) | 0.9998 |
| [Balanced accuracy](../metrics/overlap.md#balanced_accuracy) | 0.4999 |
| [MCC](../metrics/agreement.md#mcc) | −0.0002 |

A related effect hits information measures: a *perfect* prediction of the same 64-voxel lesion has
[mutual information](../metrics/agreement.md#mutual_information) 0.0023 nats, against 0.693 nats for a perfect
prediction of a structure filling half the volume.

**How to avoid it.** Use TN-free metrics (Dice, IoU, precision, recall, F\(_\beta\), distances) as primary
endpoints. Treat accuracy, specificity and whole-volume AUROC as sanity checks only. For probabilistic outputs
prefer [AUPRC](../metrics/calibration.md#auprc) to AUROC, and restrict to a band around the object (see
[pitfall 11](#calibration-background)).

## 3. Boundary, volume and topology blindness {#boundary-volume}

**What goes wrong.** Each family sees one aspect of the error and is blind to the others (Reinke et al. 2024,
P2: disregard of the domain interest).

**Examples.**

*Volume metrics are blind to position.* Two equal-sized cubes that do not touch:
[VS](../metrics/volume.md#volumetric_similarity) = 1, [RVD](../metrics/volume.md#relative_volume_difference) = 0,
[AVD](../metrics/volume.md#absolute_volume_difference) = 0 mL, Dice = 0.

*Overlap metrics are blind to where the error is.* Two predictions of a 4096-voxel cube with the same Dice:

| Prediction | Dice | HD95 (mm) | ASSD (mm) | NSD (τ = 2 mm) | lesion precision |
|---|---|---|---|---|---|
| shifted by 2 voxels | 0.875 | 2.0 | 0.675 | 1.000 | 1.0 |
| 2 slices missing + a separate 512-voxel blob about 22 mm away | 0.875 | 30.8 | 3.279 | 0.897 | 0.5 |

*Overlap and distance are blind to topology.* A 2-voxel gap cut into a 56-voxel vessel: Dice 0.982, HD95 0 mm,
NSD 1.0, clDice 0.982, but [β₀ error](../metrics/topology.md#betti0_error) = 1 (the vessel is in two pieces).
Filling the lumen of a ring gives Dice 0.80 and [β₁ error](../metrics/topology.md#betti1_error) = 1.

**How to avoid it.** Report at least one [overlap](../metrics/overlap.md) and one
[boundary](../metrics/distance.md) metric (the `default` metric set does this). Add
[volume](../metrics/volume.md) metrics for volumetry endpoints, [topology](../metrics/topology.md) metrics for
tubular or hollow structures, and [detection](../metrics/detection.md) metrics when there are several instances.
The [decision guide](choosing.md) maps task properties to metric sets.

## 4. Outlier sensitivity of the Hausdorff distance {#outliers}

**What goes wrong.** [HD](../metrics/distance.md#hd) is a maximum, so a single stray voxel far from the object
sets its value (Reinke et al. 2024, P2: spatial outliers). Conversely, noisy reference annotations with isolated
voxels make HD large for every method.

**Example.** A perfect prediction of a 20 mm cube plus one false-positive voxel about 54 mm away:

| Dice | HD | HD95 | ASSD | NSD (τ = 2 mm) |
|---|---|---|---|---|
| 0.9999 | 53.69 mm | 0.00 mm | 0.012 mm | 0.9998 |

One voxel out of 8,001 moves HD from 0 to 53.7 mm.

**How to avoid it.** Use [HD95](../metrics/distance.md#hd95) or [NSD](../metrics/distance.md#nsd) as the boundary
endpoint. Keep HD only where a worst-case bound is the clinical question (radiotherapy organs at risk), and report
it next to a robust metric. Consider `min_component_voxels` only if tiny components are noise by definition of
the task; it applies to detection metrics, not to HD.

## 5. Empty references {#empty-references}

**What goes wrong.** When a structure is absent from the reference (no enhancing tumour, post-operative scans,
healthy controls), Dice is 0/0 if the prediction is also empty and 0 otherwise; HD is undefined in both cases.
Tools silently disagree: NaN, `inf`, exceptions, 1, or a fixed penalty (Reinke et al. 2024, P2: occurrence of
cases with an empty reference; possibility of empty prediction). The choice can dominate the dataset mean.

**Example.** Four cases: A (Dice 0.90, HD95 3 mm), B (Dice 0.85, HD95 5 mm), C (structure correctly absent), and
D (structure absent, but the model predicts a false positive), on a \(512\times512\times100\) CT at
\(0.8\times0.8\times2.5\) mm, whose diagonal is 630.9 mm:

| Policy | Dice C | Dice D | Mean Dice | HD95 C | HD95 D | Mean HD95 |
|---|---|---|---|---|---|---|
| `segevalkit` (best / image diagonal) | 1 | 0 | 0.688 | 0 | 630.9 | 159.7 mm |
| `nan` (exclude undefined) | NaN | 0 | 0.583 | NaN | NaN | 4.0 mm |

Under `"nan"` the hallucinated structure in case D has no effect on the mean HD95, and case C, a correct call,
disappears from both means.

**How to avoid it.**

- Choose an [`EmptyPolicy`](../metrics/index.md#empty-masks) before evaluation and report it. SegEvalKit never
  returns a silent 0 for an undefined ratio, and returns a finite, size-aware penalty for one-sided distance
  cases by default; `EmptyPolicy.preset("brats2023")` reproduces the 374 mm BraTS convention.
- Report the number of empty-reference cases and of NaN values (`EvaluationResult.summary()` has an `n_nan`
  column; every case carries `_ref_empty` and `_pred_empty` flags).
- Where references can be empty, evaluate presence detection at the case level
  (`segevalkit.stats.presence_detection`), as Metrics Reloaded recommends.

## 6. Aggregation {#aggregation}

**What goes wrong.** How per-case values become one number can change the conclusion (Reinke et al. 2024, P3:
inadequate aggregation).

**Examples.**

*Mean over structures of different size.* Liver Dice 0.96 (1.5 million voxels) and tumour Dice 0.40
(10,000 voxels). The mean over structures is 0.68 and hides a failing tumour model behind the liver; a "global"
Dice pooled over both structures is 0.956 and hides it completely.

*NaN exclusion bias.* Excluding undefined cases removes exactly the cases where something unusual happened. In
[pitfall 5](#empty-references) the mean HD95 drops from 159.7 mm to 4.0 mm by exclusion alone.

*Cancelling signed errors.* Signed metrics ([RVD](../metrics/volume.md#relative_volume_difference),
[ΔV](../metrics/volume.md#volume_difference)) average towards 0: RVD of +0.2 on half the cases and −0.2 on the
other half has a mean of 0.

*Pooled versus per-case.* Pooling voxels (or lesions) over the dataset before computing a metric weights large
patients more than averaging per-case values; the two are different quantities.

*Hierarchy.* Several scans of one patient are not independent; bootstrap at the patient level.

**How to avoid it.** Compute metrics per case and per structure first, then aggregate; SegEvalKit's
`EvaluationResult` is organised that way. Report median and IQR next to the mean, plot distributions, count NaN
values, and report absolute values for signed metrics. Use bootstrap confidence intervals
(`EvaluationResult.summary()`) and paired tests; see [Statistics & ranking](../analysis/statistics.md).

## 7. Anisotropic spacing {#anisotropic-spacing}

**What goes wrong.** Distances and volumes are physical quantities. On anisotropic images (typical MR:
\(0.8\times0.8\times5\) mm) a one-slice error is 5 mm, and ignoring the spacing reports it as "1". Resampling
also changes voxel-count metrics, and skeleton-based metrics depend on the grid (Reinke et al. 2024, P3: unit
handling, discretisation issues).

**Example.** A \(16\times16\times8\) box shifted by one slice along the through-plane axis:

| Spacing (mm) | Dice | HD95 | ASSD | NSD (τ = 2 mm) | AVD |
|---|---|---|---|---|---|
| none (treated as 1 × 1 × 1) | 0.875 | 1.0 mm | 0.518 mm | 1.000 | 0 mL |
| 0.8 × 0.8 × 5.0 | 0.875 | 5.0 mm | 1.979 mm | 0.592 | 0 mL |

Dice is unchanged; every distance changes several-fold.

**How to avoid it.** Always pass the spacing (`compute_metrics(..., spacing=...)`); the `Evaluator` reads it from
the image header and, with `alignment="strict"` (default), refuses prediction/reference pairs on different grids.
Choose [Boundary IoU](../metrics/distance.md#boundary_iou) band widths at least as large as the largest spacing,
and resample thin tubular structures to isotropic spacing before [clDice](../metrics/topology.md#cldice) if the
skeleton matters. Report the spacing (or the resampling) with the results.

## 8. Connectivity and lesion definition {#connectivity}

**What goes wrong.** Detection metrics count connected components, so their values depend on the connectivity
used to define a lesion, on any minimum lesion size, and on the matching rule (Reinke et al. 2024, P2:
occurrence of touching or disconnected structures).

**Examples.**

*Connectivity.* Two \(4^3\) reference lesions touch at a single corner; the model segments one of them perfectly:

| Connectivity | reference lesions | [lesion recall](../metrics/detection.md#lesion_recall) | [count difference](../metrics/detection.md#lesion_count_difference) | [lesion-wise Dice](../metrics/detection.md#lesionwise_dice) |
|---|---|---|---|---|
| 26 (default) | 1 | 1.0 | 0 | 0.667 |
| 18 or 6 | 2 | 0.5 | 1 | 0.500 |

*Minimum size.* In the [detection example](../metrics/detection.md#code), with three missed 27-voxel lesions and a
27-voxel false positive, lesion recall is 0.25. Setting `min_component_voxels=30` removes all four small
components from both masks and yields a perfect lesion F1 of 1.0.

*Matching rule.* A predicted blob bridging two reference lesions scores lesion recall, precision and F1 of 1.0
under the default "overlap" criterion, but [panoptic quality](../metrics/detection.md#panoptic_quality) 0 and one
[merge](../metrics/detection.md#merge_count).

**How to avoid it.** Choose connectivity, minimum lesion size and matching rule from the clinical definition of a
lesion, before evaluation, and report them. Report [split](../metrics/detection.md#split_count) and
[merge](../metrics/detection.md#merge_count) counts alongside lesion F1, and use the lesion table to check which
lesions (by size) were missed. Note that topology metrics always use (26, 6) connectivity, independently of the
lesion connectivity.

## 9. Tolerance choice for NSD {#nsd-tolerance}

**What goes wrong.** [NSD](../metrics/distance.md#nsd) counts the surface within a tolerance \(\tau\); the value
depends strongly on \(\tau\), and a tolerance chosen after seeing results can make any method look good. Errors
beyond \(\tau\) are not graded.

**Example.** A \(24^3\) cube shifted by 2 voxels:

| \(\tau\) (mm) | 0.5 | 1 | 2 | 3 |
|---|---|---|---|---|
| NSD | 0.637 | 0.693 | 1.000 | 1.000 |

**How to avoid it.** Set \(\tau\) per structure from inter-rater variability or a clinically acceptable margin,
as Nikolov et al. (2021) did for each head-and-neck organ, and fix it before evaluation. Report it with every NSD
value (the SegEvalKit default is 2 mm). Pair NSD with [HD95](../metrics/distance.md#hd95) to see how far the
errors beyond \(\tau\) reach. The same reasoning applies to the band width of
[Boundary IoU](../metrics/distance.md#boundary_iou).

## 10. Ranking instability {#ranking}

**What goes wrong.** Rankings depend on the metric, the aggregation operator, the ranking scheme and the test
cases. Maier-Hein et al. (2018) showed that challenge winners often change under such choices (Reinke et al.
2024, P3: inadequate ranking scheme, ranking uncertainty).

**Example.** Method A is excellent on four cases and fails on one; method B is uniformly mediocre:

```python
import pandas as pd
from segevalkit.stats import rank_methods

def table(vals):
    return pd.DataFrame({"case_id": [f"c{i}" for i in range(5)], "label": "tumour",
                         "metric": "dice", "value": vals})

res = {"A": table([0.90, 0.90, 0.90, 0.90, 0.10]), "B": table([0.80] * 5)}
print(rank_methods(res, "dice", scheme="aggregate-then-rank", agg="mean"))
print(rank_methods(res, "dice", scheme="aggregate-then-rank", agg="median"))
print(rank_methods(res, "dice", scheme="rank-then-aggregate"))
```

```text
  method  mean  rank
0      B  0.80   1.0
1      A  0.74   2.0
  method  median  rank
0      A     0.9   1.0
1      B     0.8   2.0
  method  mean_rank  rank
0      A        1.2   1.0
1      B        1.8   2.0
```

The mean ranks B first; the median and rank-then-aggregate rank A first.

**How to avoid it.** Fix the ranking scheme in advance and report it. Quantify ranking uncertainty with
`segevalkit.stats.ranking_stability` (bootstrap over cases, Kendall's \(\tau\) against the full-data ranking),
and use paired tests with multiplicity correction (`segevalkit.stats.compare`) before claiming one method is
better. See [Statistics & ranking](../analysis/statistics.md).

## 11. Calibration in background-dominated volumes {#calibration-background}

**What goes wrong.** Calibration metrics average over voxels. Over a whole 3D volume nearly all voxels are easy,
correctly predicted background with \(p \approx 0\), so ECE, Brier and NLL look excellent even when the model is
badly over-confident where it matters, at the boundary (Reinke et al. 2024, P2: uncalibrated scores).

**Example.** A model that is correct after thresholding everywhere except for a 2-voxel shell outside the object,
where it predicts \(p = 0.6\) (from the [calibration example](../metrics/calibration.md#code)):

| Region | [ECE](../metrics/calibration.md#ece) | [Brier](../metrics/calibration.md#brier) | [NLL](../metrics/calibration.md#nll) | [AUROC](../metrics/calibration.md#auroc) |
|---|---|---|---|---|
| whole image (`roi="all"`) | 0.022 | 0.011 | 0.032 | 1.000 |
| 5 mm band (`roi="band"`) | 0.111 | 0.057 | 0.160 | 1.000 |

The whole-image ECE is five times smaller only because of 89,120 trivially correct background voxels.

**How to avoid it.** Compute calibration metrics with `roi="band"` and state `margin_mm` (Mehrtash et al. 2020).
Show the reliability diagram, pair ECE with a proper scoring rule (Brier or NLL), and remember that
discrimination (AUROC, AUPRC) and calibration are different properties.

## 12. Convention mismatches between tools {#conventions}

**What goes wrong.** Metrics with the same name are computed differently by different libraries, so numbers are
not comparable across papers (Reinke et al. 2024, P3: non-standardised metric definitions).

**Examples.** For a 10-voxel cube predicted inside a 20-voxel reference cube:

| Quantity | Convention A | Convention B |
|---|---|---|
| HD95 | directed max, 7.14 mm (MetricsReloaded, MONAI, DeepMind, BraTS) | pooled, 7.07 mm (MedPy) |
| Average surface distance | pooled mean ASSD, 5.63 mm (MONAI, MedPy ≥ 0.5.2) | mean of directed means MASD, 5.39 mm (MetricsReloaded MASD, MedPy ≤ 0.5.1 "assd") |

NSD differs between voxel counting (MONAI, SegEvalKit) and surface-area weighting (DeepMind), especially on
anisotropic grids. Lesion-wise Dice in BraTS 2023 dilates reference lesions before matching; SegEvalKit's
[lesion-wise Dice](../metrics/detection.md#lesionwise_dice) does not.

**How to avoid it.** Name the exact variant (SegEvalKit keys are unambiguous: [`assd`](../metrics/distance.md#assd)
vs [`masd`](../metrics/distance.md#masd), [`hd95`](../metrics/distance.md#hd95) vs
[`hd_percentile`](../metrics/distance.md#hd_percentile) with `mode="pooled"`), and report the empty-mask policy
and all parameters. The [conventions page](conventions.md) lists the conformance of each metric with other
libraries.

## References

- Reinke A, Tizabi MD, Baumgartner M, et al. Understanding metric-related pitfalls in image analysis validation.
  *Nature Methods* 21, 182–194 (2024). [doi:10.1038/s41592-023-02150-0](https://doi.org/10.1038/s41592-023-02150-0)
- Maier-Hein L, Reinke A, Godau P, et al. Metrics reloaded: recommendations for image analysis validation.
  *Nature Methods* 21, 195–212 (2024). [doi:10.1038/s41592-023-02151-z](https://doi.org/10.1038/s41592-023-02151-z)
- Maier-Hein L, Eisenmann M, Reinke A, et al. Why rankings of biomedical image analysis competitions should be
  interpreted with care. *Nature Communications* 9, 5217 (2018).
  [doi:10.1038/s41467-018-07619-7](https://doi.org/10.1038/s41467-018-07619-7)
- Wiesenfarth M, Reinke A, Landman BA, et al. Methods and open-source toolkit for analyzing and visualizing
  challenge results. *Scientific Reports* 11, 2369 (2021).
  [doi:10.1038/s41598-021-82017-6](https://doi.org/10.1038/s41598-021-82017-6)
- Taha AA, Hanbury A. Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool.
  *BMC Medical Imaging* 15, 29 (2015). [doi:10.1186/s12880-015-0068-x](https://doi.org/10.1186/s12880-015-0068-x)
- Nikolov S, Blackwell S, Zverovitch A, et al. Clinically applicable segmentation of head and neck anatomy for
  radiotherapy. *J Med Internet Res* 23(7), e26151 (2021). [doi:10.2196/26151](https://doi.org/10.2196/26151)
- Mehrtash A, Wells WM, Tempany CM, Abolmaesumi P, Kapur T. Confidence calibration and predictive uncertainty
  estimation for deep medical image segmentation. *IEEE TMI* 39(12), 3868–3878 (2020).
  [doi:10.1109/TMI.2020.3006437](https://doi.org/10.1109/TMI.2020.3006437)
