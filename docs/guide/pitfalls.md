# Metric pitfalls

A metric value means little without the task, the data and the way it was computed and aggregated. This page
lists the thirteen pitfalls that most often distort 3D segmentation results, following the taxonomy of
Reinke et al. (2024): poor metric *selection* (P2) and poor metric *application* (P3). Each worked example was
computed with SegEvalKit on 1 mm isotropic voxels unless stated otherwise.

| Pitfall | What goes wrong | Remedy in SegEvalKit |
|---|---|---|
| [Size bias](#size-bias) | Same boundary error costs small structures far more Dice | Per-structure [Dice](../metrics/overlap.md#dice), `stats.stratify`, add [NSD](../metrics/distance.md#nsd) |
| [Class imbalance](#class-imbalance) | TN-based metrics are ≈ 1 for almost any prediction | TN-free [overlap](../metrics/overlap.md) metrics, [AUPRC](../metrics/calibration.md#auprc) |
| [Single-aspect blindness](#boundary-volume) | Each family misses position, boundary or topology errors | `default` set; add [volume](../metrics/volume.md), [topology](../metrics/topology.md), [detection](../metrics/detection.md) |
| [HD outliers](#outliers) | One stray voxel sets the Hausdorff distance | [HD95](../metrics/distance.md#hd95) or [NSD](../metrics/distance.md#nsd) as endpoint |
| [Empty references](#empty-references) | 0/0 cases handled differently by every tool | Explicit [`EmptyPolicy`](../metrics/index.md#empty-masks), `stats.presence_detection` |
| [Aggregation](#aggregation) | Means hide failures, NaN exclusion and sign cancellation | Per-case results, median/IQR, [bootstrap CIs](../analysis/statistics.md) |
| [Anisotropic spacing](#anisotropic-spacing) | Distances reported in voxels, not mm | Pass `spacing`; `alignment="strict"` in the `Evaluator` |
| [Lesion definition](#connectivity) | Detection scores depend on connectivity, size and matching | Set `connectivity`, `min_lesion_voxels`; [split/merge counts](../metrics/detection.md#merge_count) |
| [NSD tolerance](#nsd-tolerance) | \(\tau\) chosen after the fact flatters any method | Fix \(\tau\) per structure in advance ([NSD](../metrics/distance.md#nsd)) |
| [Ranking instability](#ranking) | Winner changes with aggregation or scheme | `stats.rank_methods`, `stats.ranking_stability` |
| [Background-dominated calibration](#calibration-background) | Easy background makes ECE, Brier, NLL look good | `roi="band"` for [calibration](../metrics/calibration.md) metrics |
| [Convention mismatches](#conventions) | Same name, different formula across libraries | Unambiguous keys; [conventions page](conventions.md) |
| [Annotation conventions](#annotation-conventions) | The reference defines a structure differently from the model | Evaluate a region both sides agree on (`ref_file="a+b"`) |

## Size and small-structure bias {#size-bias}

Overlap metrics normalise by structure size, so the same boundary error costs a small structure far more than a
large one. A mean Dice over structures of different size mixes quality with size. Report Dice per structure,
stratify by volume (`segevalkit.stats.stratify(result, "dice", by="ref_volume_ml")`) and add a size-independent
boundary metric in mm ([NSD](../metrics/distance.md#nsd), [HD95](../metrics/distance.md#hd95)).

??? example "Worked example"

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

    The error is identical (1 mm everywhere), yet Dice ranges from 0.667 to 0.967. RVD behaves the same way:
    3 extra voxels on a 2-voxel lesion give RVD = +1.5. For small lesions prefer
    [AVD](../metrics/volume.md#absolute_volume_difference) in mL and [lesion-wise metrics](../metrics/detection.md).

## Class imbalance and TN-based metrics {#class-imbalance}

In 3D the background outnumbers the foreground by \(10^3\)–\(10^6\), so metrics that use true negatives
(accuracy, specificity, voxel-wise AUROC, kappa, MCC, ARI, MI, GCE) are dominated by them and change when the image
is cropped. Use TN-free metrics (Dice, IoU, precision, recall, distances) as endpoints and prefer
[AUPRC](../metrics/calibration.md#auprc) to AUROC.

??? example "Worked example"

    A \(64^3\) volume; a 64-voxel lesion is missed entirely and a 64-voxel false positive appears elsewhere.

    | Metric | Value |
    |---|---|
    | [Dice](../metrics/overlap.md#dice) | 0.0000 |
    | [Accuracy](../metrics/overlap.md#accuracy) | 0.9995 |
    | [Specificity](../metrics/overlap.md#specificity) | 0.9998 |
    | [Balanced accuracy](../metrics/overlap.md#balanced_accuracy) | 0.4999 |
    | [MCC](../metrics/agreement.md#mcc) | −0.0002 |

    A *perfect* prediction of the same lesion has [mutual information](../metrics/agreement.md#mutual_information)
    0.0023 nats, against 0.693 nats for a perfect prediction of a structure filling half the volume.

## Boundary, volume and topology blindness {#boundary-volume}

Each metric family sees one aspect of the error. Report at least one [overlap](../metrics/overlap.md) and one
[boundary](../metrics/distance.md) metric (the `default` set does), and add volume, topology or detection metrics
when the task calls for them; the [decision guide](choosing.md) maps task properties to metric sets.

??? example "Worked example"

    - **Volume is blind to position.** Two equal, non-touching cubes: [VS](../metrics/volume.md#volumetric_similarity) = 1,
      [RVD](../metrics/volume.md#relative_volume_difference) = 0, [AVD](../metrics/volume.md#absolute_volume_difference) = 0 mL, Dice = 0.
    - **Overlap is blind to where the error is.** Two predictions of a 4096-voxel cube:

        | Prediction | Dice | HD95 (mm) | ASSD (mm) | NSD (τ = 2 mm) | Lesion precision |
        |---|---|---|---|---|---|
        | Shifted by 2 voxels | 0.875 | 2.0 | 0.675 | 1.000 | 1.0 |
        | 2 slices missing + 512-voxel blob ≈ 22 mm away | 0.875 | 30.8 | 3.279 | 0.897 | 0.5 |

    - **Overlap and distance are blind to topology.** A 2-voxel gap in a 56-voxel vessel: Dice 0.982, HD95 0 mm,
      NSD 1.0, clDice 0.982, but [β₀ error](../metrics/topology.md#betti0_error) = 1. Filling the lumen of a ring
      gives Dice 0.80 and [β₁ error](../metrics/topology.md#betti1_error) = 1.

## Outlier sensitivity of the Hausdorff distance {#outliers}

[HD](../metrics/distance.md#hd) is a maximum, so one stray voxel sets its value, and noisy references inflate it
for every method. Use [HD95](../metrics/distance.md#hd95) or [NSD](../metrics/distance.md#nsd) as the endpoint and
keep HD only where a worst-case bound is the clinical question, reported next to a robust metric.

??? example "Worked example"

    A perfect prediction of a 20 mm cube plus one false-positive voxel ≈ 54 mm away:

    | Dice | HD | HD95 | ASSD | NSD (τ = 2 mm) |
    |---|---|---|---|---|
    | 0.9999 | 53.69 mm | 0.00 mm | 0.012 mm | 0.9998 |

    One voxel out of 8,001 moves HD from 0 to 53.7 mm. (`min_component_voxels` affects detection metrics, not HD.)

## Empty references {#empty-references}

When a structure is absent from the reference, Dice is 0/0 or 0 and HD is undefined; tools silently return NaN,
`inf`, 1 or a penalty, and the choice can dominate the mean. Choose an
[`EmptyPolicy`](../metrics/index.md#empty-masks) before evaluation and report it, count NaN values (`n_nan` in
`EvaluationResult.summary()`), and evaluate case-level presence with `segevalkit.stats.presence_detection`.

??? example "Worked example"

    Cases A (Dice 0.90, HD95 3 mm) and B (0.85, 5 mm); C, structure correctly absent; D, absent but a false
    positive is predicted. CT of \(512\times512\times100\) at \(0.8\times0.8\times2.5\) mm (diagonal 630.9 mm):

    | Policy | Dice C | Dice D | Mean Dice | HD95 C | HD95 D | Mean HD95 |
    |---|---|---|---|---|---|---|
    | `segevalkit` (best / image diagonal) | 1 | 0 | 0.688 | 0 | 630.9 | 159.7 mm |
    | `nan` (exclude undefined) | NaN | 0 | 0.583 | NaN | NaN | 4.0 mm |

    Under `"nan"` the hallucination in D does not affect mean HD95, and the correct call in C vanishes from both
    means. `EmptyPolicy.preset("brats2023")` reproduces the 374 mm BraTS penalty; every case carries
    `_ref_empty` and `_pred_empty` flags.

## Aggregation {#aggregation}

How per-case values become one number can change the conclusion. Compute metrics per case and per structure
first (as `EvaluationResult` does), report median and IQR next to the mean, report absolute values for signed
metrics, and use bootstrap CIs and paired tests ([Statistics & ranking](../analysis/statistics.md)).

??? example "Worked example"

    - **Mean over structures.** Liver Dice 0.96 (1.5 M voxels) and tumour Dice 0.40 (10 k voxels): the mean is
      0.68; a pooled "global" Dice is 0.956 and hides the failing tumour completely.
    - **NaN exclusion.** In [empty references](#empty-references), mean HD95 drops from 159.7 to 4.0 mm by
      exclusion alone.
    - **Signed errors cancel.** [RVD](../metrics/volume.md#relative_volume_difference) of +0.2 on half the cases
      and −0.2 on the rest averages to 0.
    - **Pooled vs per-case.** Pooling voxels over the dataset weights large patients more; it is a different
      quantity.
    - **Hierarchy.** Scans of one patient are not independent; bootstrap at the patient level.

## Anisotropic spacing {#anisotropic-spacing}

Distances and volumes are physical: on a \(0.8\times0.8\times5\) mm MR a one-slice error is 5 mm, not "1". Always
pass `spacing` to `compute_metrics`; the `Evaluator` reads it from the header and, with `alignment="strict"`
(default), refuses pairs on different grids. Report the spacing, and resample thin tubes to isotropic spacing before
[clDice](../metrics/topology.md#cldice) if the skeleton matters.

??? example "Worked example"

    A \(16\times16\times8\) box shifted by one slice through-plane:

    | Spacing (mm) | Dice | HD95 | ASSD | NSD (τ = 2 mm) | AVD |
    |---|---|---|---|---|---|
    | None (treated as 1 × 1 × 1) | 0.875 | 1.0 mm | 0.518 mm | 1.000 | 0 mL |
    | 0.8 × 0.8 × 5.0 | 0.875 | 5.0 mm | 1.979 mm | 0.592 | 0 mL |

    Dice is unchanged; every distance changes several-fold. State [Boundary IoU](../metrics/distance.md#boundary_iou)
    band widths together with the spacing.

## Connectivity and lesion definition {#connectivity}

Detection metrics count connected components, so they depend on connectivity, minimum lesion size and the matching
rule. Fix all three from the clinical lesion definition before evaluation, report them, and show
[split](../metrics/detection.md#split_count) and [merge](../metrics/detection.md#merge_count) counts next to lesion
F1. Topology metrics always use (26, 6) connectivity.

??? example "Worked example"

    Two \(4^3\) reference lesions touch at one corner; the model segments one of them perfectly:

    | Connectivity | Reference lesions | [Lesion recall](../metrics/detection.md#lesion_recall) | [Count difference](../metrics/detection.md#lesion_count_difference) | [Lesion-wise Dice](../metrics/detection.md#lesionwise_dice) |
    |---|---|---|---|---|
    | 26 (default) | 1 | 1.0 | 0 | 0.667 |
    | 18 or 6 | 2 | 0.5 | 1 | 0.500 |

    - **Minimum size.** In the [detection example](../metrics/detection.md#code) lesion recall is 0.25;
      `min_component_voxels=30` removes the four 27-voxel components and yields lesion F1 = 1.0.
    - **Matching rule.** A blob bridging two reference lesions scores lesion F1 = 1.0 under the default "overlap"
      criterion, but [panoptic quality](../metrics/detection.md#panoptic_quality) 0 and one merge.

## Tolerance choice for NSD {#nsd-tolerance}

[NSD](../metrics/distance.md#nsd) depends strongly on its tolerance \(\tau\), and errors beyond \(\tau\) are not
graded. Set \(\tau\) per structure from inter-rater variability before evaluation (Nikolov et al. 2021), report it
(default 2 mm), and pair NSD with [HD95](../metrics/distance.md#hd95).

??? example "Worked example"

    A \(24^3\) cube shifted by 2 voxels:

    | \(\tau\) (mm) | 0.5 | 1 | 2 | 3 |
    |---|---|---|---|---|
    | NSD | 0.637 | 0.693 | 1.000 | 1.000 |

## Ranking instability {#ranking}

Rankings depend on the metric, aggregation, ranking scheme and test cases (Maier-Hein et al. 2018). Fix the scheme
in advance, quantify uncertainty with `segevalkit.stats.ranking_stability`, and use paired tests with multiplicity
correction (`segevalkit.stats.compare`) before claiming a winner.

??? example "Worked example"

    Method A is excellent on four cases and fails on one; method B is uniformly mediocre.

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

## Calibration in background-dominated volumes {#calibration-background}

Over a whole volume nearly all voxels are easy background with \(p \approx 0\), so ECE, Brier and NLL look
excellent even when the model is over-confident at the boundary. Use `roi="band"` with a stated `margin_mm`
(Mehrtash et al. 2020), pair ECE with Brier or NLL, and keep discrimination (AUROC, AUPRC) separate from calibration.

??? example "Worked example"

    Correct everywhere except a 2-voxel shell outside the object where \(p = 0.6\)
    (the [calibration example](../metrics/calibration.md#code)):

    | Region | [ECE](../metrics/calibration.md#ece) | [Brier](../metrics/calibration.md#brier) | [NLL](../metrics/calibration.md#nll) | [AUROC](../metrics/calibration.md#auroc) |
    |---|---|---|---|---|
    | Whole image (`roi="all"`) | 0.022 | 0.011 | 0.032 | 1.000 |
    | 5 mm band (`roi="band"`) | 0.111 | 0.057 | 0.160 | 1.000 |

    The whole-image ECE is five times smaller only because of 89,120 trivially correct background voxels.

## Convention mismatches between tools {#conventions}

Metrics with the same name are computed differently by different libraries. Name the exact variant: SegEvalKit
keys are unambiguous ([`assd`](../metrics/distance.md#assd) vs [`masd`](../metrics/distance.md#masd);
[`hd95`](../metrics/distance.md#hd95) `mode="directed"` by default, `"pooled"` for MedPy). Report the empty-mask
policy and all parameters; the [conventions page](conventions.md) lists conformance with other libraries.

??? example "Worked example"

    A 10-voxel cube predicted inside a 20-voxel reference cube:

    | Quantity | Convention A | Convention B |
    |---|---|---|
    | HD95 | Directed max, 7.14 mm (MetricsReloaded, MONAI, DeepMind, BraTS) | Pooled, 7.07 mm (MedPy) |
    | Average surface distance | Pooled mean ASSD, 5.63 mm (MONAI, MedPy ≥ 0.5.2) | Mean of directed means MASD, 5.39 mm (MetricsReloaded, MedPy ≤ 0.5.1 "assd") |

    NSD also differs between voxel counting (MONAI, SegEvalKit) and surface-area weighting (DeepMind), and BraTS
    2023 lesion-wise Dice dilates reference lesions before matching; SegEvalKit's
    [lesion-wise Dice](../metrics/detection.md#lesionwise_dice) does not.

## Annotation conventions of the reference {#annotation-conventions}

A dataset's masks encode choices: whether a tumour belongs to the organ, whether vessels are cut out, where an
organ ends. A model trained elsewhere follows its own choices, and the mismatch is scored as error. PanTS annotates
the pancreatic lesion inconsistently relative to the pancreas mask (sometimes inside it, sometimes outside), so a
correct pancreas prediction loses Dice wherever the lesion was left out of the reference. Evaluate the region both
conventions agree on, pancreas ∪ lesion, and state it with the results (the [PanTS preset](../datasets/presets.md#pants)
notes say the same).

??? example "Worked example"

    ```python
    ev = sek.Evaluator(labels={
        # reference: union of two per-structure files; prediction: union of ids
        "pancreas": {"ref_file": "pancreas.nii.gz+pancreatic_lesion.nii.gz",
                     "pred": [17, 18, 19, 20, 21, 28]},
        "pancreatic_lesion": {"ref_file": "pancreatic_lesion.nii.gz", "pred": 28},
    })
    ```

    All pancreas results in the [quickstart walkthrough](../getting-started/quickstart.md) use this union
    (the top row of the three-model comparison shows it on one slice).

## References

- Reinke A, et al. Understanding metric-related pitfalls in image analysis validation. *Nat Methods* 21, 182–194
  (2024). [doi:10.1038/s41592-023-02150-0](https://doi.org/10.1038/s41592-023-02150-0)
- Maier-Hein L, et al. Metrics reloaded: recommendations for image analysis validation. *Nat Methods* 21, 195–212
  (2024). [doi:10.1038/s41592-023-02151-z](https://doi.org/10.1038/s41592-023-02151-z)
- Maier-Hein L, et al. Why rankings of biomedical image analysis competitions should be interpreted with care.
  *Nat Commun* 9, 5217 (2018). [doi:10.1038/s41467-018-07619-7](https://doi.org/10.1038/s41467-018-07619-7)
- Wiesenfarth M, et al. Methods and open-source toolkit for analyzing and visualizing challenge results.
  *Sci Rep* 11, 2369 (2021). [doi:10.1038/s41598-021-82017-6](https://doi.org/10.1038/s41598-021-82017-6)
- Taha AA, Hanbury A. Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool.
  *BMC Med Imaging* 15, 29 (2015). [doi:10.1186/s12880-015-0068-x](https://doi.org/10.1186/s12880-015-0068-x)
- Nikolov S, et al. Clinically applicable segmentation of head and neck anatomy for radiotherapy. *J Med Internet
  Res* 23(7), e26151 (2021). [doi:10.2196/26151](https://doi.org/10.2196/26151)
- Mehrtash A, et al. Confidence calibration and predictive uncertainty estimation for deep medical image
  segmentation. *IEEE TMI* 39(12), 3868–3878 (2020). [doi:10.1109/TMI.2020.3006437](https://doi.org/10.1109/TMI.2020.3006437)
