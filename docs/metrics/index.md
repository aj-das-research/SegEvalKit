# Metrics

SegEvalKit registers 53 metrics in seven families; each answers a different question and is blind to something
the others see. Combine **at least one overlap and one boundary metric**, plus detection metrics whenever there are
several instances (Metrics Reloaded). Every metric carries its metadata in one registry, from which the
[full catalogue](catalogue.md), `segevalkit metrics -v`, plot labels and reports are generated. For a
recommendation see the [decision guide](../guide/choosing.md); for traps, [Pitfalls](../guide/pitfalls.md); for
surface, distance and lesion definitions, [Conventions](../guide/conventions.md).

## The seven families

<div class="grid cards" markdown>

-   **[Overlap](overlap.md)**

    How many voxels are right?

    [Dice](overlap.md#dice), IoU, precision, recall, F\(_\beta\)

-   **[Volume](volume.md)**

    Is the amount right, wherever it is?

    [AVD](volume.md#absolute_volume_difference), [RVD](volume.md#relative_volume_difference), volumes in mL

-   **[Distance & boundary](distance.md)**

    How far off is the boundary, in mm?

    [NSD](distance.md#nsd), [HD95](distance.md#hd95), ASSD, Boundary IoU

-   **[Topology](topology.md)**

    Is the shape connected correctly?

    [clDice](topology.md#cldice), [Betti errors](topology.md#betti0_error), Euler error

-   **[Detection & instances](detection.md)**

    Were the lesions found, and only them?

    [Lesion F1](detection.md#lesion_f1), [PQ](detection.md#panoptic_quality), [lesion-wise Dice](detection.md#lesionwise_dice)

-   **[Calibration](calibration.md)**

    Can the probabilities be trusted?

    [ECE](calibration.md#ece) with `roi="band"`, [Brier](calibration.md#brier), AUPRC

-   **[Agreement](agreement.md)**

    Do the partitions agree beyond chance?

    [Cohen's kappa](agreement.md#cohen_kappa), [MCC](agreement.md#mcc), MI, VI

-   **[Full catalogue](catalogue.md)**

    Every registered metric in one table.

    Key, direction, range, unit and a one-line summary

</div>

## How to compute {#computing-metrics}

`metrics` accepts keys, aliases, [set names](#metric-sets) or `"all"`; `params` overrides defaults per metric.
Without `spacing` voxels are 1 mm; probability metrics are skipped without `prob`; a failing metric becomes NaN
with a warning unless `strict=True`.

=== "One pair"

    ```python
    from segevalkit.metrics import compute_metrics

    scores = compute_metrics(pred, ref, ["default", "masd", "cldice"],
                             spacing=(0.8, 0.8, 2.0),
                             params={"nsd": {"tolerance_mm": 1.0}})
    ```

=== "Many metrics"

    ```python
    from segevalkit.metrics import PairContext, compute_metrics

    ctx = PairContext(pred, ref, spacing=(0.8, 0.8, 2.0))    # caches surfaces, components...
    first = compute_metrics(ctx, metrics=["hd95", "assd"])   # computes surface distances
    second = compute_metrics(ctx, metrics=["nsd", "masd"])   # reuses them
    ```

=== "A dataset"

    ```python
    from segevalkit import Evaluator, EmptyPolicy

    ev = Evaluator(labels={"organ": 1, "lesion": 2},
                   metrics=["dice", "nsd", "hd95", "lesion_f1"],
                   empty=EmptyPolicy.preset("segevalkit"))
    res = ev.evaluate("preds/", "labelsTr/", n_workers=8)
    res.summary(); res.save("eval_out/")
    ```

See the [Quickstart](../getting-started/quickstart.md) and [Configuration](../getting-started/configuration.md).

## Metric sets {#metric-sets}

| Set | Metrics |
|---|---|
| `default` | dice, iou, nsd, hd95, assd, precision, recall, relative_volume_difference |
| `overlap` | dice, iou, precision, recall, specificity, voe, fbeta, tversky, mcc, cohen_kappa, balanced_accuracy |
| `distance` | hd, hd95, assd, masd, nsd, boundary_iou |
| `volume` | pred_volume, ref_volume, absolute_volume_difference, relative_volume_difference, volumetric_similarity |
| `topology` | cldice, betti0_error, betti1_error, betti2_error, euler_error |
| `detection` | all ten [detection metrics](detection.md#at-a-glance) |
| `calibration` | ece, brier, nll, auroc, auprc, soft_dice |
| `agreement` | cohen_kappa, mcc, adjusted_rand_index, variation_of_information, mutual_information, global_consistency_error |
| `all` / `all_binary` | every metric / every metric not needing probabilities |

`fpr`, `fnr`, `accuracy`, `volume_difference`, `hd_percentile` and `centroid_distance` are available by name only.
Aliases (`dsc`, `jaccard`, `sensitivity`, `hd100`, `surface_dice`, `kappa`, `pq`...) resolve everywhere;
`resolve_metrics(spec)` shows an expansion.

## Empty masks {#empty-masks}

An `EmptyPolicy` fixes what undefined cases return. `both_empty`: `"best"` (each metric's ideal value; BraTS/KiTS)
or `"nan"` (excluded; nnU-Net). `one_empty_distance` (HD, HD95, ASSD, MASD, centroid distance with exactly one mask
empty): `"worst"` = image diagonal \(\sqrt{\sum_k (n_k s_k)^2}\) mm, `"nan"`, or a number.

| Preset | `both_empty` | `one_empty_distance` |
|---|---|---|
| `segevalkit` (default) | best | image diagonal |
| `brats2023` | best | 374 mm |
| `metrics_reloaded` | best | image diagonal |
| `nan` / `nnunet` | NaN | NaN |
| `topcow` | best | 90 mm |

Independent of the policy, a ratio undefined because one mask is empty (precision with empty prediction, recall
and RVD with empty reference, lesion recall/precision without lesions) is NaN, and bounded overlap scores are 0.
Choose and report the policy before looking at results ([Pitfalls](../guide/pitfalls.md#empty-references)); each
metric's exact behaviour is in its card.

## Custom metrics {#custom-metrics}

Decorate `fn(ctx, **params) -> float` with `@register_metric` and it works in `compute_metrics`, `Evaluator`, the
CLI, plots and reports. See [Adding a metric](../developer/adding-a-metric.md).
