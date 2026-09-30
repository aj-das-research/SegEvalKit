# Decision guide

SegEvalKit follows the **problem fingerprint** idea of Metrics Reloaded (Maier-Hein et al. 2024,
*Nat Methods* 21:195): describe the problem, then derive the metrics from the description instead of from habit.

## The recommender

=== "Python"

    ```python
    from segevalkit.guide import Fingerprint, recommend

    rec = recommend(Fingerprint(structure="small_lesion", multi_instance=True, volumetry=True))
    print(rec)
    rec.table()          # metric, family, role, why, reference (DataFrame)
    rec.params           # suggested parameters, e.g. {'nsd': {'tolerance_mm': 1.0}}
    rec.to_markdown()    # paste into a paper's methods section
    ```

=== "Command line"

    ```console
    $ segevalkit recommend --structure small_lesion --multi-instance --volumetry
    dice                        primary    Overlap for comparability; unstable for small objects ...
    nsd                         primary    Boundary agreement within a clinically acceptable 1.0 mm ...
    masd                        secondary  Average boundary error; HD-type metrics are dominated by single voxels ...
    lesion_f1                   primary    Unit of analysis is the lesion: were they found, and only they?
    ...
    ```

The recommendation feeds straight into an evaluation:

```python
import segevalkit as sek

ev = sek.Evaluator(labels=..., metrics=rec.metrics, params=rec.params,
                   empty=sek.EmptyPolicy.preset(rec.empty_policy))
```

## Fingerprint items

`structure` (str)
:   `large_organ`, `small_structure`, `small_lesion`, `large_lesion`, `tubular` or `hollow`. Size decides whether overlap metrics are stable; shape decides whether topology matters.

`multi_instance` (bool)
:   If objects are counted, the lesion (not the voxel) is the unit of analysis.

`boundary_critical` (bool)
:   Radiotherapy and surgery care about worst-case boundary error.

`volumetry` (bool)
:   If volume is the endpoint, report it directly (mL, bias, limits of agreement).

`empty_references` (bool)
:   Dice and HD are undefined for absent structures; presence detection must be reported separately.

`probabilistic` (bool)
:   Probabilities shown to users need calibration metrics.

`fp_fn_asymmetric` (bool)
:   Missing tissue can cost more than adding it: F-beta with β > 1.

`noisy_reference` (bool)
:   Imprecise references call for tolerance-based metrics.

`tolerance_mm` (float)
:   Acceptable boundary deviation, ideally from inter-rater variability.

`ranking` (bool)
:   Rankings need paired tests and a stability analysis.

## Decision table

| Scenario | Overlap | Boundary | Add | Rationale |
|---|---|---|---|---|
| **Large compact organ** (liver, kidney, spleen) | [Dice](../metrics/overlap.md#dice) | [NSD](../metrics/distance.md#nsd) + [HD95](../metrics/distance.md#hd95) | [RVD](../metrics/volume.md#relative_volume_difference) | Dice is informative for large objects but saturates; NSD tolerates annotation noise, HD95 catches leakage. |
| **Small structure / lesion** | Dice (never alone) | NSD + [MASD](../metrics/distance.md#masd) | [lesion F1](../metrics/detection.md#lesion_f1), [lesion-wise Dice](../metrics/detection.md#lesionwise_dice) | One voxel moves Dice a lot; HD is dominated by single voxels; detection is the clinical question. |
| **Tubular** (vessels, ducts, airways) | [clDice](../metrics/topology.md#cldice) + Dice | NSD | [Betti-0/1 errors](../metrics/topology.md#betti0_error) | A broken vessel scores a high Dice; skeleton and Betti metrics see the break. |
| **Hollow / thin-walled** (colon, bladder wall, myocardium) | Dice | NSD (small τ) | [Betti-2 error](../metrics/topology.md#betti2_error) | A filled lumen is a cavity error, invisible to overlap. |
| **Multi-instance lesions** | per-lesion Dice | per-lesion distance | [PQ](../metrics/detection.md#panoptic_quality), lesion F1, [split / merge](../metrics/detection.md#split_count), [count difference](../metrics/detection.md#lesion_count_difference) | Detection first, then per-instance quality; state the matching rule. |
| **Boundary-critical** (radiotherapy) | Dice | NSD at organ τ, HD95, [HD](../metrics/distance.md#hd) (safety) | [boundary IoU](../metrics/distance.md#boundary_iou) | NSD approximates correction effort; HD bounds the geometric error. |
| **Volumetry** | Dice (sanity) | – | [AVD](../metrics/volume.md#absolute_volume_difference), RVD, Bland–Altman, ICC | Volume is the endpoint; position-blind metrics need an overlap companion. |
| **Probabilities** | [soft Dice](../metrics/calibration.md#soft_dice) | – | [ECE](../metrics/calibration.md#ece) (`roi="band"`), [Brier](../metrics/calibration.md#brier), [AUPRC](../metrics/calibration.md#auprc) | Background dominates whole-volume calibration; restrict to the uncertain band. |
| **Empty references possible** | Dice with an explicit policy | HD95 with a penalty policy | [presence detection](../analysis/statistics.md#presence-detection) | Report "did it find the structure at all?" separately. |
| **Method ranking** | as above | as above | paired tests, [ranking stability](../analysis/statistics.md#ranking) | Rankings are fragile (Maier-Hein et al. 2018). |

## Metrics to avoid as primary endpoints

| Metric | Why |
|---|---|
| [Accuracy](../metrics/overlap.md#accuracy), [specificity](../metrics/overlap.md#specificity) | Dominated by true-negative background; ≈ 1 for almost any 3D prediction. |
| Voxel-level [AUROC](../metrics/calibration.md#auroc) | Dominated by easy background; prefer AUPRC or ECE in a band. |
| [Volumetric similarity](../metrics/volume.md#volumetric_similarity) or RVD alone | Blind to position: a mask in the wrong place can score perfectly. |
| [HD](../metrics/distance.md#hd) alone | Set by one outlier voxel. |
| [GCE](../metrics/agreement.md#global_consistency_error), [ARI](../metrics/agreement.md#adjusted_rand_index) | Weak interpretation for binary tasks; they track Dice. |
| Mean Dice across structures of very different sizes | Not comparable; report per structure. |
