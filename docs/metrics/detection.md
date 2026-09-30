# Detection & instance metrics

A voxel-level Dice of 0.97 can hide a model that finds the one large tumour and misses every small metastasis.
When the clinical question is **"were the lesions found?"**, the unit of analysis must be the lesion, not the
voxel. Detection metrics count lesions; instance metrics (panoptic quality, lesion-wise Dice) combine that count
with the segmentation quality of each lesion.

They say nothing about boundary accuracy within a detected lesion, and their values depend on two choices you
must report: how lesions are defined (connectivity, minimum size) and how they are matched.

## Lesions and matching

Lesions (instances)
:   The connected components of each binary mask. The connectivity is an argument of `PairContext`,
    `compute_metrics` and `Evaluator`: `connectivity=26` (default), `18` or `6`. `min_component_voxels`
    (`min_lesion_voxels` in `Evaluator`) drops components smaller than that many voxels from **both** masks before
    matching; 0 (default) keeps everything.

`criterion="overlap"` (default)
:   Many-to-many, as in the ISLES, autoPET and BraTS lesion-wise evaluations. A reference lesion is **detected** if
    at least one predicted component overlaps it by at least `min_overlap` of the reference lesion's volume (any
    overlap when `min_overlap = 0`). A predicted component is a **false positive** if it overlaps no reference
    lesion at all.

`criterion="iou"`
:   One-to-one. Components are paired by maximum-IoU assignment (Hungarian algorithm) and a pair counts as a match
    only if \(\mathrm{IoU} > \) `iou_threshold` (strictly). With the default threshold of 0.5 every match is
    unique, which is the definition behind panoptic quality.

Metrics that accept `criterion`, `iou_threshold` and `min_overlap`: `lesion_recall`, `lesion_precision`,
`lesion_f1`, `false_positive_lesions`, `false_negative_lesions`. The others use a fixed rule: `split_count`,
`merge_count` and `lesionwise_dice` always use "overlap" with any overlap; `panoptic_quality` always uses "iou"
with its own `iou_threshold`; `lesion_count_difference` only counts components.

!!! warning "`min_overlap` is one-sided"
    Under the "overlap" criterion, `min_overlap` applies to reference lesions only. A predicted component that
    touches a lesion below the threshold leaves that lesion undetected (a false negative) but is not counted as
    a false positive either. With `min_overlap > 0`, lesion precision can therefore stay at 1 while lesion
    recall drops to 0.

## Notation

\(G_1,\dots,G_{N_G}\) are the reference lesions and \(P_1,\dots,P_{N_P}\) the predicted components. After
matching, \(TP_{les}\) is the number of matched lesions, \(FP_{les}\) the number of unmatched predicted components
and \(FN_{les}\) the number of unmatched reference lesions. Under the one-to-one "iou" criterion,
\(TP_{les}\) is the same whether counted on the reference or on the prediction side. Under the many-to-many
"overlap" criterion it is not: recall and F1 use the number of detected reference lesions, precision uses the
number of matched predicted components.

## At a glance

| Metric | Key | Unit of count | Matching | Answers |
|---|---|---|---|---|
| [Lesion recall](#lesion_recall) | `lesion_recall` | reference lesions | configurable | what fraction of lesions was found? |
| [Lesion precision](#lesion_precision) | `lesion_precision` | predicted components | configurable | what fraction of detections is real? |
| [Lesion F1](#lesion_f1) | `lesion_f1` | both | configurable | found the lesions, and only them? |
| [Count difference](#lesion_count_difference) | `lesion_count_difference` | components | none | is the lesion count right? |
| [FP lesions](#false_positive_lesions) | `false_positive_lesions` | predicted components | configurable | how many false alarms? |
| [Missed lesions](#false_negative_lesions) | `false_negative_lesions` | reference lesions | configurable | how many misses? |
| [Splits](#split_count) | `split_count` | reference lesions | any overlap | lesions broken into pieces? |
| [Merges](#merge_count) | `merge_count` | predicted components | any overlap | lesions fused together? |
| [Panoptic quality](#panoptic_quality) | `panoptic_quality` | pairs | IoU > 0.5, Hungarian | detection × outline quality |
| [Lesion-wise Dice](#lesionwise_dice) | `lesionwise_dice` | reference lesions + FPs | any overlap | Dice with every lesion weighted equally |

## Metrics

<div class="sek-metric" markdown>

### Lesion-wise recall (detection sensitivity) (L-TPR) {#lesion_recall}

<div class="sek-meta"><span class="sek-chip">key: <code>lesion_recall</code></span><span class="sek-chip">range [0, 1]</span><span class="sek-chip teal">higher is better</span><span class="sek-chip orange">dimensionless</span></div>

\[
\mathrm{L\text{-}TPR} = \frac{TP_{les}}{TP_{les} + FN_{les}} = \frac{\#\,\text{detected reference lesions}}{N_G}
\]

**In plain words:** the fraction of reference lesions that were detected, regardless of their size.

**Use it when:** missing a lesion is the main clinical risk (metastases, MS lesions, nodules, stroke).

**Watch out for:** it is trivially maximised by predicting many components; pair it with
[lesion precision](#lesion_precision). Under "overlap" a single voxel touching a lesion detects it; raise
`min_overlap` or use `criterion="iou"` for a stricter definition.

**Empty masks:** no reference lesions: 1 (`"best"`) or NaN (`"nan"`) if there are also no predicted
components, NaN otherwise (undefined). No predicted components but reference lesions: 0. "Empty" means no
components after `min_component_voxels` filtering.

**Parameters:** `criterion = "overlap"`, `iou_threshold = 0.5`, `min_overlap = 0.0`.

**Reference:** Hernandez Petzsche MR, de la Rosa E, Hanning U, et al. ISLES 2022: A multi-center magnetic
resonance imaging stroke lesion segmentation dataset. *Scientific Data* 9, 762 (2022).
[doi:10.1038/s41597-022-01875-5](https://doi.org/10.1038/s41597-022-01875-5). Maier-Hein L, Reinke A, Godau P,
et al. Metrics reloaded. *Nature Methods* 21, 195–212 (2024).
[doi:10.1038/s41592-023-02151-z](https://doi.org/10.1038/s41592-023-02151-z)

</div>

<div class="sek-metric" markdown>

### Lesion-wise precision (L-PPV) {#lesion_precision}

<div class="sek-meta"><span class="sek-chip">key: <code>lesion_precision</code></span><span class="sek-chip">range [0, 1]</span><span class="sek-chip teal">higher is better</span><span class="sek-chip orange">dimensionless</span></div>

\[
\mathrm{L\text{-}PPV} = \frac{TP_{les}}{TP_{les} + FP_{les}} = \frac{\#\,\text{matched predicted components}}{N_P}
\]

**In plain words:** the fraction of predicted lesions that correspond to a real lesion (one minus the false
discovery rate).

**Use it when:** false alarms cost reading time or lead to unnecessary follow-up.

**Watch out for:** trivially maximised by predicting only the most obvious lesion. Under "overlap" a component
that merges three lesions counts as one correct detection.

**Empty masks:** no predicted components: 1 (`"best"`) or NaN if there are also no reference lesions, NaN
otherwise. No reference lesions but predicted components: 0.

**Parameters:** `criterion = "overlap"`, `iou_threshold = 0.5`, `min_overlap = 0.0`.

**Reference:** Maier-Hein et al. 2024,
[doi:10.1038/s41592-023-02151-z](https://doi.org/10.1038/s41592-023-02151-z).

</div>

<div class="sek-metric" markdown>

### Lesion-wise F1 score (L-F1) {#lesion_f1}

<div class="sek-meta"><span class="sek-chip">key: <code>lesion_f1</code></span><span class="sek-chip">range [0, 1]</span><span class="sek-chip teal">higher is better</span><span class="sek-chip orange">dimensionless</span></div>

\[
\mathrm{L\text{-}F1} = \frac{2\,TP_{les}}{2\,TP_{les} + FP_{les} + FN_{les}}
\]

Under the "overlap" criterion \(TP_{les}\) is the number of detected reference lesions, as in the ISLES'22
evaluation code.

**In plain words:** one number for "found the lesions, and only them": the harmonic mean of lesion precision and
recall.

**Use it when:** you need a single detection score for multi-lesion disease (ISLES'22 primary metric).

**Watch out for:** it depends heavily on the matching rule, on connectivity and on any minimum lesion size.
Confluent lesions can merge or split under connected-component labelling. Report the rule with the number.

**Empty masks:** no lesions in either mask: 1 (`"best"`) or NaN. Otherwise always defined; one side empty gives 0.

**Parameters:** `criterion = "overlap"`, `iou_threshold = 0.5`, `min_overlap = 0.0`.

**Reference:** Hernandez Petzsche et al. 2022 (ISLES'22),
[doi:10.1038/s41597-022-01875-5](https://doi.org/10.1038/s41597-022-01875-5).

</div>

<div class="sek-metric" markdown>

### Absolute lesion count difference (|ΔN|) {#lesion_count_difference}

<div class="sek-meta"><span class="sek-chip">key: <code>lesion_count_difference</code></span><span class="sek-chip">range [0, ∞)</span><span class="sek-chip teal">lower is better</span><span class="sek-chip orange">lesions</span></div>

\[
\lvert\Delta N\rvert = \lvert N_P - N_G\rvert
\]

**In plain words:** how many more or fewer lesions were predicted than exist.

**Use it when:** lesion count is itself a clinical quantity (MS lesion load, number of metastases); it is an
ISLES'22 secondary metric.

**Watch out for:** it involves no matching at all: missing three lesions and hallucinating three others gives 0.
Read it next to lesion F1.

**Empty masks:** always defined; both empty gives 0.

**Parameters:** none (uses `connectivity` and `min_component_voxels`).

**Reference:** Hernandez Petzsche et al. 2022 (ISLES'22),
[doi:10.1038/s41597-022-01875-5](https://doi.org/10.1038/s41597-022-01875-5).

</div>

<div class="sek-metric" markdown>

### False-positive lesion count (FP_les) {#false_positive_lesions}

<div class="sek-meta"><span class="sek-chip">key: <code>false_positive_lesions</code></span><span class="sek-chip">range [0, ∞)</span><span class="sek-chip teal">lower is better</span><span class="sek-chip orange">lesions</span></div>

\[
FP_{les} = N_P - \#\,\text{matched predicted components}
\]

**In plain words:** the number of predicted components that correspond to no reference lesion.

**Use it when:** reporting false alarms per scan, the x-axis of FROC-style analyses.

**Watch out for:** the count depends on connectivity: a noisy prediction fragmented into specks under
6-connectivity can have many more false positives than under 26-connectivity. Consider `min_component_voxels`
for noise, and report it.

**Empty masks:** always defined; 0 when the prediction has no components.

**Parameters:** `criterion = "overlap"`, `iou_threshold = 0.5`, `min_overlap = 0.0`.

**Reference:** Maier-Hein et al. 2024,
[doi:10.1038/s41592-023-02151-z](https://doi.org/10.1038/s41592-023-02151-z). For FROC analysis: Chakraborty DP,
Berbaum KS. Observer studies involving detection and localization. *Medical Physics* 31(8), 2313–2330 (2004).
[doi:10.1118/1.1769352](https://doi.org/10.1118/1.1769352)

</div>

<div class="sek-metric" markdown>

### Missed lesion count (FN_les) {#false_negative_lesions}

<div class="sek-meta"><span class="sek-chip">key: <code>false_negative_lesions</code></span><span class="sek-chip">range [0, ∞)</span><span class="sek-chip teal">lower is better</span><span class="sek-chip orange">lesions</span></div>

\[
FN_{les} = N_G - \#\,\text{detected reference lesions}
\]

**In plain words:** the number of reference lesions that were not detected.

**Use it when:** reporting misses per scan, or stratifying misses by lesion size with the
[lesion table](#code).

**Watch out for:** a count does not say which lesions were missed; small lesions are missed far more often. Use
`lesion_table` to see the size of each missed lesion.

**Empty masks:** always defined; 0 when the reference has no components.

**Parameters:** `criterion = "overlap"`, `iou_threshold = 0.5`, `min_overlap = 0.0`.

**Reference:** Maier-Hein et al. 2024,
[doi:10.1038/s41592-023-02151-z](https://doi.org/10.1038/s41592-023-02151-z).

</div>

<div class="sek-metric" markdown>

### Split lesions (Splits) {#split_count}

<div class="sek-meta"><span class="sek-chip">key: <code>split_count</code></span><span class="sek-chip">range [0, ∞)</span><span class="sek-chip teal">lower is better</span><span class="sek-chip orange">lesions</span></div>

\[
\mathrm{Splits} = \#\{\, i : \lvert\{ j : P_j\cap G_i\neq\emptyset\}\rvert > 1 \,\}
\]

**In plain words:** the number of reference lesions covered by more than one predicted component
(over-fragmentation).

**Use it when:** lesion counts matter, or when a model tends to break elongated or confluent lesions into pieces.
Splits are invisible to voxel Dice and to "overlap"-criterion lesion F1.

**Watch out for:** it counts affected reference lesions, not pieces: a lesion split into four counts as 1, not 3.
It depends on connectivity.

**Empty masks:** always defined; 0 when either mask has no components.

**Parameters:** none (always any-overlap matching).

**Reference:** Maier-Hein et al. 2024,
[doi:10.1038/s41592-023-02151-z](https://doi.org/10.1038/s41592-023-02151-z). Carass A, Roy S, Gherman A, et al.
Evaluating white matter lesion segmentations with refined Sørensen-Dice analysis. *Scientific Reports* 10, 8242
(2020). [doi:10.1038/s41598-020-64803-w](https://doi.org/10.1038/s41598-020-64803-w)

</div>

<div class="sek-metric" markdown>

### Merged lesions (Merges) {#merge_count}

<div class="sek-meta"><span class="sek-chip">key: <code>merge_count</code></span><span class="sek-chip">range [0, ∞)</span><span class="sek-chip teal">lower is better</span><span class="sek-chip orange">lesions</span></div>

\[
\mathrm{Merges} = \#\{\, j : \lvert\{ i : P_j\cap G_i\neq\emptyset\}\rvert > 1 \,\}
\]

**In plain words:** the number of predicted components that cover more than one reference lesion
(under-separation).

**Use it when:** neighbouring lesions must be kept apart (metastasis counting, vertebra or tooth labelling). Two
reference lesions predicted as one bridged blob give lesion recall 1, lesion precision 1 and lesion F1 1 under
"overlap", but one merge and panoptic quality 0.

**Watch out for:** it counts merged predicted components, not the number of lesions swallowed. It depends on
connectivity: two lesions touching at a corner are one lesion under 26-connectivity and two under 6.

**Empty masks:** always defined; 0 when either mask has no components.

**Parameters:** none (always any-overlap matching).

**Reference:** Maier-Hein et al. 2024,
[doi:10.1038/s41592-023-02151-z](https://doi.org/10.1038/s41592-023-02151-z); Carass et al. 2020,
[doi:10.1038/s41598-020-64803-w](https://doi.org/10.1038/s41598-020-64803-w).

</div>

<div class="sek-metric" markdown>

### Panoptic quality (PQ) {#panoptic_quality}

<div class="sek-meta"><span class="sek-chip">key: <code>panoptic_quality</code></span><span class="sek-chip">alias: <code>pq</code></span><span class="sek-chip">range [0, 1]</span><span class="sek-chip teal">higher is better</span><span class="sek-chip orange">dimensionless</span></div>

\[
\mathrm{PQ} = \underbrace{\frac{\sum_{(p,g)\in TP}\mathrm{IoU}(p,g)}{\lvert TP\rvert}}_{SQ}\times
\underbrace{\frac{\lvert TP\rvert}{\lvert TP\rvert + \tfrac12\lvert FP\rvert + \tfrac12\lvert FN\rvert}}_{RQ}
= \frac{\sum_{(p,g)\in TP}\mathrm{IoU}(p,g)}{\lvert TP\rvert + \tfrac12\lvert FP\rvert + \tfrac12\lvert FN\rvert}
\]

Matches are one-to-one pairs with \(\mathrm{IoU} > 0.5\) (Hungarian assignment).

**In plain words:** detection quality (RQ, the F1 of matched lesions) times segmentation quality (SQ, the mean IoU
of the matched pairs).

**Use it when:** evaluating instance segmentation (multiple lesions, vertebrae, cells). Metrics Reloaded
recommends it for instance segmentation.

**Watch out for:** a small lesion whose IoU falls just below the threshold counts as both a false negative and a
false positive. Splits and merges are punished hard (a merge of two lesions gives 0 matches). Averaging PQ per
image differs from pooling TP/FP/FN over the dataset.

**Empty masks:** no instances in either mask: 1 (`"best"`) or NaN. One side without instances: 0.

**Parameters:** `iou_threshold = 0.5`.

**Reference:** Kirillov A, He K, Girshick R, Rother C, Dollár P. Panoptic segmentation. *CVPR 2019*, 9404–9413.
[arXiv:1801.00868](https://arxiv.org/abs/1801.00868)

</div>

<div class="sek-metric" markdown>

### Lesion-wise Dice (BraTS 2023) (LW-DSC) {#lesionwise_dice}

<div class="sek-meta"><span class="sek-chip">key: <code>lesionwise_dice</code></span><span class="sek-chip">range [0, 1]</span><span class="sek-chip teal">higher is better</span><span class="sek-chip orange">dimensionless</span></div>

\[
\mathrm{LW\text{-}DSC} = \frac{1}{N_G + FP_{les}}\sum_{i=1}^{N_G}
\mathrm{DSC}\Big(G_i,\ \textstyle\bigcup_{j:\, P_j\cap G_i\neq\emptyset} P_j\Big)
\]

**In plain words:** Dice computed per lesion and averaged, with every missed lesion and every false-positive
component scoring 0. A missed 27-voxel metastasis counts as much as a well-segmented 8000-voxel tumour.

**Use it when:** multi-focal disease, where whole-volume Dice is dominated by the largest lesion (BraTS 2023
lesion-wise protocol).

**Watch out for:** each reference lesion is compared with the union of all predicted components touching it,
so one large component touching two lesions is counted in both. SegEvalKit does not dilate the reference before
matching and applies no volume threshold; the official BraTS code dilates each lesion (1–3 iterations,
depending on the sub-challenge) and drops small reference lesions (50 mm³ for most sub-challenges). To reproduce it, pass a dilated reference and set `min_lesion_voxels`
(which filters both masks).

**Empty masks:** no reference lesions and no false positives: 1 (`"best"`) or NaN. Empty reference with predicted
components: 0. Empty prediction with reference lesions: 0.

**Parameters:** none (always any-overlap matching).

**Reference:** Kazerooni AF, Khalili N, Liu X, et al. The Brain Tumor Segmentation (BraTS) Challenge 2023: focus on
pediatrics (CBTN-CONNECT-DIPGR-ASNR-MICCAI BraTS-PEDs). [arXiv:2305.17033](https://arxiv.org/abs/2305.17033)
(2023). BraTS 2023 lesion-wise evaluation code:
[rachitsaluja/BraTS-2023-Metrics](https://github.com/rachitsaluja/BraTS-2023-Metrics).

</div>

## Code

The `detection` metric set contains all ten metrics. In this example one large lesion is found, three small ones
are missed and one false positive is added.

```python
import numpy as np
from segevalkit.metrics import compute_metrics, PairContext, lesion_table

ref = np.zeros((48, 48, 48), bool)
ref[4:24, 4:24, 4:24] = True                         # one large lesion (8000 voxels)
for z in (30, 36, 42):
    ref[z:z + 3, 30:33, 30:33] = True                # three 27-voxel lesions
pred = np.zeros_like(ref)
pred[5:24, 4:24, 4:24] = True                        # large lesion found
pred[40:43, 5:8, 5:8] = True                         # one false positive

scores = compute_metrics(pred, ref, ["dice", "detection"])
for name, value in scores.items():
    print(f"{name:24s} {value:.4f}")

ctx = PairContext(pred, ref, spacing=(1, 1, 1), connectivity=26)
for row in lesion_table(ctx):
    print(f"{row['kind']:8s} #{row['component']}  {row['volume_ml']:.3f} mL  "
          f"detected={row['detected']}  dice={row['dice']:.3f}")
```

```text
dice                     0.9677
lesion_precision         0.5000
lesion_recall            0.2500
lesion_f1                0.3333
panoptic_quality         0.3167
lesionwise_dice          0.1949
lesion_count_difference  2.0000
false_positive_lesions   1.0000
false_negative_lesions   3.0000
split_count              0.0000
merge_count              0.0000
ref      #1  8.000 mL  detected=True  dice=0.974
ref      #2  0.027 mL  detected=False  dice=0.000
ref      #3  0.027 mL  detected=False  dice=0.000
ref      #4  0.027 mL  detected=False  dice=0.000
pred_fp  #2  0.027 mL  detected=False  dice=0.000
```

Voxel Dice is 0.97; lesion recall is 0.25. The lesion table (also collected by `Evaluator` whenever a detection
metric runs) has one row per reference lesion and per unmatched predicted component, with columns `kind`,
`component`, `volume_ml`, `detected`, `dice`, `iou` and `n_touching`, for size-stratified analysis.

Matching rule, connectivity and minimum size are passed like any other option:

```python
keys = ["lesion_recall", "lesion_precision", "lesion_f1"]
strict = compute_metrics(
    pred, ref, keys,
    params={k: {"criterion": "iou", "iou_threshold": 0.5} for k in keys},
    connectivity=6,               # face-connected lesions
    min_component_voxels=30,      # ignore components smaller than 30 voxels
)
print({k: round(v, 4) for k, v in strict.items()})
```

```text
{'lesion_recall': 1.0, 'lesion_precision': 1.0, 'lesion_f1': 1.0}
```

The perfect scores are an artefact: the 30-voxel threshold removed the three missed 27-voxel lesions and the
false positive from both masks. A minimum lesion size changes the question being asked; choose it from the
clinical definition of a lesion, not from the results, and report it.

!!! note "FROC"
    Free-response ROC curves and the LUNA16 CPM need a confidence score per predicted lesion, which a binary mask
    does not provide. Use the lesion table together with your model's lesion scores to build them.
