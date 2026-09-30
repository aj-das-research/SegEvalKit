# Overlap metrics

Overlap metrics count voxels. Every one of them is a function of the four cells of the voxel confusion matrix,
so they answer one question: **how many voxels were labelled correctly?** They are cheap, bounded and
spacing-free, which is why Dice is the default score in almost every segmentation paper. They are also blind to
*where* the wrong voxels are. A one-voxel rim error and a detached blob of the same size cost the same.

Metrics that ignore the true negatives (Dice, IoU, precision, recall, F\(_\beta\), Tversky) do not depend on
the size of the image. Metrics that use \(TN\) (specificity, FPR, accuracy, balanced accuracy) depend on the field
of view. In 3D volumes, where background outnumbers foreground by \(10^3\) to \(10^6\), they sit close to 1 for
almost any prediction.

!!! note "MCC and Cohen's kappa"
    The Matthews correlation coefficient and Cohen's kappa are also computed from the confusion counts, but
    they are registered in the `agreement` family. See [MCC](agreement.md#mcc) and
    [Cohen's kappa](agreement.md#cohen_kappa). Both are part of the `overlap` metric set.

## Notation

For one structure, \(G\) is the set of reference ("ground truth") foreground voxels and \(P\) the set of
predicted foreground voxels. With \(N\) the number of voxels in the image:

| Symbol | Definition |
|---|---|
| \(TP = \lvert P \cap G \rvert\) | voxels labelled foreground by both |
| \(FP = \lvert P \setminus G \rvert\) | predicted foreground, reference background |
| \(FN = \lvert G \setminus P \rvert\) | reference foreground, predicted background |
| \(TN = N - TP - FP - FN\) | background in both |

Any non-zero voxel is foreground. 2D inputs are treated as a single slice. Counts are exact integers (on the
GPU they are computed in PyTorch; the result is identical).

## At a glance

| Metric | Key | Uses \(TN\) | Direction | Symmetric in \(P, G\) | Main blind spot |
|---|---|:-:|---|:-:|---|
| [Dice](#dice) | `dice` | no | higher | yes | size-dependent; ignores error location |
| [IoU](#iou) | `iou` | no | higher | yes | monotone transform of Dice |
| [VOE](#voe) | `voe` | no | lower | yes | \(1-\mathrm{IoU}\) |
| [Precision](#precision) | `precision` | no | higher | no | maximised by under-segmentation |
| [Recall](#recall) | `recall` | no | higher | no | maximised by over-segmentation |
| [Specificity](#specificity) | `specificity` | yes | higher | no | \(\approx 1\) in any 3D volume |
| [FPR](#fpr) | `fpr` | yes | lower | no | \(\approx 0\) in any 3D volume |
| [FNR](#fnr) | `fnr` | no | lower | no | \(1-\) recall |
| [F\(_\beta\)](#fbeta) | `fbeta` | no | higher | no (\(\beta\ne1\)) | \(\beta\) must be justified |
| [Tversky](#tversky) | `tversky` | no | higher | no (\(\alpha\ne\beta\)) | \(\alpha,\beta\) must be justified |
| [Accuracy](#accuracy) | `accuracy` | yes | higher | yes | dominated by background |
| [Balanced accuracy](#balanced_accuracy) | `balanced_accuracy` | yes | higher | no | specificity half is \(\approx 1\) |

Useful identities: \(\mathrm{IoU} = \mathrm{DSC}/(2-\mathrm{DSC})\); Dice is the harmonic mean of precision and
recall; Tversky with \(\alpha=\beta=0.5\) is Dice and with \(\alpha=\beta=1\) is IoU; F\(_\beta\) is Tversky with
\(\alpha = 1/(1+\beta^2)\) and \(\beta_{\mathrm{TI}} = \beta^2/(1+\beta^2)\). Reporting several of these adds
little information (Taha & Hanbury 2015).

## Metrics

<div class="sek-metric" markdown>

### Dice similarity coefficient (DSC) {#dice}

<div class="sek-meta"><span class="sek-chip">key: <code>dice</code></span><span class="sek-chip">aliases: <code>dsc</code>, <code>f1</code></span><span class="sek-chip">range [0, 1]</span><span class="sek-chip teal">higher is better</span><span class="sek-chip orange">dimensionless</span></div>

\[
\mathrm{DSC} = \frac{2\lvert P\cap G\rvert}{\lvert P\rvert+\lvert G\rvert} = \frac{2\,TP}{2\,TP+FP+FN}
\]

**In plain words:** the fraction of voxels the two masks share, relative to their average size. 1 means
identical, 0 means no shared voxel.

**Use it when:** you need the standard overall-agreement score for a medium or large structure. Metrics Reloaded
recommends Dice (or IoU) as the default overlap metric for semantic segmentation, paired with a boundary metric
such as [NSD](distance.md#nsd).

**Watch out for:**

- *Small structures* (Reinke et al. 2024, category P2). A one-voxel shift costs a 3×3×3 cube a third of its
  Dice (0.667) and a 30×30×30 cube 3 % (0.967) (see [Pitfalls](../guide/pitfalls.md#size-bias)). Mean Dice across structures of
  different size is not comparable.
- It ignores where errors lie and their shape. Very different error patterns can give the same Dice.
- It is 0 for any non-overlapping prediction, however near or far it is.
- It ignores topology: a vessel cut in two keeps a Dice near 1 (see [clDice](topology.md#cldice)).
- A mean hides catastrophic failures; report the distribution.

**Empty masks:** both empty: 1 under `EmptyPolicy(both_empty="best")` (the default), NaN under `"nan"`.
Exactly one empty: 0.

**Parameters:** none.

**Reference:** Dice LR. Measures of the amount of ecologic association between species. *Ecology* 26(3),
297–302 (1945). [doi:10.2307/1932409](https://doi.org/10.2307/1932409). Sørensen T. *Kongelige Danske
Videnskabernes Selskab, Biologiske Skrifter* 5(4), 1–34 (1948).

</div>

<div class="sek-metric" markdown>

### Intersection over union (Jaccard index) (IoU) {#iou}

<div class="sek-meta"><span class="sek-chip">key: <code>iou</code></span><span class="sek-chip">aliases: <code>jaccard</code>, <code>jac</code></span><span class="sek-chip">range [0, 1]</span><span class="sek-chip teal">higher is better</span><span class="sek-chip orange">dimensionless</span></div>

\[
\mathrm{IoU} = \frac{\lvert P\cap G\rvert}{\lvert P\cup G\rvert} = \frac{TP}{TP+FP+FN} = \frac{\mathrm{DSC}}{2-\mathrm{DSC}}
\]

**In plain words:** shared volume divided by the combined volume of both masks.

**Use it when:** your community reports IoU (computer vision, instance matching). IoU > 0.5 is the matching
rule behind [panoptic quality](detection.md#panoptic_quality).

**Watch out for:** IoU is a strictly monotone transform of Dice, so it ranks methods identically and adds no
information when reported next to it. It is numerically stricter: Dice 0.8 is IoU 0.67. All pitfalls of Dice
apply.

**Empty masks:** both empty: 1 (`"best"`) or NaN (`"nan"`). Exactly one empty: 0.

**Parameters:** none.

**Reference:** Jaccard P. The distribution of the flora in the alpine zone. *New Phytologist* 11(2), 37–50
(1912). [doi:10.1111/j.1469-8137.1912.tb05611.x](https://doi.org/10.1111/j.1469-8137.1912.tb05611.x)

</div>

<div class="sek-metric" markdown>

### Volumetric overlap error (VOE) {#voe}

<div class="sek-meta"><span class="sek-chip">key: <code>voe</code></span><span class="sek-chip">range [0, 1]</span><span class="sek-chip teal">lower is better</span><span class="sek-chip orange">dimensionless</span></div>

\[
\mathrm{VOE} = 1 - \mathrm{IoU} = 1 - \frac{TP}{TP+FP+FN}
\]

**In plain words:** the fraction of the union of both masks that is not shared.

**Use it when:** comparing with liver-segmentation benchmarks (SLIVER07, LiTS) that report VOE, often in percent.

**Watch out for:** it carries exactly the information of IoU (and Dice). SegEvalKit returns a fraction, not a
percentage.

**Empty masks:** both empty: 0 (`"best"`) or NaN (`"nan"`). Exactly one empty: 1.

**Parameters:** none.

**Reference:** Heimann T, van Ginneken B, Styner MA, et al. Comparison and evaluation of methods for liver
segmentation from CT datasets. *IEEE TMI* 28(8), 1251–1265 (2009).
[doi:10.1109/TMI.2009.2013851](https://doi.org/10.1109/TMI.2009.2013851)

</div>

<div class="sek-metric" markdown>

### Precision (positive predictive value) (PPV) {#precision}

<div class="sek-meta"><span class="sek-chip">key: <code>precision</code></span><span class="sek-chip">alias: <code>ppv</code></span><span class="sek-chip">range [0, 1]</span><span class="sek-chip teal">higher is better</span><span class="sek-chip orange">dimensionless</span></div>

\[
\mathrm{PPV} = \frac{TP}{TP+FP} = \frac{\lvert P\cap G\rvert}{\lvert P\rvert}
\]

**In plain words:** of everything the model marked, the fraction that is truly foreground. It penalises
over-segmentation and leakage into neighbouring structures.

**Use it when:** you need to separate over-segmentation from under-segmentation. Report it together with
[recall](#recall).

**Watch out for:** it is trivially maximised by predicting very little. Never report it alone.

**Empty masks:** both empty: 1 (`"best"`) or NaN (`"nan"`). Empty prediction with a non-empty reference: NaN
(the ratio is undefined; SegEvalKit does not silently return 0). Empty reference with a non-empty prediction: 0.

**Parameters:** none.

**Reference:** Taha AA, Hanbury A. Metrics for evaluating 3D medical image segmentation: analysis, selection,
and tool. *BMC Medical Imaging* 15, 29 (2015).
[doi:10.1186/s12880-015-0068-x](https://doi.org/10.1186/s12880-015-0068-x)

</div>

<div class="sek-metric" markdown>

### Recall (sensitivity, true-positive rate) (TPR) {#recall}

<div class="sek-meta"><span class="sek-chip">key: <code>recall</code></span><span class="sek-chip">aliases: <code>sensitivity</code>, <code>tpr</code></span><span class="sek-chip">range [0, 1]</span><span class="sek-chip teal">higher is better</span><span class="sek-chip orange">dimensionless</span></div>

\[
\mathrm{TPR} = \frac{TP}{TP+FN} = \frac{\lvert P\cap G\rvert}{\lvert G\rvert}
\]

**In plain words:** the fraction of the reference structure that the model captured.

**Use it when:** missing tissue is worse than adding it (tumour coverage, radiotherapy targets). Taha & Hanbury
(2015) recommend recall-type metrics in that setting.

**Watch out for:** it is trivially maximised by over-segmentation. Pair it with [precision](#precision) or Dice.

**Empty masks:** both empty: 1 (`"best"`) or NaN (`"nan"`). Empty reference with a non-empty prediction: NaN
(undefined). Empty prediction with a non-empty reference: 0.

**Parameters:** none.

**Reference:** Taha & Hanbury 2015,
[doi:10.1186/s12880-015-0068-x](https://doi.org/10.1186/s12880-015-0068-x).

</div>

<div class="sek-metric" markdown>

### Specificity (true-negative rate) (TNR) {#specificity}

<div class="sek-meta"><span class="sek-chip">key: <code>specificity</code></span><span class="sek-chip">alias: <code>tnr</code></span><span class="sek-chip">range [0, 1]</span><span class="sek-chip teal">higher is better</span><span class="sek-chip orange">dimensionless</span></div>

\[
\mathrm{TNR} = \frac{TN}{TN+FP}
\]

**In plain words:** the fraction of background correctly left as background.

**Use it when:** rarely, for 3D segmentation. It is useful as a sanity check and for image-level (presence)
classification.

**Watch out for:** *high class imbalance* and *field-of-view dependence* (Reinke et al. 2024, category P2).
A prediction that misses a 64-voxel lesion entirely and adds 64 false voxels elsewhere in a \(64^3\) volume still
scores specificity 0.9998 (see [Pitfalls](../guide/pitfalls.md#class-imbalance)). Cropping the image changes
\(TN\) and therefore the value.

**Empty masks:** not governed by the `EmptyPolicy`, because \(TN+FP\) is the reference background. Both masks
empty gives 1. The value is NaN only when the reference fills the entire image.

**Parameters:** none.

**Reference:** Taha & Hanbury 2015,
[doi:10.1186/s12880-015-0068-x](https://doi.org/10.1186/s12880-015-0068-x).

</div>

<div class="sek-metric" markdown>

### False-positive rate (fall-out) (FPR) {#fpr}

<div class="sek-meta"><span class="sek-chip">key: <code>fpr</code></span><span class="sek-chip">range [0, 1]</span><span class="sek-chip teal">lower is better</span><span class="sek-chip orange">dimensionless</span></div>

\[
\mathrm{FPR} = \frac{FP}{FP+TN} = 1 - \mathrm{TNR}
\]

**In plain words:** the fraction of background wrongly labelled foreground.

**Use it when:** building ROC-style analyses at the voxel level, or as a sanity check.

**Watch out for:** the same background domination as specificity: in 3D volumes it is close to 0 for almost any
prediction, so large absolute differences in false-positive volume look negligible.

**Empty masks:** not governed by the `EmptyPolicy`. Both empty gives 0; NaN only when the reference fills the
entire image.

**Parameters:** none.

**Reference:** Taha & Hanbury 2015,
[doi:10.1186/s12880-015-0068-x](https://doi.org/10.1186/s12880-015-0068-x).

</div>

<div class="sek-metric" markdown>

### False-negative rate (miss rate) (FNR) {#fnr}

<div class="sek-meta"><span class="sek-chip">key: <code>fnr</code></span><span class="sek-chip">range [0, 1]</span><span class="sek-chip teal">lower is better</span><span class="sek-chip orange">dimensionless</span></div>

\[
\mathrm{FNR} = \frac{FN}{FN+TP} = 1 - \mathrm{TPR}
\]

**In plain words:** the fraction of the reference structure that was missed.

**Use it when:** you prefer to report an error rate rather than recall. It carries the same information.

**Watch out for:** trivially minimised by over-segmentation.

**Empty masks:** computed as \(1 - \mathrm{recall}\): both empty gives 0 (`"best"`) or NaN (`"nan"`); empty
reference with a non-empty prediction gives NaN; empty prediction with a non-empty reference gives 1.

**Parameters:** none.

**Reference:** Taha & Hanbury 2015,
[doi:10.1186/s12880-015-0068-x](https://doi.org/10.1186/s12880-015-0068-x).

</div>

<div class="sek-metric" markdown>

### F-beta score (Fβ) {#fbeta}

<div class="sek-meta"><span class="sek-chip">key: <code>fbeta</code></span><span class="sek-chip">range [0, 1]</span><span class="sek-chip teal">higher is better</span><span class="sek-chip orange">dimensionless</span></div>

\[
F_\beta = \frac{(1+\beta^2)\,TP}{(1+\beta^2)\,TP + \beta^2\,FN + FP}
\]

\(\beta\) sets the relative importance of recall: \(\beta > 1\) weights false negatives more, \(\beta < 1\)
weights false positives more, and \(\beta = 1\) is Dice.

**In plain words:** a Dice-like score in which you decide whether missed tissue or extra tissue is worse.

**Use it when:** false positives and false negatives have clearly unequal clinical cost. Metrics Reloaded selects
F\(_\beta\) over Dice in that case.

**Watch out for:** the choice of \(\beta\) must be fixed and justified before looking at results. It keeps Dice's
size dependence. The SegEvalKit default is \(\beta = 2\), not 1; pass `{"fbeta": {"beta": 1.0}}` for F1.

**Empty masks:** both empty: 1 (`"best"`) or NaN (`"nan"`). Exactly one empty: 0.

**Parameters:** `beta = 2.0`.

**Reference:** van Rijsbergen CJ. *Information Retrieval*, 2nd ed. Butterworths (1979), ch. 7.
[Online edition](http://www.dcs.gla.ac.uk/Keith/Preface.html). Maier-Hein L, Reinke A, Godau P, et al. Metrics
reloaded: recommendations for image analysis validation. *Nature Methods* 21, 195–212 (2024).
[doi:10.1038/s41592-023-02151-z](https://doi.org/10.1038/s41592-023-02151-z)

</div>

<div class="sek-metric" markdown>

### Tversky index (TI) {#tversky}

<div class="sek-meta"><span class="sek-chip">key: <code>tversky</code></span><span class="sek-chip">range [0, 1]</span><span class="sek-chip teal">higher is better</span><span class="sek-chip orange">dimensionless</span></div>

\[
\mathrm{TI}_{\alpha,\beta} = \frac{TP}{TP + \alpha\,FP + \beta\,FN}
\]

\(\alpha\) weights false positives and \(\beta\) weights false negatives. \(\alpha=\beta=0.5\) gives Dice,
\(\alpha=\beta=1\) gives IoU.

**In plain words:** an asymmetric Dice with separate penalties for extra and missing voxels.

**Use it when:** you trained with a Tversky loss and want the matching evaluation score, or you need an
asymmetric overlap score parameterised directly by the two error weights.

**Watch out for:** with the defaults (\(\alpha=0.3\), \(\beta=0.7\)) missed voxels cost more than twice as much as
extra voxels. State \(\alpha\) and \(\beta\) whenever you report it.

**Empty masks:** both empty: 1 (`"best"`) or NaN (`"nan"`). Exactly one empty: 0.

**Parameters:** `alpha = 0.3`, `beta = 0.7`.

**Reference:** Tversky A. Features of similarity. *Psychological Review* 84(4), 327–352 (1977).
[doi:10.1037/0033-295X.84.4.327](https://doi.org/10.1037/0033-295X.84.4.327). Salehi SSM, Erdogmus D,
Gholipour A. Tversky loss function for image segmentation using 3D fully convolutional deep networks.
*MLMI 2017*, LNCS 10541. [arXiv:1706.05721](https://arxiv.org/abs/1706.05721)

</div>

<div class="sek-metric" markdown>

### Voxel accuracy (Acc) {#accuracy}

<div class="sek-meta"><span class="sek-chip">key: <code>accuracy</code></span><span class="sek-chip">range [0, 1]</span><span class="sek-chip teal">higher is better</span><span class="sek-chip orange">dimensionless</span></div>

\[
\mathrm{Acc} = \frac{TP+TN}{TP+FP+FN+TN}
\]

**In plain words:** the fraction of all voxels labelled correctly.

**Use it when:** almost never as a segmentation score. It is included for completeness and comparison with
older literature.

**Watch out for:** background dominates it completely. A prediction with Dice 0 can have accuracy 0.9995
(see [Pitfalls](../guide/pitfalls.md#class-imbalance)). It changes with the field of view.

**Empty masks:** always defined; not governed by the `EmptyPolicy`. Both empty gives 1.

**Parameters:** none.

**Reference:** Taha & Hanbury 2015,
[doi:10.1186/s12880-015-0068-x](https://doi.org/10.1186/s12880-015-0068-x).

</div>

<div class="sek-metric" markdown>

### Balanced accuracy (BAcc) {#balanced_accuracy}

<div class="sek-meta"><span class="sek-chip">key: <code>balanced_accuracy</code></span><span class="sek-chip">range [0, 1]</span><span class="sek-chip teal">higher is better</span><span class="sek-chip orange">dimensionless</span></div>

\[
\mathrm{BAcc} = \tfrac{1}{2}\left(\frac{TP}{TP+FN} + \frac{TN}{TN+FP}\right)
\]

**In plain words:** the average of sensitivity and specificity. For a binary (thresholded) prediction it equals
the area under the one-point ROC curve, the "AUC" of Taha & Hanbury (2015).

**Use it when:** you need a TN-aware score that does not reward predicting everything as background, for example
in image-level presence classification.

**Watch out for:** in 3D the specificity half is close to 1, so balanced accuracy is roughly \((1+\mathrm{TPR})/2\)
and rarely drops below 0.5. An empty prediction scores 0.5, not 0. For probabilistic outputs use the proper
[AUROC](calibration.md#auroc).

**Empty masks:** empty reference: 1 (`"best"`) or NaN (`"nan"`) when the prediction is also empty, NaN otherwise.
Empty prediction with a non-empty reference: 0.5 when the reference does not fill the image.

**Parameters:** none.

**Reference:** Brodersen KH, Ong CS, Stephan KE, Buhmann JM. The balanced accuracy and its posterior
distribution. *ICPR 2010*, 3121–3124. [doi:10.1109/ICPR.2010.764](https://doi.org/10.1109/ICPR.2010.764).
Taha & Hanbury 2015, [doi:10.1186/s12880-015-0068-x](https://doi.org/10.1186/s12880-015-0068-x).

</div>

## Code

The `overlap` metric set contains `dice`, `iou`, `precision`, `recall`, `specificity`, `voe`, `fbeta`,
`tversky`, `mcc`, `cohen_kappa` and `balanced_accuracy`. Parameters are passed per metric through `params`.

```python
import numpy as np
from segevalkit.metrics import compute_metrics

ref = np.zeros((32, 32, 32), bool)
ref[8:24, 8:24, 8:24] = True            # 16^3 = 4096-voxel cube
pred = np.zeros_like(ref)
pred[10:24, 8:24, 8:26] = True          # misses 2 slices, leaks 2 slices

scores = compute_metrics(
    pred, ref,
    metrics=["overlap", "fnr", "accuracy"],
    params={"fbeta": {"beta": 1.0}},    # F1 instead of the default F2
)
for name, value in scores.items():
    print(f"{name:18s} {value:.4f}")
```

```text
dice               0.8819
iou                0.7887
precision          0.8889
recall             0.8750
specificity        0.9844
voe                0.2113
fbeta              0.8819
tversky            0.8791
mcc                0.8652
cohen_kappa        0.8652
balanced_accuracy  0.9297
fnr                0.1250
accuracy           0.9707
```

With \(\beta = 1\), `fbeta` equals `dice`. Overlap metrics need no voxel spacing: the same arrays give the same
numbers at any resolution. The same parameters are accepted by
[`Evaluator(params=...)`](index.md#computing-metrics) and on the command line.

## See also

- [Distance & boundary metrics](distance.md): the recommended companion to any overlap metric.
- [Detection metrics](detection.md): when the unit of interest is the lesion, not the voxel.
- [Pitfalls](../guide/pitfalls.md): size bias, class imbalance and aggregation.
