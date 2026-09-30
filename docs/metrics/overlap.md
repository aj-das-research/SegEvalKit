# Overlap metrics

Overlap metrics count voxels: every one is a function of the voxel confusion matrix, so they answer **how many
voxels were labelled correctly**, not *where* the wrong ones are. They are cheap, bounded and spacing-free. Metrics
that use \(TN\) depend on the field of view and sit near 1 in 3D volumes.

## At a glance

| Metric | Key | Uses \(TN\) | Direction | Symmetric | Main blind spot |
|---|---|:-:|---|:-:|---|
| [Dice](#dice) | `dice` | no | ↑ | yes | size bias; error location |
| [IoU](#iou) | `iou` | no | ↑ | yes | monotone in Dice |
| [VOE](#voe) | `voe` | no | ↓ | yes | \(1-\mathrm{IoU}\) |
| [Precision](#precision) | `precision` | no | ↑ | no | rewards under-segmentation |
| [Recall](#recall) | `recall` | no | ↑ | no | rewards over-segmentation |
| [Specificity](#specificity) | `specificity` | yes | ↑ | no | \(\approx 1\) in 3D |
| [FPR](#fpr) | `fpr` | yes | ↓ | no | \(\approx 0\) in 3D |
| [FNR](#fnr) | `fnr` | no | ↓ | no | \(1-\mathrm{recall}\) |
| [F\(_\beta\)](#fbeta) | `fbeta` | no | ↑ | if \(\beta=1\) | \(\beta\) must be justified |
| [Tversky](#tversky) | `tversky` | no | ↑ | if \(\alpha=\beta\) | \(\alpha,\beta\) must be justified |
| [Accuracy](#accuracy) | `accuracy` | yes | ↑ | yes | background dominates |
| [Balanced accuracy](#balanced_accuracy) | `balanced_accuracy` | yes | ↑ | no | specificity half \(\approx 1\) |

**Identities.** \(\mathrm{IoU} = \mathrm{DSC}/(2-\mathrm{DSC})\); Dice is the harmonic mean of precision and
recall; Tversky with \(\alpha=\beta=0.5\) is Dice, with \(\alpha=\beta=1\) IoU; F\(_\beta\) is Tversky with
\(\alpha = 1/(1+\beta^2)\), \(\beta_{\mathrm{TI}} = \beta^2/(1+\beta^2)\). Reporting several adds little (Taha &
Hanbury 2015). [MCC](agreement.md#mcc) and [Cohen's kappa](agreement.md#cohen_kappa) live in the `agreement`
family but belong to the `overlap` set.

## Notation

\(G\) and \(P\) are the reference and predicted foreground voxel sets (any non-zero voxel; 2D is one slice) and
\(N\) the image size: \(TP = \lvert P\cap G\rvert\), \(FP = \lvert P\setminus G\rvert\),
\(FN = \lvert G\setminus P\rvert\), \(TN = N-TP-FP-FN\). Counts are exact integers on CPU and GPU.
**Empty masks** follow `EmptyPolicy(both_empty=...)`: `"best"` (default) gives the ideal value, `"nan"` gives NaN.

## Metrics

<div class="sek-metric" markdown>

### Dice similarity coefficient (DSC) {#dice}

<div class="sek-meta"><span class="sek-chip">dice · dsc · f1</span><span class="sek-chip">[0, 1]</span><span class="sek-chip teal">↑ higher is better</span></div>

\[
\mathrm{DSC} = \frac{2\lvert P\cap G\rvert}{\lvert P\rvert+\lvert G\rvert} = \frac{2\,TP}{2\,TP+FP+FN}
\]

**In words** Shared voxels relative to the average size of the two masks; 1 is identical, 0 no overlap.

**Use when** The default overlap score for medium and large structures; pair it with a boundary metric such as [NSD](distance.md#nsd).

**Watch out** Unstable on small structures ([size bias](../guide/pitfalls.md#size-bias)); blind to error location, distance of misses and topology ([clDice](topology.md#cldice)).

??? info "Details: empty masks, parameters, reference"
    **Empty masks.** Both empty: 1 (`"best"`) or NaN (`"nan"`). One empty: 0.

    **Example.** A one-voxel shift costs a 3×3×3 cube a third of its Dice (0.667) but a 30×30×30 cube only 3 % (0.967), so mean Dice across structures of different size is not comparable (Reinke et al. 2024, P2). Very different error patterns can give the same Dice, and a mean hides catastrophic failures: report the distribution.

    **Reference.** Dice LR. *Ecology* 26(3):297–302, 1945. [doi:10.2307/1932409](https://doi.org/10.2307/1932409). Sørensen T. *Biol. Skr. K. Dan. Vidensk. Selsk.* 5(4):1–34, 1948.

</div>

<div class="sek-metric" markdown>

### Intersection over union (IoU; Jaccard index) {#iou}

<div class="sek-meta"><span class="sek-chip">iou · jaccard · jac</span><span class="sek-chip">[0, 1]</span><span class="sek-chip teal">↑ higher is better</span></div>

\[
\mathrm{IoU} = \frac{\lvert P\cap G\rvert}{\lvert P\cup G\rvert} = \frac{TP}{TP+FP+FN} = \frac{\mathrm{DSC}}{2-\mathrm{DSC}}
\]

**In words** Shared volume divided by the combined volume of both masks.

**Use when** Your community reports IoU (vision, instance matching); IoU > 0.5 is the matching rule of [panoptic quality](detection.md#panoptic_quality).

**Watch out** A monotone transform of Dice: same ranking, no extra information, numerically stricter (Dice 0.8 = IoU 0.67).

??? info "Details: empty masks, parameters, reference"
    **Empty masks.** Both empty: 1 (`"best"`) or NaN (`"nan"`). One empty: 0.

    **Reference.** Jaccard P. *New Phytologist* 11(2):37–50, 1912. [doi:10.1111/j.1469-8137.1912.tb05611.x](https://doi.org/10.1111/j.1469-8137.1912.tb05611.x)

</div>

<div class="sek-metric" markdown>

### Volumetric overlap error (VOE) {#voe}

<div class="sek-meta"><span class="sek-chip">voe</span><span class="sek-chip">[0, 1]</span><span class="sek-chip teal">↓ lower is better</span></div>

\[
\mathrm{VOE} = 1 - \mathrm{IoU} = 1 - \frac{TP}{TP+FP+FN}
\]

**In words** The fraction of the union of both masks that is not shared.

**Use when** Comparing with liver benchmarks (SLIVER07, LiTS) that report VOE, often in percent.

**Watch out** Carries exactly the information of IoU and Dice; SegEvalKit returns a fraction, not a percentage.

??? info "Details: empty masks, parameters, reference"
    **Empty masks.** Both empty: 0 (`"best"`) or NaN (`"nan"`). One empty: 1.

    **Reference.** Heimann T et al. *IEEE TMI* 28(8):1251–1265, 2009. [doi:10.1109/TMI.2009.2013851](https://doi.org/10.1109/TMI.2009.2013851)

</div>

<div class="sek-metric" markdown>

### Precision (PPV; positive predictive value) {#precision}

<div class="sek-meta"><span class="sek-chip">precision · ppv</span><span class="sek-chip">[0, 1]</span><span class="sek-chip teal">↑ higher is better</span></div>

\[
\mathrm{PPV} = \frac{TP}{TP+FP} = \frac{\lvert P\cap G\rvert}{\lvert P\rvert}
\]

**In words** Of everything the model marked, the fraction that is truly foreground; it penalises over-segmentation and leakage.

**Use when** Separating over- from under-segmentation, always together with [recall](#recall).

**Watch out** Trivially maximised by predicting very little; never report it alone.

??? info "Details: empty masks, parameters, reference"
    **Empty masks.** Both empty: 1 (`"best"`) or NaN (`"nan"`). Empty prediction, non-empty reference: NaN (undefined; not silently 0). Empty reference, non-empty prediction: 0.

    **Reference.** Taha AA, Hanbury A. *BMC Medical Imaging* 15:29, 2015. [doi:10.1186/s12880-015-0068-x](https://doi.org/10.1186/s12880-015-0068-x)

</div>

<div class="sek-metric" markdown>

### Recall (TPR; sensitivity, true-positive rate) {#recall}

<div class="sek-meta"><span class="sek-chip">recall · sensitivity · tpr</span><span class="sek-chip">[0, 1]</span><span class="sek-chip teal">↑ higher is better</span></div>

\[
\mathrm{TPR} = \frac{TP}{TP+FN} = \frac{\lvert P\cap G\rvert}{\lvert G\rvert}
\]

**In words** The fraction of the reference structure that the model captured.

**Use when** Missing tissue is worse than adding it (tumour coverage, radiotherapy targets), as Taha & Hanbury (2015) recommend.

**Watch out** Trivially maximised by over-segmentation; pair it with [precision](#precision) or Dice.

??? info "Details: empty masks, parameters, reference"
    **Empty masks.** Both empty: 1 (`"best"`) or NaN (`"nan"`). Empty reference, non-empty prediction: NaN (undefined). Empty prediction, non-empty reference: 0.

    **Reference.** Taha & Hanbury 2015, [doi:10.1186/s12880-015-0068-x](https://doi.org/10.1186/s12880-015-0068-x).

</div>

<div class="sek-metric" markdown>

### Specificity (TNR; true-negative rate) {#specificity}

<div class="sek-meta"><span class="sek-chip">specificity · tnr</span><span class="sek-chip">[0, 1]</span><span class="sek-chip teal">↑ higher is better</span></div>

\[
\mathrm{TNR} = \frac{TN}{TN+FP}
\]

**In words** The fraction of background correctly left as background.

**Use when** Rarely in 3D: as a sanity check or for image-level presence classification.

**Watch out** Dominated by background and changed by cropping ([class imbalance](../guide/pitfalls.md#class-imbalance), Reinke et al. 2024, P2).

??? info "Details: empty masks, parameters, reference"
    **Example.** Missing a 64-voxel lesion entirely and adding 64 false voxels elsewhere in a \(64^3\) volume still scores 0.9998.

    **Empty masks.** Not governed by `EmptyPolicy` (the denominator is the reference background). Both empty: 1. NaN only when the reference fills the image.

    **Reference.** Taha & Hanbury 2015, [doi:10.1186/s12880-015-0068-x](https://doi.org/10.1186/s12880-015-0068-x).

</div>

<div class="sek-metric" markdown>

### False-positive rate (FPR; fall-out) {#fpr}

<div class="sek-meta"><span class="sek-chip">fpr</span><span class="sek-chip">[0, 1]</span><span class="sek-chip teal">↓ lower is better</span></div>

\[
\mathrm{FPR} = \frac{FP}{FP+TN} = 1 - \mathrm{TNR}
\]

**In words** The fraction of background wrongly labelled foreground.

**Use when** Building voxel-level ROC-style analyses, or as a sanity check.

**Watch out** Near 0 for almost any 3D prediction, so large false-positive volumes look negligible.

??? info "Details: empty masks, parameters, reference"
    **Empty masks.** Not governed by `EmptyPolicy`. Both empty: 0. NaN only when the reference fills the image.

    **Reference.** Taha & Hanbury 2015, [doi:10.1186/s12880-015-0068-x](https://doi.org/10.1186/s12880-015-0068-x).

</div>

<div class="sek-metric" markdown>

### False-negative rate (FNR; miss rate) {#fnr}

<div class="sek-meta"><span class="sek-chip">fnr</span><span class="sek-chip">[0, 1]</span><span class="sek-chip teal">↓ lower is better</span></div>

\[
\mathrm{FNR} = \frac{FN}{FN+TP} = 1 - \mathrm{TPR}
\]

**In words** The fraction of the reference structure that was missed.

**Use when** You prefer an error rate to recall; it carries the same information.

**Watch out** Trivially minimised by over-segmentation.

??? info "Details: empty masks, parameters, reference"
    **Empty masks.** Computed as \(1-\mathrm{recall}\): both empty 0 (`"best"`) or NaN; empty reference with non-empty prediction NaN; empty prediction with non-empty reference 1.

    **Reference.** Taha & Hanbury 2015, [doi:10.1186/s12880-015-0068-x](https://doi.org/10.1186/s12880-015-0068-x).

</div>

<div class="sek-metric" markdown>

### F-beta score (Fβ) {#fbeta}

<div class="sek-meta"><span class="sek-chip">fbeta</span><span class="sek-chip">[0, 1]</span><span class="sek-chip teal">↑ higher is better</span></div>

\[
F_\beta = \frac{(1+\beta^2)\,TP}{(1+\beta^2)\,TP + \beta^2\,FN + FP}
\]

**In words** A Dice-like score where \(\beta>1\) weights misses more, \(\beta<1\) extras more, and \(\beta=1\) is Dice.

**Use when** False positives and negatives have clearly unequal clinical cost; Metrics Reloaded then prefers F\(_\beta\) to Dice.

**Watch out** The default is \(\beta=2\), not 1; fix and justify \(\beta\) before seeing results; Dice's size bias remains.

??? info "Details: empty masks, parameters, reference"
    **Parameters.** `beta = 2.0`; pass `{"fbeta": {"beta": 1.0}}` for F1.

    **Empty masks.** Both empty: 1 (`"best"`) or NaN (`"nan"`). One empty: 0.

    **Reference.** van Rijsbergen CJ. *Information Retrieval*, 2nd ed., ch. 7, 1979. [Online edition](http://www.dcs.gla.ac.uk/Keith/Preface.html). Maier-Hein L et al. Metrics reloaded. *Nature Methods* 21:195–212, 2024. [doi:10.1038/s41592-023-02151-z](https://doi.org/10.1038/s41592-023-02151-z)

</div>

<div class="sek-metric" markdown>

### Tversky index (TI) {#tversky}

<div class="sek-meta"><span class="sek-chip">tversky</span><span class="sek-chip">[0, 1]</span><span class="sek-chip teal">↑ higher is better</span></div>

\[
\mathrm{TI}_{\alpha,\beta} = \frac{TP}{TP + \alpha\,FP + \beta\,FN}
\]

**In words** An asymmetric Dice with separate weights \(\alpha\) for extra and \(\beta\) for missing voxels.

**Use when** You trained with a Tversky loss and want the matching score, or need asymmetric error weights directly.

**Watch out** With the defaults a missed voxel costs more than twice an extra one; always state \(\alpha\) and \(\beta\).

??? info "Details: empty masks, parameters, reference"
    **Parameters.** `alpha = 0.3`, `beta = 0.7`. \(\alpha=\beta=0.5\) gives Dice, \(\alpha=\beta=1\) IoU.

    **Empty masks.** Both empty: 1 (`"best"`) or NaN (`"nan"`). One empty: 0.

    **Reference.** Tversky A. *Psychological Review* 84(4):327–352, 1977. [doi:10.1037/0033-295X.84.4.327](https://doi.org/10.1037/0033-295X.84.4.327). Salehi SSM et al. Tversky loss. *MLMI 2017*, LNCS 10541. [arXiv:1706.05721](https://arxiv.org/abs/1706.05721)

</div>

<div class="sek-metric" markdown>

### Voxel accuracy (Acc) {#accuracy}

<div class="sek-meta"><span class="sek-chip">accuracy</span><span class="sek-chip">[0, 1]</span><span class="sek-chip teal">↑ higher is better</span></div>

\[
\mathrm{Acc} = \frac{TP+TN}{TP+FP+FN+TN}
\]

**In words** The fraction of all voxels labelled correctly.

**Use when** Almost never as a segmentation score; included for comparison with older literature.

**Watch out** Background dominates it: a Dice-0 prediction can score 0.9995 ([class imbalance](../guide/pitfalls.md#class-imbalance)); it changes with the field of view.

??? info "Details: empty masks, parameters, reference"
    **Empty masks.** Always defined; not governed by `EmptyPolicy`. Both empty: 1.

    **Reference.** Taha & Hanbury 2015, [doi:10.1186/s12880-015-0068-x](https://doi.org/10.1186/s12880-015-0068-x).

</div>

<div class="sek-metric" markdown>

### Balanced accuracy (BAcc) {#balanced_accuracy}

<div class="sek-meta"><span class="sek-chip">balanced_accuracy</span><span class="sek-chip">[0, 1]</span><span class="sek-chip teal">↑ higher is better</span></div>

\[
\mathrm{BAcc} = \tfrac{1}{2}\left(\frac{TP}{TP+FN} + \frac{TN}{TN+FP}\right)
\]

**In words** The mean of sensitivity and specificity; for a binary prediction, the one-point ROC "AUC" of Taha & Hanbury (2015).

**Use when** You need a TN-aware score that does not reward all-background, e.g. image-level presence classification.

**Watch out** In 3D it is roughly \((1+\mathrm{TPR})/2\): an empty prediction scores 0.5; use [AUROC](calibration.md#auroc) for probabilities.

??? info "Details: empty masks, parameters, reference"
    **Empty masks.** Empty reference: 1 (`"best"`) or NaN (`"nan"`) if the prediction is also empty, NaN otherwise. Empty prediction with non-empty reference: 0.5 (unless the reference fills the image).

    **Reference.** Brodersen KH et al. *ICPR 2010*, 3121–3124. [doi:10.1109/ICPR.2010.764](https://doi.org/10.1109/ICPR.2010.764). Taha & Hanbury 2015, [doi:10.1186/s12880-015-0068-x](https://doi.org/10.1186/s12880-015-0068-x).

</div>

## Code

The `overlap` set is `dice`, `iou`, `precision`, `recall`, `specificity`, `voe`, `fbeta`, `tversky`, `mcc`,
`cohen_kappa`, `balanced_accuracy`. Per-metric parameters go in `params` (also accepted by
[`Evaluator(params=...)`](index.md#computing-metrics) and the CLI); no spacing is needed.

```python
import numpy as np
from segevalkit.metrics import compute_metrics

ref = np.zeros((32, 32, 32), bool); ref[8:24, 8:24, 8:24] = True   # 16^3 cube
pred = np.zeros_like(ref); pred[10:24, 8:24, 8:26] = True          # misses 2, leaks 2 slices
scores = compute_metrics(pred, ref, metrics=["overlap", "fnr", "accuracy"],
                         params={"fbeta": {"beta": 1.0}})          # F1 instead of F2
# dice 0.8819  iou 0.7887  precision 0.8889  recall 0.8750  specificity 0.9844
# voe 0.2113  fbeta 0.8819  tversky 0.8791  mcc 0.8652  cohen_kappa 0.8652
# balanced_accuracy 0.9297  fnr 0.1250  accuracy 0.9707
```

See also [distance metrics](distance.md) (the recommended companion), [detection](detection.md) (lesion-level)
and [pitfalls](../guide/pitfalls.md).
