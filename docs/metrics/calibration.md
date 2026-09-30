# Probabilistic & calibration metrics

A thresholded mask discards the model's confidence. When a foreground probability map is available, two further
questions can be asked:

- **Discrimination** ([AUROC](#auroc), [AUPRC](#auprc)): does the model rank foreground voxels above background
  voxels?
- **Calibration** ([ECE](#ece), [Brier](#brier), [NLL](#nll)): when the model says 80 %, is it right 80 % of the
  time? This matters whenever probabilities are shown to a clinician, thresholded at a non-default value, or
  propagated into downstream decisions.

[Soft Dice](#soft_dice) sits in between: an overlap score computed on probabilities.

These metrics say nothing about the geometry of the thresholded mask. A perfectly calibrated model can still
segment poorly, and a model with excellent Dice can be badly over-confident (Mehrtash et al. 2020).

## Inputs and region of interest

Every metric on this page needs `prob`, a float array in \([0, 1]\) of the same shape as the masks. Without it,
`compute_metrics` and `Evaluator` **skip** these metrics (they do not appear in the output) rather than return
NaN. The `Evaluator` reads per-structure probability maps from a folder (`prob=` in `Evaluator.evaluate`).

3D volumes are overwhelmingly background, and whole-volume calibration is dominated by easy background voxels
with \(p \approx 0\). Every metric except soft Dice therefore takes a region of interest:

`roi="all"` (default)
:   The whole image: the literal definition.

`roi="band"`
:   The union of the reference foreground and the predicted foreground (\(p \ge 0.5\)), grown by `margin_mm`
    (Euclidean distance with the voxel spacing): the region where the model is actually uncertain, following
    Mehrtash et al. (2020). If both are empty the band is empty and the metric is NaN.

## Notation

\(p_i \in [0, 1]\) is the predicted foreground probability of voxel \(i\) and \(g_i \in \{0, 1\}\) its reference
label; \(N\) is the number of voxels in the region of interest. For calibration, SegEvalKit uses the top-label
view of Guo et al. (2017): the predicted label is \(\hat y_i = [p_i \ge 0.5]\), its confidence
\(c_i = \max(p_i, 1-p_i) \in [0.5, 1]\), and a voxel is correct if \(\hat y_i = g_i\).

## At a glance

| Metric | Key | Measures | Proper scoring rule | ROI | Background-dominated with `roi="all"` |
|---|---|---|:-:|:-:|:-:|
| [Soft Dice](#soft_dice) | `soft_dice` | soft overlap | no | whole image | no (ignores TN) |
| [AUROC](#auroc) | `auroc` | discrimination | no | yes | yes |
| [AUPRC](#auprc) | `auprc` | discrimination under imbalance | no | yes | less |
| [Brier](#brier) | `brier` | calibration + sharpness | yes | yes | yes |
| [NLL](#nll) | `nll` | calibration + sharpness | yes | yes | yes |
| [ECE](#ece) | `ece` | calibration | no | yes | yes |

## Metrics

<div class="sek-metric" markdown>

### Soft Dice (sDSC) {#soft_dice}

<div class="sek-meta"><span class="sek-chip">key: <code>soft_dice</code></span><span class="sek-chip">range [0, 1]</span><span class="sek-chip teal">higher is better</span><span class="sek-chip orange">dimensionless</span><span class="sek-chip">needs probabilities</span></div>

\[
\mathrm{sDSC} = \frac{2\sum_i p_i g_i}{\sum_i p_i + \sum_i g_i}
\]

summed over the whole image.

**In plain words:** Dice with probabilities in place of a thresholded mask; confident correct voxels count fully,
hesitant ones partially.

**Use it when:** comparing models that were trained with a soft Dice loss, or checking how much confidence is
spread outside the object.

**Watch out for:** it is not a calibration measure: a perfectly calibrated model with \(p = 0.9\) inside the object
scores below 1. Low-probability mass spread over a large background lowers it even when the thresholded mask is
perfect.

**Empty masks:** if \(\sum_i p_i + \sum_i g_i = 0\) (empty reference and an all-zero probability map): 1
(`"best"`) or NaN.

**Parameters:** none (no `roi`).

**Reference:** Milletari F, Navab N, Ahmadi SA. V-Net: fully convolutional neural networks for volumetric medical
image segmentation. *3DV 2016*, 565–571. [doi:10.1109/3DV.2016.79](https://doi.org/10.1109/3DV.2016.79)

</div>

<div class="sek-metric" markdown>

### Area under the ROC curve (AUROC) {#auroc}

<div class="sek-meta"><span class="sek-chip">key: <code>auroc</code></span><span class="sek-chip">range [0, 1]</span><span class="sek-chip teal">higher is better</span><span class="sek-chip orange">dimensionless</span><span class="sek-chip">needs probabilities</span></div>

\[
\mathrm{AUROC} = \int_0^1 \mathrm{TPR}\; d\,\mathrm{FPR} = P\big(p_i > p_j \mid g_i = 1,\, g_j = 0\big)
\]

computed as the Mann–Whitney statistic on a 1024-bin histogram of the scores (ties within a bin count one half),
which is exact up to the bin width and runs in \(O(N)\).

**In plain words:** the probability that a randomly chosen foreground voxel gets a higher score than a randomly
chosen background voxel.

**Use it when:** you need a threshold-free discrimination measure, preferably inside `roi="band"`.

**Watch out for:** with `roi="all"` it is dominated by the huge number of easy background voxels and is close to
1 for almost any reasonable model. Prefer [AUPRC](#auprc) under heavy imbalance. It says nothing about
calibration: any monotone rescaling of \(p\) leaves it unchanged.

**Empty masks:** NaN whenever the region contains no foreground or no background voxel (not governed by the
`EmptyPolicy`).

**Parameters:** `roi = "all"`, `margin_mm = 10.0`.

**Reference:** Hanley JA, McNeil BJ. The meaning and use of the area under a receiver operating characteristic
(ROC) curve. *Radiology* 143(1), 29–36 (1982).
[doi:10.1148/radiology.143.1.7063747](https://doi.org/10.1148/radiology.143.1.7063747)

</div>

<div class="sek-metric" markdown>

### Area under the precision-recall curve (average precision) (AP) {#auprc}

<div class="sek-meta"><span class="sek-chip">key: <code>auprc</code></span><span class="sek-chip">range [0, 1]</span><span class="sek-chip teal">higher is better</span><span class="sek-chip orange">dimensionless</span><span class="sek-chip">needs probabilities</span></div>

\[
\mathrm{AP} = \sum_k (R_k - R_{k-1})\,P_k
\]

over descending thresholds \(k\) (the 1024 histogram bin edges), with \(P_k\) and \(R_k\) the voxel precision and
recall at threshold \(k\) and \(R_0 = 0\).

**In plain words:** average precision over all thresholds: how well the model ranks foreground voxels first,
judged by the precision it keeps as recall grows.

**Use it when:** the foreground is a small fraction of the region, which is always the case in 3D. Unlike AUROC it
stays informative under heavy class imbalance.

**Watch out for:** its baseline is the foreground prevalence, not 0.5, so values are not comparable between
structures or regions with different prevalence. It is a discrimination measure, not a calibration measure.

**Empty masks:** NaN when the region contains no reference foreground.

**Parameters:** `roi = "all"`, `margin_mm = 10.0`.

**Reference:** Saito T, Rehmsmeier M. The precision-recall plot is more informative than the ROC plot when
evaluating binary classifiers on imbalanced datasets. *PLoS ONE* 10(3), e0118432 (2015).
[doi:10.1371/journal.pone.0118432](https://doi.org/10.1371/journal.pone.0118432)

</div>

<div class="sek-metric" markdown>

### Brier score (BS) {#brier}

<div class="sek-meta"><span class="sek-chip">key: <code>brier</code></span><span class="sek-chip">range [0, 1]</span><span class="sek-chip teal">lower is better</span><span class="sek-chip orange">dimensionless</span><span class="sek-chip">needs probabilities</span></div>

\[
\mathrm{BS} = \frac1N\sum_{i=1}^{N} (p_i - g_i)^2
\]

**In plain words:** the mean squared difference between predicted probability and the true 0/1 label.

**Use it when:** you want a proper scoring rule that rewards both calibration and sharpness; it is the standard
companion to ECE (Mehrtash et al. 2020).

**Watch out for:** with `roi="all"` it is diluted by background: in the example below it is 0.011 over the whole
image and 0.057 in the band. Absolute values are hard to interpret without a baseline; compare methods on the
same region.

**Empty masks:** NaN when the region is empty (only possible with `roi="band"`).

**Parameters:** `roi = "all"`, `margin_mm = 10.0`.

**Reference:** Brier GW. Verification of forecasts expressed in terms of probability. *Monthly Weather Review*
78(1), 1–3 (1950).
[doi:10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2](https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2)

</div>

<div class="sek-metric" markdown>

### Negative log-likelihood (NLL) {#nll}

<div class="sek-meta"><span class="sek-chip">key: <code>nll</code></span><span class="sek-chip">range [0, ∞)</span><span class="sek-chip teal">lower is better</span><span class="sek-chip orange">nats per voxel</span><span class="sek-chip">needs probabilities</span></div>

\[
\mathrm{NLL} = -\frac1N\sum_{i=1}^{N} \big[g_i\log p_i + (1-g_i)\log(1-p_i)\big]
\]

with \(p_i\) clipped to \([10^{-7},\, 1-10^{-7}]\), so a single voxel contributes at most about 16.1 nats.

**In plain words:** the average surprise of the true label under the predicted probabilities. Confident mistakes
are punished very hard.

**Use it when:** comparing calibration methods (temperature scaling, ensembling) with a proper scoring rule.

**Watch out for:** it is dominated by a few confidently wrong voxels and, with `roi="all"`, by the background. The
clipping constant bounds the penalty and must be the same when comparing numbers across tools.

**Empty masks:** NaN when the region is empty.

**Parameters:** `roi = "all"`, `margin_mm = 10.0`.

**Reference:** Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. *ICML 2017*,
PMLR 70, 1321–1330. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)

</div>

<div class="sek-metric" markdown>

### Expected calibration error (ECE) {#ece}

<div class="sek-meta"><span class="sek-chip">key: <code>ece</code></span><span class="sek-chip">range [0, 1]</span><span class="sek-chip teal">lower is better</span><span class="sek-chip orange">dimensionless</span><span class="sek-chip">needs probabilities</span></div>

\[
\mathrm{ECE} = \sum_{b=1}^{B}\frac{\lvert B_b\rvert}{N}\,\big\lvert\mathrm{acc}(B_b) - \mathrm{conf}(B_b)\big\rvert
\]

Voxels are assigned to \(B\) equal-width bins of their top-label confidence \(c_i \in [0.5, 1]\);
\(\mathrm{conf}(B_b)\) is the mean confidence and \(\mathrm{acc}(B_b)\) the fraction of correct voxels in bin
\(b\).

**In plain words:** the average gap between how confident the model is and how often it is right, weighted by how
many voxels fall in each confidence range. 0 means perfectly calibrated.

**Use it when:** probabilities will be read by people or used for decisions. Always report the reliability
diagram with it; `segevalkit.metrics.calibration.reliability_curve(p, y, n_bins)` returns the per-bin data.

**Watch out for:** whole-volume ECE looks excellent because most voxels are trivially correct background: in the
example below it is 0.022 over the image and 0.111 in a 5 mm band around the object
(see [Pitfalls](../guide/pitfalls.md#calibration-background)). The number of bins changes the value. ECE is not a
proper scoring rule: a model that always predicts the prevalence can be perfectly calibrated and useless; pair it
with Brier or NLL.

**Empty masks:** NaN when the region is empty.

**Parameters:** `roi = "all"`, `margin_mm = 10.0`, `n_bins = 15`.

**Reference:** Naeini MP, Cooper GF, Hauskrecht M. Obtaining well calibrated probabilities using Bayesian
binning. *AAAI 2015*. [doi:10.1609/aaai.v29i1.9602](https://doi.org/10.1609/aaai.v29i1.9602). Guo et al. 2017,
[arXiv:1706.04599](https://arxiv.org/abs/1706.04599). Mehrtash A, Wells WM, Tempany CM, Abolmaesumi P, Kapur T.
Confidence calibration and predictive uncertainty estimation for deep medical image segmentation. *IEEE TMI*
39(12), 3868–3878 (2020). [doi:10.1109/TMI.2020.3006437](https://doi.org/10.1109/TMI.2020.3006437)

</div>

## Code

The `calibration` metric set contains `ece`, `brier`, `nll`, `auroc`, `auprc` and `soft_dice`. In this example
the model is correct everywhere after thresholding at 0.5 except for an over-confident 2-voxel shell just outside
the object.

```python
import numpy as np
from scipy import ndimage
from segevalkit.metrics import compute_metrics

ref = np.zeros((48, 48, 48), bool)
ref[16:32, 16:32, 16:32] = True
dist = ndimage.distance_transform_edt(~ref)

prob = np.full(ref.shape, 0.001, np.float32)   # confident, correct background
prob[ref] = 0.9                                # foreground
prob[(dist > 0) & (dist <= 2)] = 0.6           # over-confident shell outside the object
pred = prob >= 0.5

names = ["soft_dice", "auroc", "auprc", "brier", "nll", "ece"]
whole = compute_metrics(pred, ref, names, spacing=(1, 1, 1), prob=prob)
band = compute_metrics(pred, ref, names, spacing=(1, 1, 1), prob=prob,
                       params={m: {"roi": "band", "margin_mm": 5.0} for m in names[1:]})
for m in names:
    print(f"{m:10s} all={whole[m]:.4f}  band={band[m]:.4f}")
```

```text
soft_dice  all=0.7486  band=0.7486
auroc      all=1.0000  band=1.0000
auprc      all=1.0000  band=1.0000
brier      all=0.0110  band=0.0568
nll        all=0.0319  band=0.1604
ece        all=0.0224  band=0.1112
```

The ranking is perfect (AUROC = AUPRC = 1), yet the model is badly over-confident near the boundary. The
calibration error is five times larger inside the band than over the whole image; the whole-image number is
diluted by the 89,120 easy background voxels outside the band (the band holds 21,472). Report the region with every calibration number.
