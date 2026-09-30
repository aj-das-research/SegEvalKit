# Agreement & information-theoretic metrics

These metrics treat the prediction and the reference as two **partitions** of the image into foreground and
background, and compare the partitions as a whole, background included. They come from the inter-rater agreement
and clustering literature. Two of them, the Matthews correlation coefficient and Cohen's kappa, are computed from
the same confusion counts as the [overlap metrics](overlap.md) and are part of the `overlap` metric set as well.

For segmentation quality they are strongly correlated with Dice, but every one of them depends on the true
negatives and therefore on the field of view: cropping the image changes the value. They are reported mainly for
completeness and for comparability with Taha & Hanbury (2015), EvaluateSegmentation and pymia. They are the
natural choice for inter-rater agreement studies, where "agreement beyond chance" is the question.

## Notation

With \(N\) voxels, the \(2\times2\) contingency table of reference (rows) against prediction (columns) is
\(n_{11} = TP\), \(n_{10} = FN\), \(n_{01} = FP\), \(n_{00} = TN\), with row sums \(a_i\) and column sums \(b_j\).
Entropies use the natural logarithm, so information quantities are in nats:

\[
H(G) = -\sum_{i} \frac{a_i}{N}\log\frac{a_i}{N},\qquad
H(P) = -\sum_{j} \frac{b_j}{N}\log\frac{b_j}{N},\qquad
H(P,G) = -\sum_{i,j} \frac{n_{ij}}{N}\log\frac{n_{ij}}{N}.
\]

## At a glance

| Metric | Key | Range | Chance level | Main caveat |
|---|---|---|---|---|
| [MCC](#mcc) | `mcc` | [−1, 1] | 0 | TN-dependent; \(\approx \sqrt{\mathrm{PPV}\cdot\mathrm{TPR}}\) in 3D |
| [Cohen's kappa](#cohen_kappa) | `cohen_kappa` | [−1, 1] | 0 | TN-dependent; \(\approx\) Dice for small foregrounds |
| [Mutual information](#mutual_information) | `mutual_information` | [0, ln 2] nats | 0 | maximum depends on prevalence |
| [Variation of information](#variation_of_information) | `variation_of_information` | [0, 2 ln 2] nats | – | scale depends on prevalence |
| [Adjusted Rand index](#adjusted_rand_index) | `adjusted_rand_index` | [−1, 1] | 0 | TN-dominated pair counts |
| [Global consistency error](#global_consistency_error) | `global_consistency_error` | [0, 1] | – | forgives refinements |

## Metrics

<div class="sek-metric" markdown>

### Matthews correlation coefficient (MCC) {#mcc}

<div class="sek-meta"><span class="sek-chip">key: <code>mcc</code></span><span class="sek-chip">range [−1, 1]</span><span class="sek-chip teal">higher is better</span><span class="sek-chip orange">dimensionless</span></div>

\[
\mathrm{MCC} = \frac{TP\cdot TN - FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
\]

**In plain words:** the correlation between the predicted and the reference masks over all voxels. 1 is perfect,
0 is chance, −1 is perfect disagreement.

**Use it when:** you need a single TN-aware summary of a binary prediction that, unlike accuracy, is not fooled by
class imbalance (Chicco & Jurman 2020).

**Watch out for:** in 3D, where \(TN\) dwarfs everything else, MCC \(\approx \sqrt{\mathrm{PPV}\cdot\mathrm{TPR}}\)
and adds little beyond Dice. It still depends on the field of view. Counts are converted to float64 internally, so
large volumes do not overflow.

**Empty masks:** both empty: 1 (`"best"`) or NaN. Any other zero marginal (for example exactly one mask empty):
0.

**Parameters:** none.

**Reference:** Matthews BW. Comparison of the predicted and observed secondary structure of T4 phage lysozyme.
*Biochimica et Biophysica Acta* 405(2), 442–451 (1975).
[doi:10.1016/0005-2795(75)90109-9](https://doi.org/10.1016/0005-2795(75)90109-9). Chicco D, Jurman G. The
advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification
evaluation. *BMC Genomics* 21, 6 (2020).
[doi:10.1186/s12864-019-6413-7](https://doi.org/10.1186/s12864-019-6413-7)

</div>

<div class="sek-metric" markdown>

### Cohen's kappa (κ) {#cohen_kappa}

<div class="sek-meta"><span class="sek-chip">key: <code>cohen_kappa</code></span><span class="sek-chip">alias: <code>kappa</code></span><span class="sek-chip">range [−1, 1]</span><span class="sek-chip teal">higher is better</span><span class="sek-chip orange">dimensionless</span></div>

\[
\kappa = \frac{p_o - p_e}{1 - p_e},\qquad p_o = \frac{TP+TN}{N},\qquad
p_e = \frac{(TP+FP)(TP+FN) + (TN+FN)(TN+FP)}{N^2}
\]

**In plain words:** how much better than chance the two segmentations agree, voxel by voxel.

**Use it when:** reporting inter-rater agreement, especially for multi-rater or multi-class tissue maps where kappa
is the customary statistic.

**Watch out for:** it is TN-dominated in 3D; for small foregrounds its value is close to Dice (0.8730 vs 0.8750 in
the example below) and it changes with the field of view. The kappa paradoxes apply under strong prevalence
imbalance: high observed agreement can coexist with low kappa.

**Empty masks:** when \(p_e = 1\) (both masks empty, or both covering the whole image): 1 (`"best"`) or NaN.
Exactly one mask empty: 0.

**Parameters:** none.

**Reference:** Cohen J. A coefficient of agreement for nominal scales. *Educational and Psychological Measurement*
20(1), 37–46 (1960). [doi:10.1177/001316446002000104](https://doi.org/10.1177/001316446002000104). Taha AA,
Hanbury A. *BMC Medical Imaging* 15, 29 (2015).
[doi:10.1186/s12880-015-0068-x](https://doi.org/10.1186/s12880-015-0068-x)

</div>

<div class="sek-metric" markdown>

### Mutual information (MI) {#mutual_information}

<div class="sek-meta"><span class="sek-chip">key: <code>mutual_information</code></span><span class="sek-chip">range [0, ln 2]</span><span class="sek-chip teal">higher is better</span><span class="sek-chip orange">nats</span></div>

\[
\mathrm{MI}(P,G) = H(P) + H(G) - H(P,G)
\]

**In plain words:** how much knowing the predicted label of a voxel reduces uncertainty about its reference label.

**Use it when:** comparing with the Taha & Hanbury metric suite; it also generalises to multi-class partitions.

**Watch out for:** its maximum is \(\min(H(P), H(G))\), which depends on the foreground fraction. A **perfect**
prediction of a 64-voxel lesion in a \(64^3\) volume scores 0.0023 nats, while a perfect prediction of a
structure filling half the volume scores \(\ln 2 = 0.693\). Unnormalised values are therefore not comparable
across structures or images, and have no clinical reading.

**Empty masks:** both empty: 0 (`"best"`, the value of the zero-entropy partitions) or NaN (`"nan"`). Note that 0
is the lowest value on this scale, so under `"best"` a correctly absent structure does not score a "best" MI;
use `"nan"` to exclude such cases.

**Parameters:** none.

**Reference:** Russakoff DB, Tomasi C, Rohlfing T, Maurer CR. Image similarity using mutual information of
regions. *ECCV 2004*, LNCS 3023. Taha & Hanbury 2015,
[doi:10.1186/s12880-015-0068-x](https://doi.org/10.1186/s12880-015-0068-x). Cover TM, Thomas JA. *Elements of
Information Theory*, 2nd ed. Wiley (2006).

</div>

<div class="sek-metric" markdown>

### Variation of information (VI) {#variation_of_information}

<div class="sek-meta"><span class="sek-chip">key: <code>variation_of_information</code></span><span class="sek-chip">alias: <code>vi</code></span><span class="sek-chip">range [0, 2 ln 2]</span><span class="sek-chip teal">lower is better</span><span class="sek-chip orange">nats</span></div>

\[
\mathrm{VI}(P,G) = H(P) + H(G) - 2\,\mathrm{MI}(P,G) = H(P\mid G) + H(G\mid P)
\]

**In plain words:** the information lost and gained when switching from one partition to the other. 0 means
identical partitions.

**Use it when:** you need a true metric on partitions (it satisfies the triangle inequality), for example to
compare against clustering or connectomics results. For instance maps, the two conditional terms separate split
from merge errors.

**Watch out for:** like MI, its scale depends on the foreground fraction and on the field of view. For binary masks
it is bounded by \(2\ln 2\) nats.

**Empty masks:** both empty: 0 (`"best"`) or NaN (`"nan"`).

**Parameters:** none.

**Reference:** Meilă M. Comparing clusterings — an information based distance. *Journal of Multivariate
Analysis* 98(5), 873–895 (2007). [doi:10.1016/j.jmva.2006.11.013](https://doi.org/10.1016/j.jmva.2006.11.013).
Taha & Hanbury 2015, [doi:10.1186/s12880-015-0068-x](https://doi.org/10.1186/s12880-015-0068-x).

</div>

<div class="sek-metric" markdown>

### Adjusted Rand index (ARI) {#adjusted_rand_index}

<div class="sek-meta"><span class="sek-chip">key: <code>adjusted_rand_index</code></span><span class="sek-chip">alias: <code>ari</code></span><span class="sek-chip">range [−1, 1]</span><span class="sek-chip teal">higher is better</span><span class="sek-chip orange">dimensionless</span></div>

\[
\mathrm{ARI} = \frac{\sum_{ij}\binom{n_{ij}}{2} - \big[\sum_i\binom{a_i}{2}\sum_j\binom{b_j}{2}\big]/\binom{N}{2}}
{\tfrac12\big[\sum_i\binom{a_i}{2}+\sum_j\binom{b_j}{2}\big] - \big[\sum_i\binom{a_i}{2}\sum_j\binom{b_j}{2}\big]/\binom{N}{2}}
\]

**In plain words:** of all pairs of voxels, how often the two segmentations agree on whether the pair belongs
together, corrected for the agreement expected by chance.

**Use it when:** comparing instance or cluster partitions, or for comparability with the Taha & Hanbury suite.

**Watch out for:** for binary 3D masks the pair counts are dominated by background–background pairs, so it tracks
Dice with little extra information. Pair counts reach \(10^{15}\) for typical volumes; SegEvalKit computes them in
floating point.

**Empty masks:** both empty (numerator and denominator both 0): 1 (`"best"`) or NaN. Exactly one mask empty: 0.

**Parameters:** none.

**Reference:** Hubert L, Arabie P. Comparing partitions. *Journal of Classification* 2, 193–218 (1985).
[doi:10.1007/BF01908075](https://doi.org/10.1007/BF01908075). Taha & Hanbury 2015,
[doi:10.1186/s12880-015-0068-x](https://doi.org/10.1186/s12880-015-0068-x).

</div>

<div class="sek-metric" markdown>

### Global consistency error (GCE) {#global_consistency_error}

<div class="sek-meta"><span class="sek-chip">key: <code>global_consistency_error</code></span><span class="sek-chip">range [0, 1]</span><span class="sek-chip teal">lower is better</span><span class="sek-chip orange">dimensionless</span></div>

\[
\mathrm{GCE} = \frac1N\min\Big\{\tfrac{2\,TP\cdot FP}{TP+FP} + \tfrac{2\,TN\cdot FN}{TN+FN},\;
\tfrac{2\,TP\cdot FN}{TP+FN} + \tfrac{2\,TN\cdot FP}{TN+FP}\Big\}
\]

This is Martin et al.'s definition,
\(rac1N\min\{\sum_x E(P,G,x),\ \sum_x E(G,P,x)\}\) with the local refinement error
\(E(S_1,S_2,x) = \lvert R(S_1,x)\setminus R(S_2,x)
vert / \lvert R(S_1,x)
vert\) (\(R(S,x)\) is the region of
\(S\) containing \(x\)), evaluated in closed form for two binary partitions. Terms with a zero denominator are
taken as 0.

**In plain words:** how far one segmentation is from being a refinement of the other. 0 when identical.

**Use it when:** only for completeness and comparability with EvaluateSegmentation and pymia.

**Watch out for:** GCE was designed to forgive differences in granularity between human segmentations of natural
images: if one segmentation is a refinement of the other it scores 0. For binary masks this means an empty (or
all-foreground) prediction scores 0, a perfect value. It is rarely appropriate for binary medical segmentation.
The binary closed form printed by Taha & Hanbury (2015, Eq. 11) does not reproduce Martin's definition (it adds
\(FN^2/(TP+FN)\)-type terms), so values from tools that implement that printed form differ from SegEvalKit's.

**Empty masks:** both empty: 0 (`"best"`) or NaN (`"nan"`). An empty prediction with a non-empty reference
gives 0 (see above).

**Parameters:** none.

**Reference:** Martin D, Fowlkes C, Tal D, Malik J. A database of human segmented natural images and its
application to evaluating segmentation algorithms and measuring ecological statistics. *ICCV 2001*, 416–423.
[doi:10.1109/ICCV.2001.937655](https://doi.org/10.1109/ICCV.2001.937655)

</div>

## Code

The `agreement` metric set contains `cohen_kappa`, `mcc`, `adjusted_rand_index`, `variation_of_information`,
`mutual_information` and `global_consistency_error`.

```python
import numpy as np
from segevalkit.metrics import compute_metrics

ref = np.zeros((64, 64, 64), bool)
ref[10:26, 10:26, 10:26] = True                 # 4096 voxels, 1.6 % of the image
pred = np.roll(ref, 2, axis=0)

scores = compute_metrics(pred, ref, ["dice", "agreement"])
for name, value in scores.items():
    print(f"{name:26s} {value:.4f}")
```

```text
dice                       0.8750
cohen_kappa                0.8730
mcc                        0.8730
adjusted_rand_index        0.8695
variation_of_information   0.0400
mutual_information         0.0605
global_consistency_error   0.0073
```

Kappa, MCC and ARI all sit within 0.006 of Dice. The information quantities are small numbers whose scale is set
by the 1.6 % foreground fraction, not by segmentation quality.
