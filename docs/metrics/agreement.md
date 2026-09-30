# Agreement & information-theoretic metrics

These metrics compare prediction and reference as two **partitions** of the whole image, background included,
from the inter-rater and clustering literature. They track Dice closely but depend on true negatives, so cropping
changes them. Use them for inter-rater studies and comparability with Taha & Hanbury (2015), EvaluateSegmentation
and pymia. MCC and kappa are also in the `overlap` set.

## At a glance

| Metric | Range | Chance | Main caveat |
|---|---|---|---|
| [MCC](#mcc) | [−1, 1] | 0 | TN-dependent; \(\approx \sqrt{\mathrm{PPV}\cdot\mathrm{TPR}}\) in 3D |
| [Cohen's kappa](#cohen_kappa) | [−1, 1] | 0 | TN-dependent; \(\approx\) Dice for small foregrounds |
| [Mutual information](#mutual_information) | [0, ln 2] nats | 0 | maximum depends on prevalence |
| [Variation of information](#variation_of_information) | [0, 2 ln 2] nats | – | scale depends on prevalence |
| [Adjusted Rand index](#adjusted_rand_index) | [−1, 1] | 0 | TN-dominated pair counts |
| [Global consistency error](#global_consistency_error) | [0, 1] | – | forgives refinements |

## Notation

With \(N\) voxels, the \(2\times2\) table of reference (rows) against prediction (columns) is \(n_{11} = TP\),
\(n_{10} = FN\), \(n_{01} = FP\), \(n_{00} = TN\), with row sums \(a_i\) and column sums \(b_j\). Entropies use
the natural logarithm (nats):

\[
H(G) = -\sum_{i} \frac{a_i}{N}\log\frac{a_i}{N},\qquad
H(P) = -\sum_{j} \frac{b_j}{N}\log\frac{b_j}{N},\qquad
H(P,G) = -\sum_{i,j} \frac{n_{ij}}{N}\log\frac{n_{ij}}{N}.
\]

## Metrics

<div class="sek-metric" markdown>

### Matthews correlation coefficient (MCC) {#mcc}

<div class="sek-meta"><span class="sek-chip">mcc</span><span class="sek-chip">[−1, 1]</span><span class="sek-chip teal">↑ higher is better</span></div>

\[
\mathrm{MCC} = \frac{TP\cdot TN - FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
\]

**In words** The correlation between predicted and reference masks over all voxels: 1 perfect, 0 chance, −1 perfect disagreement.

**Use when** You need one TN-aware summary that, unlike accuracy, is not fooled by class imbalance (Chicco & Jurman 2020).

**Watch out** In 3D, MCC \(\approx \sqrt{\mathrm{PPV}\cdot\mathrm{TPR}}\) adds little beyond Dice, yet still depends on the field of view.

??? info "Details: empty masks, parameters, reference"
    **Empty masks.** Both empty: 1 (`"best"`) or NaN. Any other zero marginal (e.g. exactly one mask empty): 0.

    **Numerics.** Counts are converted to float64, so large volumes do not overflow. No parameters.

    **Reference.** Matthews BW. *Biochimica et Biophysica Acta* 405(2), 442–451 (1975). [doi:10.1016/0005-2795(75)90109-9](https://doi.org/10.1016/0005-2795(75)90109-9). Chicco D, Jurman G. *BMC Genomics* 21, 6 (2020). [doi:10.1186/s12864-019-6413-7](https://doi.org/10.1186/s12864-019-6413-7)

</div>

<div class="sek-metric" markdown>

### Cohen's kappa (κ) {#cohen_kappa}

<div class="sek-meta"><span class="sek-chip">cohen_kappa · kappa</span><span class="sek-chip">[−1, 1]</span><span class="sek-chip teal">↑ higher is better</span></div>

\[
\kappa = \frac{p_o - p_e}{1 - p_e},\qquad p_o = \frac{TP+TN}{N},\qquad
p_e = \frac{(TP+FP)(TP+FN) + (TN+FN)(TN+FP)}{N^2}
\]

**In words** How much better than chance the two segmentations agree, voxel by voxel.

**Use when** Reporting inter-rater agreement, especially for multi-rater or multi-class tissue maps where kappa is customary.

**Watch out** TN-dominated in 3D and close to Dice for small foregrounds; the kappa paradoxes apply under strong prevalence imbalance.

??? info "Details: empty masks, parameters, reference"
    **Example.** 0.8730 vs Dice 0.8750 in the code below; the value changes with the field of view. High observed agreement can coexist with low kappa.

    **Empty masks.** \(p_e = 1\) (both empty, or both whole image): 1 (`"best"`) or NaN. Exactly one mask empty: 0. No parameters.

    **Reference.** Cohen J. *Educational and Psychological Measurement* 20(1), 37–46 (1960). [doi:10.1177/001316446002000104](https://doi.org/10.1177/001316446002000104). Taha AA, Hanbury A. *BMC Medical Imaging* 15, 29 (2015). [doi:10.1186/s12880-015-0068-x](https://doi.org/10.1186/s12880-015-0068-x)

</div>

<div class="sek-metric" markdown>

### Mutual information (MI) {#mutual_information}

<div class="sek-meta"><span class="sek-chip">mutual_information</span><span class="sek-chip">[0, ln 2]</span><span class="sek-chip teal">↑ higher is better</span><span class="sek-chip orange">nats</span></div>

\[
\mathrm{MI}(P,G) = H(P) + H(G) - H(P,G)
\]

**In words** How much knowing a voxel's predicted label reduces uncertainty about its reference label.

**Use when** Comparing with the Taha & Hanbury suite; it also generalises to multi-class partitions.

**Watch out** Its maximum \(\min(H(P), H(G))\) depends on foreground fraction, so raw values do not compare across structures or images.

??? info "Details: empty masks, parameters, reference"
    **Example.** A **perfect** prediction of a 64-voxel lesion in a \(64^3\) volume scores 0.0023 nats; of a half-volume structure, \(\ln 2 = 0.693\). No clinical reading.

    **Empty masks.** Both empty: 0 (`"best"`, the zero-entropy value) or NaN (`"nan"`). As 0 is the lowest value, a correctly absent structure does not score a "best" MI under `"best"`; use `"nan"` to exclude such cases. No parameters.

    **Reference.** Russakoff DB, Tomasi C, Rohlfing T, Maurer CR. *ECCV 2004*, LNCS 3023. Taha & Hanbury 2015, [doi:10.1186/s12880-015-0068-x](https://doi.org/10.1186/s12880-015-0068-x). Cover TM, Thomas JA. *Elements of Information Theory*, 2nd ed. Wiley (2006).

</div>

<div class="sek-metric" markdown>

### Variation of information (VI) {#variation_of_information}

<div class="sek-meta"><span class="sek-chip">variation_of_information · vi</span><span class="sek-chip">[0, 2 ln 2]</span><span class="sek-chip teal">↓ lower is better</span><span class="sek-chip orange">nats</span></div>

\[
\mathrm{VI}(P,G) = H(P) + H(G) - 2\,\mathrm{MI}(P,G) = H(P\mid G) + H(G\mid P)
\]

**In words** The information lost and gained when switching from one partition to the other; 0 means identical.

**Use when** You need a true metric on partitions (triangle inequality), e.g. against clustering or connectomics results.

**Watch out** Like MI, its scale depends on foreground fraction and field of view; bounded by \(2\ln 2\) nats for binary masks.

??? info "Details: empty masks, parameters, reference"
    **Note.** For instance maps, the two conditional terms separate split from merge errors.

    **Empty masks.** Both empty: 0 (`"best"`) or NaN (`"nan"`). No parameters.

    **Reference.** Meilă M. Comparing clusterings — an information based distance. *J Multivariate Analysis* 98(5), 873–895 (2007). [doi:10.1016/j.jmva.2006.11.013](https://doi.org/10.1016/j.jmva.2006.11.013). Taha & Hanbury 2015, [doi:10.1186/s12880-015-0068-x](https://doi.org/10.1186/s12880-015-0068-x)

</div>

<div class="sek-metric" markdown>

### Adjusted Rand index (ARI) {#adjusted_rand_index}

<div class="sek-meta"><span class="sek-chip">adjusted_rand_index · ari</span><span class="sek-chip">[−1, 1]</span><span class="sek-chip teal">↑ higher is better</span></div>

\[
\mathrm{ARI} = \frac{\sum_{ij}\binom{n_{ij}}{2} - \big[\sum_i\binom{a_i}{2}\sum_j\binom{b_j}{2}\big]/\binom{N}{2}}
{\tfrac12\big[\sum_i\binom{a_i}{2}+\sum_j\binom{b_j}{2}\big] - \big[\sum_i\binom{a_i}{2}\sum_j\binom{b_j}{2}\big]/\binom{N}{2}}
\]

**In words** How often the segmentations agree on whether a voxel pair belongs together, corrected for chance.

**Use when** Comparing instance or cluster partitions, or for comparability with the Taha & Hanbury suite.

**Watch out** For binary 3D masks background–background pairs dominate, so it tracks Dice with little extra information.

??? info "Details: empty masks, parameters, reference"
    **Numerics.** Pair counts reach \(10^{15}\) for typical volumes; computed in floating point.

    **Empty masks.** Both empty (0/0): 1 (`"best"`) or NaN. Exactly one mask empty: 0. No parameters.

    **Reference.** Hubert L, Arabie P. Comparing partitions. *Journal of Classification* 2, 193–218 (1985). [doi:10.1007/BF01908075](https://doi.org/10.1007/BF01908075). Taha & Hanbury 2015, [doi:10.1186/s12880-015-0068-x](https://doi.org/10.1186/s12880-015-0068-x)

</div>

<div class="sek-metric" markdown>

### Global consistency error (GCE) {#global_consistency_error}

<div class="sek-meta"><span class="sek-chip">global_consistency_error</span><span class="sek-chip">[0, 1]</span><span class="sek-chip teal">↓ lower is better</span></div>

\[
\mathrm{GCE} = \frac1N\min\Big\{\tfrac{2\,TP\cdot FP}{TP+FP} + \tfrac{2\,TN\cdot FN}{TN+FN},\;
\tfrac{2\,TP\cdot FN}{TP+FN} + \tfrac{2\,TN\cdot FP}{TN+FP}\Big\}
\]

**In words** How far one segmentation is from being a refinement of the other; 0 when identical.

**Use when** Only for completeness and comparability with EvaluateSegmentation and pymia.

**Watch out** A refinement scores 0, so an empty or all-foreground prediction scores a perfect 0; rarely appropriate for binary medical masks.

??? info "Details: empty masks, parameters, reference"
    **Definition.** Martin et al.'s \(\frac1N\min\{\sum_x E(P,G,x),\ \sum_x E(G,P,x)\}\) with local refinement error \(E(S_1,S_2,x) = \lvert R(S_1,x)\setminus R(S_2,x)\rvert / \lvert R(S_1,x)\rvert\) (\(R(S,x)\): region of \(S\) containing \(x\)), in closed form for binary partitions; zero-denominator terms are 0. It was designed to forgive granularity differences between human segmentations of natural images.

    **Tool mismatch.** The binary closed form printed by Taha & Hanbury (2015) does not reproduce Martin's definition (it adds \(FN^2/(TP+FN)\)-type terms), so tools implementing it differ from SegEvalKit.

    **Empty masks.** Both empty: 0 (`"best"`) or NaN (`"nan"`). Empty prediction with non-empty reference: 0. No parameters.

    **Reference.** Martin D, Fowlkes C, Tal D, Malik J. *ICCV 2001*, 416–423. [doi:10.1109/ICCV.2001.937655](https://doi.org/10.1109/ICCV.2001.937655)

</div>

## Code

The `agreement` set is `cohen_kappa`, `mcc`, `adjusted_rand_index`, `variation_of_information`,
`mutual_information`, `global_consistency_error`.

```python
import numpy as np
from segevalkit.metrics import compute_metrics

ref = np.zeros((64, 64, 64), bool)
ref[10:26, 10:26, 10:26] = True                 # 4096 voxels, 1.6 % of the image
pred = np.roll(ref, 2, axis=0)
scores = compute_metrics(pred, ref, ["dice", "agreement"])
```

| Metric | Value | Metric | Value |
|---|--:|---|--:|
| `dice` | 0.8750 | `variation_of_information` | 0.0400 |
| `cohen_kappa` | 0.8730 | `mutual_information` | 0.0605 |
| `mcc` | 0.8730 | `global_consistency_error` | 0.0073 |
| `adjusted_rand_index` | 0.8695 | | |

Kappa, MCC and ARI sit within 0.006 of Dice; the information quantities are small because of the 1.6 % foreground
fraction, not segmentation quality.
