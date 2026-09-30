# Evaluation Metrics for 3D Medical Image Segmentation: A Literature Review for SegEvalKit

**Scope.** This review catalogs metrics for evaluating 3D (CT/MR, NIfTI) segmentation. It covers semantic (per-class binary or multi-label) and instance (multi-lesion) settings, plus probabilistic outputs. For each metric it gives the definition, conventions (empty masks, voxel spacing, symmetric vs. directed), interpretation, use cases, pitfalls and primary references. It follows the vocabulary and recommendations of *Metrics Reloaded* (Maier-Hein et al., 2024) and its companion pitfalls paper (Reinke et al., 2024).

**Verification note.** Equations were checked against the primary sources where accessible: Taha & Hanbury 2015 (full text), Metrics Reloaded (full text), Cheng et al. 2021, Kirillov et al. 2019, Shit et al. 2021 and Nikolov et al. 2021. Implementation conventions were checked against the current source code of MetricsReloaded (`pairwise_measures.py`), MONAI (`monai/metrics`), DeepMind `surface-distance`, MedPy (`medpy/metric/binary.py`), pymia and the BraTS 2023 lesion-wise evaluation code (`rachitsaluja/BraTS-2023-Metrics`). Where a statement is a convention or an inference rather than a sourced fact, the text says so.

---

## Table of contents

0. [Common notation](#0-common-notation)
1. [Overlap (confusion-matrix) metrics](#1-overlap-confusion-matrix-metrics)
2. [Volume metrics](#2-volume-metrics)
3. [Distance and boundary metrics](#3-distance-and-boundary-metrics)
4. [Topology metrics](#4-topology-metrics)
5. [Detection and instance metrics](#5-detection-and-instance-metrics)
6. [Calibration and probabilistic metrics](#6-calibration-and-probabilistic-metrics)
7. [Information-theoretic, clustering and agreement metrics](#7-information-theoretic-clustering-and-agreement-metrics)
8. [Statistical aggregation, uncertainty and ranking](#8-statistical-aggregation-uncertainty-and-ranking)
9. [Empty-mask and edge-case conventions (cross-library table)](#9-empty-mask-and-edge-case-conventions)
10. [Metric selection guide (problem fingerprinting)](#10-metric-selection-guide)
11. [Survey of existing tools and gaps for SegEvalKit](#11-survey-of-existing-tools-and-gaps-for-segevalkit)
12. [References](#12-references)

---

## 0. Common notation

Let the image domain $\Omega \subset \mathbb{Z}^3$ contain $N = |\Omega|$ voxels with physical spacing $\mathbf{s} = (s_x, s_y, s_z)$ in mm and voxel volume $v = s_x s_y s_z$ (mm³).

| Symbol | Meaning |
|---|---|
| $G \subseteq \Omega$ | Reference ("ground truth") foreground voxels for one class |
| $P \subseteq \Omega$ | Predicted foreground voxels for the same class |
| $TP = \lvert G \cap P \rvert$ | True-positive voxel count |
| $FP = \lvert P \setminus G \rvert$ | False-positive voxel count |
| $FN = \lvert G \setminus P \rvert$ | False-negative voxel count |
| $TN = \lvert \Omega \setminus (G \cup P) \rvert$ | True-negative voxel count; $N = TP+FP+FN+TN$ |
| $\partial G, \partial P$ | Boundary (surface) of $G$, $P$, as border voxels or as a mesh of surface elements ("surfels") |
| $d(x, Y) = \min_{y \in Y} \lVert x - y \rVert_2$ | Distance from point $x$ to set $Y$, in mm (world coordinates, spacing applied) |
| $p_i \in [0,1]$ | Predicted foreground probability at voxel $i$; $y_i \in \{0,1\}$ reference label |
| $V_G = v\lvert G \rvert,\; V_P = v\lvert P\rvert$ | Physical volumes (mm³; divide by 1000 for mL) |

**Spacing awareness.** Metrics built only from counts ($TP, FP, FN, TN$) are unit-free and *spacing-invariant* for a fixed grid, but they change when resampled. Distance metrics must use the voxel spacing, and for anisotropic MR (for example 0.5×0.5×5 mm) ignoring it gives errors of several-fold. Volumes must be multiplied by $v$. The NIfTI affine (qform/sform) should be the source of spacing, and SegEvalKit should validate that $G$ and $P$ share the same grid and affine.

**Boundary extraction.** There are two families of implementation:

- **Voxel-border sets.** The border is the set of foreground voxels that have a background neighbour; each border voxel counts once. MedPy, MetricsReloaded and MONAI (default) work this way.
- **Surfel meshes with area weights.** DeepMind `surface-distance` builds these, and MONAI does too when `use_subvoxels=True`.

The two families give numerically different ASSD, HD95 and NSD, particularly on anisotropic grids. SegEvalKit should document which one it uses and ideally offer both.

---

## 1. Overlap (confusion-matrix) metrics

### 1.1 Dice similarity coefficient (DSC)

- **Aliases:** Dice, Sørensen–Dice, F1 score at voxel level, "Dice overlap".
- **Equation:**

    $$
    \mathrm{DSC}(G,P) = \frac{2\,\lvert G\cap P\rvert}{\lvert G\rvert + \lvert P\rvert} = \frac{2\,TP}{2\,TP + FP + FN}
    $$

- **Range / direction / units:** $[0,1]$, higher is better, unitless.
- **Edge cases:** If $G = P = \emptyset$ the expression is $0/0$. Conventions differ:
    - MetricsReloaded returns NaN and recommends setting it to 1 during aggregation.
    - MONAI `DiceMetric(ignore_empty=True)` returns NaN when the reference is empty, and it is excluded from the mean.
    - BraTS 2023 scores 1.0 when both are empty.

    If exactly one mask is empty, DSC = 0.

- **Plain English:** The fraction of voxels the two segmentations share, measured relative to their average size. 1 means identical and 0 means no overlap.
- **Good for:** A default measure of overall volumetric agreement for medium or large structures (organs). Metrics Reloaded recommends DSC (or IoU) as the default overlap metric for semantic segmentation.
- **Pitfalls (Reinke et al., 2024):**
    - It is size-dependent. A one-voxel boundary error costs a small lesion far more DSC than a liver, so mean DSC across structures of different sizes is not comparable.
    - It is insensitive to boundary shape and to where errors lie. Two very different error patterns can give the same DSC.
    - It is undefined for empty references.
    - It saturates at 0 for any non-overlapping prediction, however near or far the prediction is.
    - It ignores topology (holes and disconnections).
    - Mean DSC hides catastrophic failures, so report distributions.
- **References:** Dice (1945); Sørensen (1948); Maier-Hein et al. (2024); Taha & Hanbury (2015).

### 1.2 Jaccard index / Intersection over Union (IoU)

- **Aliases:** Jaccard similarity coefficient (JAC), Tanimoto coefficient (for binary sets), IoU. **Volumetric overlap error** $\mathrm{VOE} = 1 - \mathrm{IoU}$ (Heimann et al., 2009).
- **Equation:**

    $$
    \mathrm{IoU}(G,P)=\frac{\lvert G\cap P\rvert}{\lvert G\cup P\rvert}=\frac{TP}{TP+FP+FN},\qquad
    \mathrm{IoU}=\frac{\mathrm{DSC}}{2-\mathrm{DSC}},\quad \mathrm{DSC}=\frac{2\,\mathrm{IoU}}{1+\mathrm{IoU}}
    $$

- **Range / direction / units:** $[0,1]$, higher is better (VOE: lower is better, often in %). Unitless.
- **Edge cases:** The same as DSC. MetricsReloaded returns NaN when both masks are empty.
- **Plain English:** Shared volume divided by the combined volume.
- **Good for:** The same questions as DSC. IoU is the standard for instance matching (for example IoU > 0.5 in Panoptic Quality) and in computer-vision benchmarks.
- **Pitfalls:** IoU is a monotone transform of DSC, so reporting both adds no information (Taha & Hanbury, 2015). IoU is stricter numerically: DSC 0.8 corresponds to IoU 0.67. Otherwise it has the same pitfalls as DSC.
- **References:** Jaccard (1912); Heimann et al. (2009); Taha & Hanbury (2015).

### 1.3 Generalized Dice (multi-class / multi-label)

- **Aliases:** GDS, Generalized Dice Score (not to be confused with the generalized Dice *loss*).
- **Equation** (Crum et al., 2006; weighted form popularized by Sudre et al., 2017). With classes $c$ and weights $w_c$, for example $w_c = 1/(\sum_i g_{ic})^2$:

    $$
    \mathrm{GDS} = \frac{2\sum_{c} w_c \sum_i g_{ic}\,p_{ic}}{\sum_c w_c \sum_i (g_{ic} + p_{ic})}
    $$

    Here $g_{ic},p_{ic}\in\{0,1\}$ (or $[0,1]$ for soft versions) indicate membership of voxel $i$ in class $c$.

- **Range / direction:** $[0,1]$, higher is better.
- **Edge cases:** $w_c$ is infinite when class $c$ is absent from the reference, so implementations clip or zero the weight.
- **Plain English:** A single Dice-like score for all labels at once, with rare (small) labels weighted up so large structures do not dominate.
- **Good for:** A single summary across many labels, such as whole-body multi-organ segmentation.
- **Pitfalls:** The weighting choice changes the result. A pooled score hides per-class failures. Metrics Reloaded recommends per-class reporting instead of merging classes.
- **References:** Crum, Camara & Hill (2006); Sudre et al. (2017).

### 1.4 Tversky index and Fβ score

- **Aliases:** Tversky similarity; Fβ-score (Metrics Reloaded "FβScore").
- **Equations:**

    $$
    \mathrm{TI}_{\alpha,\beta}(G,P) = \frac{TP}{TP + \alpha\,FP + \beta\,FN},\qquad
    F_\beta = \frac{(1+\beta^2)\,TP}{(1+\beta^2)\,TP + \beta^2 FN + FP}
    $$

    $\alpha=\beta=0.5$ gives DSC and $\alpha=\beta=1$ gives IoU. $F_\beta$ equals $\mathrm{TI}$ with $\alpha = 1/(1+\beta^2)$, $\beta_{\mathrm{TI}} = \beta^2/(1+\beta^2)$. $\beta>1$ weights recall more.

- **Range / direction:** $[0,1]$, higher is better.
- **Plain English:** A Dice-like score where you choose whether missing tissue (FN) or extra tissue (FP) is worse.
- **Good for:** Asymmetric clinical costs. Examples: tumour-margin coverage in radiotherapy, where missing tumour is worse, or avoiding false alarms. Metrics Reloaded selects $F_\beta$ over DSC when false positives and false negatives should be penalized unequally.
- **Pitfalls:** The choice of $\beta$ (or $\alpha$) must be justified up front. It keeps DSC's size dependence.
- **References:** Tversky (1977); Salehi et al. (2017) (Tversky loss); Maier-Hein et al. (2024).

### 1.5 Sensitivity (recall, TPR) and false-negative rate (FNR)

- **Equations:**

    $$
    \mathrm{TPR} = \frac{TP}{TP+FN}=\frac{\lvert G\cap P\rvert}{\lvert G\rvert},\qquad \mathrm{FNR} = 1-\mathrm{TPR}=\frac{FN}{TP+FN}
    $$

- **Range / direction:** $[0,1]$. TPR: higher is better. FNR: lower is better.
- **Edge cases:** Undefined when $G=\emptyset$ (MetricsReloaded warns and returns NaN).
- **Plain English:** The fraction of the true structure that the algorithm captured.
- **Good for:** Coverage questions (did we capture the whole tumour?). Taha & Hanbury (2015) recommend recall-type metrics when missed regions harm more than added ones.
- **Pitfalls:** It is trivially maximized by over-segmentation, so always pair it with precision or DSC.
- **References:** Taha & Hanbury (2015); Maier-Hein et al. (2024).

### 1.6 Precision (PPV) and false discovery rate (FDR)

- **Equations:**

    $$
    \mathrm{PPV} = \frac{TP}{TP+FP}=\frac{\lvert G\cap P\rvert}{\lvert P\rvert},\qquad \mathrm{FDR}=1-\mathrm{PPV}
    $$

- **Range / direction:** $[0,1]$. PPV: higher is better.
- **Edge cases:** Undefined when $P=\emptyset$.
- **Plain English:** The fraction of what the algorithm marked that is actually correct.
- **Good for:** Detecting over-segmentation and leakage into neighbouring organs.
- **Pitfalls:** Trivially maximized by under-segmentation. DSC is the harmonic mean of PPV and TPR.

### 1.7 Specificity (TNR), false-positive rate (FPR, "fallout") and accuracy

- **Equations:**

    $$
    \mathrm{TNR}=\frac{TN}{TN+FP},\quad \mathrm{FPR}=1-\mathrm{TNR}=\frac{FP}{FP+TN},\quad \mathrm{Acc}=\frac{TP+TN}{N},\quad \mathrm{BA}=\tfrac12(\mathrm{TPR}+\mathrm{TNR})
    $$

- **Range / direction:** $[0,1]$. TNR, Acc and balanced accuracy (BA): higher is better. FPR: lower is better.
- **Plain English:** Specificity is how much of the background was correctly left unlabelled.
- **Good for:** Rarely informative for 3D segmentation. They are useful mainly as sanity checks or in image-level classification.
- **Pitfalls:** Class imbalance is the main problem. In a CT volume the background ($TN$) is typically 10³–10⁶ times the foreground, so TNR and accuracy are ≈1 for almost any prediction. They also depend on field of view (cropping changes $TN$). Reinke et al. (2024) list this as a prevalence-dependence and TN-dominance pitfall. Taha & Hanbury (2015) likewise caution against TN-including metrics for small segments.
- **References:** Taha & Hanbury (2015); Reinke et al. (2024).

### 1.8 Matthews correlation coefficient (MCC)

- **Aliases:** Phi coefficient.
- **Equation:**

    $$
    \mathrm{MCC} = \frac{TP\cdot TN - FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
    $$

- **Range / direction:** $[-1,1]$, higher is better. 0 means chance level.
- **Edge cases:** Undefined when any marginal is 0, for example an empty reference or empty prediction.
- **Plain English:** A correlation between the predicted and true masks that accounts for all four confusion-matrix cells.
- **Good for:** Balanced single-number summaries in imbalanced binary problems (Chicco & Jurman, 2020). It is more informative than accuracy.
- **Pitfalls:** It still depends on $TN$, hence on field of view. In 3D segmentation with huge $TN$, MCC ≈ $\sqrt{\mathrm{PPV}\cdot\mathrm{TPR}}$ and adds little beyond DSC. Implementations can overflow with int32 counts, so use float64.
- **References:** Matthews (1975); Chicco & Jurman (2020).

### 1.9 Cohen's kappa (KAP)

- **Equation** (as in Taha & Hanbury, 2015, Eqs. 45–46):

    $$
    \kappa=\frac{f_a - f_c}{N - f_c},\quad f_a = TP+TN,\quad f_c = \frac{(TN+FN)(TN+FP) + (FP+TP)(FN+TP)}{N}
    $$

    Equivalently $\kappa=(p_o-p_e)/(1-p_e)$, with observed agreement $p_o=f_a/N$ and chance agreement $p_e=f_c/N$.

- **Range / direction:** $[-1,1]$, higher is better.
- **Plain English:** How much better than chance the two segmentations agree.
- **Good for:** Inter-rater agreement studies, especially multi-class tissue maps.
- **Pitfalls:** It is TN-dominated in 3D, so values are close to DSC for small foregrounds and depend on field of view. It has the known kappa paradoxes under strong prevalence imbalance.
- **References:** Cohen (1960); Taha & Hanbury (2015).

### 1.10 Single-threshold AUC (balanced accuracy)

- **Equation** (Taha & Hanbury, 2015, Eq. 47), for a binary (thresholded) prediction:

    $$
    \mathrm{AUC}_{\text{binary}} = 1-\frac{\mathrm{FPR}+\mathrm{FNR}}{2} = 1-\frac12\left(\frac{FP}{FP+TN}+\frac{FN}{FN+TP}\right)
    $$

    This equals balanced accuracy, the area under the one-point ROC "curve". For *probabilistic* output the proper voxel-wise ROC-AUC is $\Pr(p_i > p_j \mid y_i=1, y_j=0)$, computed with the Mann–Whitney statistic.

- **Range / direction:** $[0,1]$, higher is better. 0.5 is chance.
- **Plain English:** The average of sensitivity and specificity.
- **Pitfalls:** It is TN-dominated. Voxel-wise ROC-AUC is ≈1 for nearly any reasonable model in 3D and mostly measures separability of the easy background. Precision–recall summaries are more informative under imbalance (Reinke et al., 2024).
- **References:** Taha & Hanbury (2015).

---

## 2. Volume metrics

### 2.1 Volumetric similarity (VS)

- **Equation** (Taha & Hanbury, 2015, Eq. 21; Cárdenes et al., 2009):

    $$
    \mathrm{VS} = 1-\frac{\lvert FN-FP\rvert}{2TP+FP+FN} = 1-\frac{\big\lvert\,\lvert G\rvert-\lvert P\rvert\,\big\rvert}{\lvert G\rvert+\lvert P\rvert}
    $$

- **Range / direction:** $[0,1]$, higher is better.
- **Edge cases:** $0/0$ when both masks are empty (conventionally 1).
- **Plain English:** How similar the two volumes are in size, ignoring where they are.
- **Good for:** Volumetry, where "the segmented volume is in the focus of interest regardless of the boundary and the alignment" (Taha & Hanbury, 2015). Examples are hippocampal atrophy and tumour burden.
- **Pitfalls:** VS = 1 is possible with zero overlap, since a shifted mask of equal size scores perfectly. Never use it alone as a segmentation-quality measure.
- **References:** Cárdenes et al. (2009); Taha & Hanbury (2015).

### 2.2 Relative volume difference (RVD) and absolute volume difference (AVD)

- **Aliases:**
    - Relative absolute volume difference (RAVD, MedPy `ravd`; Heimann et al., 2009, report the signed RVD in %).
    - Absolute volume difference in mL (MONAI `AbsoluteVolumeDifference`).
    - Volume error.
- **Equations:**

    $$
    \mathrm{RVD} = \frac{\lvert P\rvert-\lvert G\rvert}{\lvert G\rvert}\ (\times 100\%),\qquad
    \mathrm{AVD}_{\mathrm{mL}} = \frac{v\,\big\lvert\,\lvert P\rvert-\lvert G\rvert\,\big\rvert}{1000},\qquad
    \mathrm{AVD}_{\%}=\frac{\big\lvert\,\lvert P\rvert-\lvert G\rvert\,\big\rvert}{\lvert G\rvert}\times100
    $$

- **Range / direction:** RVD is in $[-1,\infty)$ (as a fraction). 0 is best, positive means over-segmentation and negative means under-segmentation. AVD is $\ge 0$, lower is better, in mL (or %).
- **Edge cases:** RVD is undefined when $G=\emptyset$. For AVD in mL, report the absolute predicted volume as the error in that case.
- **Spacing:** AVD in mL requires the voxel volume $v$ from the NIfTI header.
- **Plain English:** How much larger or smaller the predicted structure is than the true one.
- **Good for:** Volumetry endpoints (tumour-volume response, organ volume, lesion load in MS), and detecting systematic bias. The signed RVD reveals bias. The absolute value does not.
- **Pitfalls:** Positional errors are invisible, as with VS. Relative errors explode for tiny references, so report absolute mL for small lesions. Averaging signed RVD lets over- and under-segmentation cancel.
- **References:** Heimann et al. (2009); MedPy; MONAI.

### 2.3 Intraclass correlation coefficient (ICC) for volumes

- **Aliases:** ICC(2,1) (two-way random, absolute agreement, single rater) and ICC(3,1) (two-way mixed, consistency). Taha & Hanbury (2015) also define a voxel-wise ICC between two masks (pymia `InterclassCorrelation`).
- **Equations** (Shrout & Fleiss, 1979). For $n$ subjects and $k$ raters or methods (here $k=2$: algorithm vs reference), with between-subject mean square $\mathrm{BMS}$, between-rater mean square $\mathrm{JMS}$ and residual mean square $\mathrm{EMS}$:

    $$
    \mathrm{ICC}(2,1)=\frac{\mathrm{BMS}-\mathrm{EMS}}{\mathrm{BMS}+(k-1)\mathrm{EMS}+\frac{k}{n}(\mathrm{JMS}-\mathrm{EMS})},\qquad
    \mathrm{ICC}(3,1)=\frac{\mathrm{BMS}-\mathrm{EMS}}{\mathrm{BMS}+(k-1)\mathrm{EMS}}
    $$

- **Range / direction:** $(-1, 1]$ in practice, higher is better. Koo & Li (2016) suggest <0.5 poor, 0.5–0.75 moderate, 0.75–0.9 good and >0.9 excellent.
- **Plain English:** How well the algorithm's volumes agree with the reference volumes across patients, relative to how much volumes vary between patients.
- **Good for:** Population-level volumetry validation and method-comparison studies. It is computed over a *dataset*, not per case.
- **Pitfalls:** A high ICC can coexist with large absolute errors when the between-subject spread is large. Choose the ICC form deliberately and report it (Koo & Li, 2016). Pearson's $r$ is *not* an agreement measure (Bland & Altman, 1986).
- **References:** Shrout & Fleiss (1979); Koo & Li (2016); Taha & Hanbury (2015).

### 2.4 Bland–Altman analysis (bias and limits of agreement)

- **Equations.** For subject $j$, $\delta_j = V_{P,j}-V_{G,j}$ (or percentage difference relative to the mean volume $\bar V_j=(V_{P,j}+V_{G,j})/2$):

    $$
    \bar\delta=\frac1n\sum_j\delta_j,\qquad s_\delta=\sqrt{\tfrac{1}{n-1}\sum_j(\delta_j-\bar\delta)^2},\qquad \mathrm{LoA}=\bar\delta\pm1.96\,s_\delta
    $$

    Plot $\delta_j$ against $\bar V_j$.

- **Units:** mL (or %). A bias near 0 and narrow LoA are better.
- **Plain English:** On average, how much does the algorithm over- or under-estimate volume, and within what range do 95% of individual errors fall?
- **Good for:** Clinical acceptability of volumetry. Compare the LoA against a clinically acceptable difference or against inter-rater LoA.
- **Pitfalls:** It assumes the differences are roughly normal and do not depend on magnitude. Use the log or percentage version for proportional error. Give confidence intervals for the LoA with small $n$.
- **References:** Bland & Altman (1986).

---

## 3. Distance and boundary metrics

All metrics in this section must be computed in **mm using voxel spacing**. They are **undefined when either mask is empty.** Conventions differ:

- MetricsReloaded returns NaN and advises using the worst case during aggregation.
- DeepMind `surface-distance` returns `inf` for distances (surface Dice is NaN when both masks are empty and 0 when one is). Under NumPy 2 these branches raise, because the code uses `np.Inf`.
- MONAI returns NaN or `inf` with a warning.
- MedPy raises an exception.
- BraTS 2023 substitutes 374 mm, the corner-to-corner diagonal of the 240×240×155 mm SRI-24 atlas space ($\sqrt{240^2+240^2+155^2}\approx 373.13$, rounded up to 374 in the official code).

A common, reproducible alternative is to penalize with the image diagonal in mm of the evaluated volume.

### 3.1 Hausdorff distance (HD)

- **Aliases:** Maximum symmetric surface distance (MSSD), HD100.
- **Equations.** Directed distance, then the symmetric maximum:

    $$
    \vec h(A,B)=\max_{a\in A} d(a,B),\qquad \mathrm{HD}(G,P)=\max\{\vec h(\partial G,\partial P),\ \vec h(\partial P,\partial G)\}
    $$

- **Range / direction / units:** $[0,\infty)$ mm, lower is better.
- **Symmetric vs directed:** The symmetric form is standard. The directed form ($\vec h(\partial P,\partial G)$, "how far the prediction strays") is available in MONAI with `directed=True`.
- **Plain English:** The single worst boundary error, meaning the largest distance from any point on one contour to the nearest point on the other.
- **Good for:** Worst-case safety analysis, such as radiotherapy organs at risk, where a single far-off error matters.
- **Pitfalls:**
    - It is extremely sensitive to outliers. One stray false-positive voxel far away sets HD. Reinke et al. (2024) recommend percentile HD instead.
    - It is undefined for empty masks.
    - It depends on resolution and spacing.
    - It is not comparable across structures of different sizes.
- **References:** Huttenlocher, Klanderman & Rucklidge (1993).

### 3.2 Percentile Hausdorff distance (HD95, HDk)

- **Aliases:** Robust Hausdorff, 95% HD, HD95.
- **Equation.** Let $P_k$ denote the $k$-th percentile.
    - **Directed-max convention** (MetricsReloaded, MONAI, DeepMind `compute_robust_hausdorff`, BraTS):

        $$
        \mathrm{HD}_k(G,P)=\max\Big\{P_k\big(\{d(g,\partial P)\}_{g\in\partial G}\big),\ P_k\big(\{d(p,\partial G)\}_{p\in\partial P}\big)\Big\}
        $$

    - **Pooled convention** (MedPy `hd95`):

        $$
        \mathrm{HD}_k^{\text{pooled}}=P_k\big(\{d(g,\partial P)\}_{g\in\partial G}\ \cup\ \{d(p,\partial G)\}_{p\in\partial P}\big)
        $$

    DeepMind weights each distance by its surfel area. MONAI and MetricsReloaded weight each border voxel equally.

- **Range / direction / units:** $[0,\infty)$ mm, lower is better.
- **Plain English:** A "nearly worst-case" boundary error. 95% of boundary points are within this distance of the other contour.
- **Good for:** Robust boundary quality. It is widely used in BraTS, KiTS and radiotherapy auto-contouring. Metrics Reloaded lists HD95 among the boundary-based options for when errors should be penalized according to their distance.
- **Pitfalls:**
    - The two conventions give different numbers. SegEvalKit should expose `mode={"max_directed","pooled"}` and `weighting={"voxel","surfel"}`.
    - The percentile $k$ must be reported.
    - It is still undefined for empty masks.
    - For small structures with few boundary points, HD95 approaches HD.
- **References:** Huttenlocher et al. (1993); Maier-Hein et al. (2024); Nikolov et al. (2021) (DeepMind implementation).

### 3.3 Average symmetric surface distance (ASSD)

- **Aliases:** ASD (ambiguous), average surface distance, mean symmetric surface distance.
- **Equation** (Heimann et al., 2009; MetricsReloaded `measured_average_distance`). The average is pooled over *all* boundary points of both surfaces:

    $$
    \mathrm{ASSD}(G,P)=\frac{\sum_{g\in\partial G} d(g,\partial P)+\sum_{p\in\partial P} d(p,\partial G)}{\lvert\partial G\rvert+\lvert\partial P\rvert}
    $$

- **Range / direction / units:** $[0,\infty)$ mm, lower is better.
- **Plain English:** On average, how far the two contours are apart, in millimetres.
- **Good for:** Typical boundary accuracy. Classic liver-segmentation benchmarks (SLIVER07) use it.
- **Pitfalls:**
    - Averaging hides localized large errors.
    - The larger surface dominates (compare MASD).
    - Naming is inconsistent across papers and tools. MedPy `asd` is the *directed* mean. MedPy `assd` changed with the version: up to 0.4.0 it is the mean of the two directed means (MASD-style); in 0.5.0 and 0.5.1 it calls `numpy.mean` on the ragged pair of distance arrays, which raises a `ValueError` under current NumPy whenever the two surfaces differ in size; in 0.5.2 it concatenates both distance arrays and takes one mean, which is the pooled ASSD above. Always state the formula. See Yeghiazaryan & Voiculescu (2018) for a taxonomy.
- **References:** Heimann et al. (2009); Yeghiazaryan & Voiculescu (2018); Maier-Hein et al. (2024).

### 3.4 Mean average surface distance (MASD)

- **Equation** (Beneš & Zitová, 2015; MetricsReloaded `measured_masd`). This is the mean of the two directed means:

    $$
    \mathrm{MASD}(G,P)=\frac12\left(\frac{\sum_{g\in\partial G} d(g,\partial P)}{\lvert\partial G\rvert}+\frac{\sum_{p\in\partial P} d(p,\partial G)}{\lvert\partial P\rvert}\right)
    $$

- **Range / direction / units:** $[0,\infty)$ mm, lower is better.
- **Plain English:** Like ASSD, but each contour gets equal weight regardless of its size.
- **Good for:** Symmetric typical boundary error. Metrics Reloaded offers it alongside ASSD, HD95 and Boundary IoU.
- **Pitfalls:** It is easily confused with ASSD. The two are equal only when $\lvert\partial G\rvert=\lvert\partial P\rvert$.
- **References:** Beneš & Zitová (2015); Maier-Hein et al. (2024).

### 3.5 Directed average surface distance and average Hausdorff distance

- **Equations:**

    $$
    \vec{d}_{\text{avg}}(A,B)=\frac{1}{\lvert A\rvert}\sum_{a\in A} d(a,B),\qquad
    \mathrm{AVD}_{\text{Taha}}(G,P)=\max\{\vec d_{\text{avg}}(G,P),\ \vec d_{\text{avg}}(P,G)\}
    $$

    The second is the "average Hausdorff distance", Taha & Hanbury (2015) Eqs. 50–51. It is computed over voxel *sets*, not only surfaces, in EvaluateSegmentation.

- **Range / direction / units:** mm, lower is better.
- **Plain English:** The average distance from one segmentation to the other, in one direction or taking the worse direction.
- **Good for:** Diagnosing whether error comes from over-segmentation ($P\to G$ large) or under-segmentation ($G\to P$ large). DeepMind `compute_average_surface_distance` returns both directed values as a tuple.
- **Pitfalls:** The directed forms are asymmetric by design. The Taha AVD differs from ASSD and MASD.
- **References:** Taha & Hanbury (2015); Nikolov et al. (2021).

### 3.6 Normalized surface Dice (NSD)

- **Aliases:** Surface Dice at tolerance τ, surface DSC, normalized surface distance (Metrics Reloaded).
- **Equation** (Nikolov et al., 2021). Let $\mathcal B_A^{(\tau)}=\{x: d(x,\partial A)\le\tau\}$ be the border region of tolerance $\tau$ (mm) around surface $\partial A$, with $\lvert\cdot\rvert$ denoting surface *area*:

    $$
    \mathrm{NSD}^{(\tau)}(G,P)=\frac{\lvert\partial P\cap\mathcal B_G^{(\tau)}\rvert+\lvert\partial G\cap\mathcal B_P^{(\tau)}\rvert}{\lvert\partial P\rvert+\lvert\partial G\rvert}
    $$

- **Range / direction:** $[0,1]$, higher is better.
- **Parameters:** $\tau$ in mm. Nikolov et al. set organ-specific $\tau$ from inter-observer variability. The Medical Segmentation Decathlon also reports NSD (Antonelli et al., 2022).
- **Edge cases:** If both masks are empty, MetricsReloaded returns NaN and advises the best value at aggregation. If exactly one is empty the result is 0 (DeepMind), or NaN in some tools.
- **Plain English:** The fraction of the contour that is "close enough", within a clinically acceptable tolerance, so it would not need manual correction.
- **Good for:** Boundary-critical tasks, especially radiotherapy contouring (estimating correction effort). Metrics Reloaded makes NSD the default boundary metric when annotation imprecision should be compensated, and recommends pairing it with DSC.
- **Pitfalls:**
    - It depends on $\tau$, so always report $\tau$ and justify it (inter-rater variability or a clinical margin).
    - It gives no information about error magnitude beyond $\tau$.
    - Surfel-area weighting (DeepMind) and border-voxel counting (MetricsReloaded, MONAI default) disagree, particularly on anisotropic grids.
- **References:** Nikolov et al. (2021); Maier-Hein et al. (2024); Antonelli et al. (2022).

### 3.7 Surface overlap at tolerance (directed)

- **Equation** (DeepMind `compute_surface_overlap_at_tolerance`; pymia `SurfaceOverlap`):

    $$
    \mathrm{SO}_{G\to P}^{(\tau)}=\frac{\lvert\partial G\cap\mathcal B_P^{(\tau)}\rvert}{\lvert\partial G\rvert},\qquad
    \mathrm{SO}_{P\to G}^{(\tau)}=\frac{\lvert\partial P\cap\mathcal B_G^{(\tau)}\rvert}{\lvert\partial P\rvert}
    $$

- **Range / direction:** $[0,1]$ each, higher is better. They are a boundary "recall" and "precision".
- **Plain English:** How much of the true contour the prediction reproduces within tolerance, and how much of the predicted contour is correct within tolerance.
- **Good for:** Separating under-contouring from over-contouring in radiotherapy QA. NSD combines the two.
- **References:** Nikolov et al. (2021).

### 3.8 Boundary IoU (BIoU)

- **Equation** (Cheng et al., 2021). Let $G_d=\{x\in G: d(x,\partial G)\le d\}$ be the inner boundary band of width $d$, and likewise $P_d$:

    $$
    \mathrm{BIoU}(G,P)=\frac{\lvert G_d\cap P_d\rvert}{\lvert G_d\cup P_d\rvert}
    $$

- **Range / direction:** $[0,1]$, higher is better.
- **Parameters:** Band width $d$. The original 2D paper uses 2% of the image diagonal. For 3D medical images, specify $d$ in mm. MetricsReloaded defaults to 1 voxel.
- **Edge cases:** If both masks are empty, NaN (MetricsReloaded, set to best at aggregation).
- **Plain English:** Overlap measured only near the boundaries, so it rewards precise contours and not just bulk overlap.
- **Good for:** Boundary quality for large objects, where DSC saturates. It is less outlier-sensitive than HD. It is listed in Metrics Reloaded's boundary-metric pool and as an instance-matching localization criterion.
- **Pitfalls:** It depends on $d$. For objects thinner than $2d$ it reduces to IoU. Defining $d$ in voxels on anisotropic grids is ambiguous.
- **References:** Cheng et al. (2021); Maier-Hein et al. (2024).

### 3.9 Centre-of-mass (centroid) distance

- **Equation:** $\;\mathrm{CoMD}=\big\lVert \mathbf c_G-\mathbf c_P\big\rVert_2$ with $\mathbf c_A=\frac1{\lvert A\rvert}\sum_{x\in A} \mathbf x$ (world coordinates, mm).
- **Range / direction / units:** $[0,\infty)$ mm, lower is better. Undefined for empty masks (MetricsReloaded returns −1).
- **Plain English:** How far apart the centres of the true and predicted structures are.
- **Good for:** Localization and detection matching (for example a centroid-within-radius criterion for small lesions), and a registration-style sanity check.
- **Pitfalls:** It is blind to shape and size. Concave or multi-component objects have centroids outside the object.
- **References:** Maier-Hein et al. (2024) (localization criteria); MetricsReloaded implementation.

### 3.10 Mahalanobis distance (MHD)

- **Equation** (Taha & Hanbury, 2015, Eqs. 53–54). Voxel coordinates of $G$ and $P$ are treated as point clouds with means $\boldsymbol\mu_G,\boldsymbol\mu_P$, covariances $S_G,S_P$ and sizes $n_G=\lvert G\rvert$, $n_P=\lvert P\rvert$:

    $$
    \mathrm{MHD}(G,P)=\sqrt{(\boldsymbol\mu_G-\boldsymbol\mu_P)^\top S^{-1}(\boldsymbol\mu_G-\boldsymbol\mu_P)},\qquad S=\frac{n_G S_G+n_P S_P}{n_G+n_P}
    $$

    Taha & Hanbury print the squared form without the square root. State which one is used.

- **Range / direction:** $[0,\infty)$, unitless (normalized by covariance), lower is better.
- **Plain English:** How far apart the two shapes' centres are, measured relative to the shapes' spread and orientation.
- **Good for:** "General shape and alignment", ignoring boundary detail (Taha & Hanbury, 2015).
- **Pitfalls:** It is insensitive to boundary errors and details. It is degenerate for flat or single-slice objects (singular $S$). It is rarely used in modern benchmarks.
- **References:** Mahalanobis (1936); Taha & Hanbury (2015).

---

## 4. Topology metrics

### 4.1 Centerline Dice (clDice)

- **Equation** (Shit et al., 2021). Let $S_G=\mathrm{skel}(G)$ and $S_P=\mathrm{skel}(P)$ be morphological skeletons:

    $$
    T_{\mathrm{prec}}(S_P,G)=\frac{\lvert S_P\cap G\rvert}{\lvert S_P\rvert},\quad
    T_{\mathrm{sens}}(S_G,P)=\frac{\lvert S_G\cap P\rvert}{\lvert S_G\rvert},\quad
    \mathrm{clDice}=\frac{2\,T_{\mathrm{prec}}\,T_{\mathrm{sens}}}{T_{\mathrm{prec}}+T_{\mathrm{sens}}}
    $$

- **Range / direction:** $[0,1]$, higher is better.
- **Edge cases:** Undefined for empty skeletons. MetricsReloaded returns NaN when both are empty and advises the max at aggregation.
- **Plain English:** Checks whether the *centre lines* of vessels or ducts are captured and whether the predicted centre lines stay inside the true structure. It rewards connected, unbroken tubes.
- **Good for:** Tubular structures such as vessels, airways, nerves and ducts. Metrics Reloaded recommends clDice as the overlap metric when the fingerprint indicates tubular structures and topology matters.
- **Pitfalls:**
    - The result depends on the skeletonization algorithm (3D thinning, e.g. Lee et al. 1994 in scikit-image) and on voxel anisotropy. Resample to isotropic spacing, or document the choice.
    - It is insensitive to vessel radius errors.
    - Skeletons of blobby, non-tubular structures are unstable.
- **References:** Shit et al. (2021); Maier-Hein et al. (2024).

### 4.2 Betti number error

- **Equation.** $\beta_0$, $\beta_1$ and $\beta_2$ count, respectively, connected components, independent loops (tunnels or handles) and enclosed cavities of a 3D binary object:

    $$
    \mathrm{BE}_k=\lvert\beta_k(P)-\beta_k(G)\rvert,\qquad \mathrm{BE}=\sum_{k=0}^{2}\mathrm{BE}_k
    $$

    It can be computed globally or averaged over random patches (Hu et al., 2019).

- **Range / direction:** Non-negative integers, lower is better. 0 means the same topology.
- **Conventions:** The connectivity pair must be declared, such as 26-connectivity for foreground with 6 for background, or vice versa. $\beta$ values depend on it.
- **Plain English:** Counts how many extra or missing pieces, loops and holes the prediction has compared with the truth.
- **Good for:** Vessel trees, airways, cortical surfaces and any task where connectivity is clinically relevant (for example whether a stenosis is split). $\mathrm{BE}_0$ alone is a useful component-count error for multi-lesion tasks.
- **Pitfalls:** It is spatially blind. A missing loop in one place and a spurious loop elsewhere give $\mathrm{BE}_1=0$. This is the motivation for Betti matching. It is sensitive to single-voxel noise, so consider a minimum component size.
- **References:** Hu et al. (2019); Stucki et al. (2023).

### 4.3 Betti matching error

- **Definition** (Stucki et al., 2023). Persistence barcodes of $G$ and $P$ are matched spatially using *induced matchings* through their union or comparison image. The error counts topological features (per dimension) that are unmatched in either direction. Unlike the Betti number error, a feature counts as correct only if it corresponds to a feature at the same location.
- **Range / direction:** Non-negative, lower is better.
- **Plain English:** Like the Betti error, but it also checks that each loop or piece is in the *right place*.
- **Good for:** Rigorous topological evaluation of curvilinear structures. It is also usable as a differentiable loss.
- **Pitfalls:** It requires persistent-homology software (the authors' `Betti-matching` C++/Python code), is computationally heavy in large 3D volumes, and depends on filtration and connectivity choices.
- **References:** Stucki et al. (2023).

### 4.4 Euler characteristic error

- **Equation:**

    $$
    \chi(X)=\beta_0-\beta_1+\beta_2,\qquad \mathrm{ECE}_\chi=\lvert\chi(P)-\chi(G)\rvert
    $$

    $\chi$ can be computed locally from voxel/edge/face/cube counts of the cubical complex ($\chi=V-E+F-C$), for example `skimage.measure.euler_number`.

- **Range / direction:** Non-negative integers, lower is better.
- **Plain English:** A quick single-number topology check that combines counts of pieces, loops and cavities.
- **Good for:** Cheap topology screening (for example cortical-surface genus: a sphere-like cortex has $\chi=2$).
- **Pitfalls:** Errors in different Betti numbers can cancel. For example, one extra component plus one extra loop gives $\Delta\chi = 0$. It depends on connectivity. Use it as a screen and follow with a Betti or Betti-matching analysis. Do not confuse the abbreviation with the Expected Calibration Error; SegEvalKit should name it `euler_error`.
- **References:** Standard algebraic topology (Euler–Poincaré formula); used as a topology-error metric alongside Betti errors in the topology-aware segmentation literature (e.g., Hu et al., 2019). This is our synthesis; no single canonical paper introduced it as a segmentation metric.

---

## 5. Detection and instance metrics

**Instance definition.** For semantic masks, instances are obtained by connected-component labelling (CCL). The connectivity (6, 18 or 26 in 3D) must be declared. Matching between predicted and reference instances uses a *localization criterion*: IoU > τ, any overlap, dilated overlap, centroid distance or Boundary IoU. Metrics Reloaded treats the choice of localization criterion and assignment strategy (greedy, Hungarian or one-to-many) as a first-class fingerprint decision.

### 5.1 Lesion-wise sensitivity, precision and F1

- **Aliases:**
    - Lesion true-positive rate (LTPR) and lesion false-positive rate (LFPR) (Carass et al., 2017).
    - Detection F1.
    - Object-level TPR/FPR (MedPy `obj_tpr`, `obj_fpr`).
- **Equations.** With $\mathrm{TP}_\ell$, $\mathrm{FP}_\ell$ and $\mathrm{FN}_\ell$ as *counts of lesions* after matching:

    $$
    \mathrm{LTPR}=\frac{\mathrm{TP}_\ell}{\mathrm{TP}_\ell+\mathrm{FN}_\ell},\quad
    \mathrm{LFPR}=\frac{\mathrm{FP}_\ell}{\mathrm{TP}_\ell+\mathrm{FP}_\ell},\quad
    \mathrm{F1}_\ell=\frac{2\,\mathrm{TP}_\ell}{2\,\mathrm{TP}_\ell+\mathrm{FP}_\ell+\mathrm{FN}_\ell}
    $$

    In LTPR and LFPR, the TP count is taken with respect to the reference and the prediction respectively, since one-to-many matches can make them differ.

- **Range / direction:** $[0,1]$. LTPR and F1: higher is better. LFPR: lower is better.
- **Plain English:** Of all true lesions, how many were found, and of all flagged lesions, how many were real?
- **Good for:** Multi-lesion diseases such as MS, brain metastases and lung nodules, where counting lesions matters clinically.
- **Pitfalls:**
    - The results depend heavily on the matching rule (overlap threshold, dilation, minimum lesion size) and on CCL connectivity.
    - Confluent lesions merge or split under CCL.
    - Very small reference lesions are often excluded by volume thresholds, and these must be reported.
- **References:** Carass et al. (2017); Commowick et al. (2018); Maier-Hein et al. (2024).

### 5.2 Lesion-wise Dice and lesion-wise HD95 (BraTS 2023 protocol)

- **Protocol** (BraTS 2023 evaluation code, `rachitsaluja/BraTS-2023-Metrics`; described in Moawad et al., 2023 and Kazerooni et al., 2024):
    1. For each tumour sub-region (WT, TC, ET), label the reference components (26-connectivity).
    2. For each reference lesion $g_i$, dilate it by $k$ iterations of a 3×3×3 18-connected structuring element ($k=3$ for GLI/SSA/PED and $k=1$ for MEN/MET). Every predicted component intersecting the dilated region is assigned to $g_i$.
    3. Compute Dice and HD95 (DeepMind implementation, spacing-aware) between $g_i$ and the union of its assigned predicted components.
    4. Predicted components assigned to no lesion are FP lesions.
    5. Reference lesions with volume ≤ θ are excluded (θ = 50 mm³, or 2 mm³ for METS).
- **Equations.** $\mathcal G$ is the set of retained reference lesions and $\mathcal F$ the set of FP predicted components:

    $$
    \mathrm{LesionDice}=\frac{\sum_{i\in\mathcal G}\mathrm{DSC}(g_i,\hat p_i)}{\lvert\mathcal G\rvert+\lvert\mathcal F\rvert},\qquad
    \mathrm{LesionHD95}=\frac{\sum_{i\in\mathcal G}\mathrm{HD95}(g_i,\hat p_i)+374\,\lvert\mathcal F\rvert}{\lvert\mathcal G\rvert+\lvert\mathcal F\rvert}
    $$

    Missed lesions (FN) contribute DSC = 0 and HD95 = ∞, replaced by 374 mm. FP lesions contribute 0 and 374. If both reference and prediction are empty (0/0), the code sets LesionDice = 1 and LesionHD95 = 0.

- **Range / direction / units:** Dice in $[0,1]$, higher is better. HD95 in $[0,374]$ mm, lower is better.
- **Plain English:** Scores each tumour focus separately and averages, so a missed small metastasis counts as much as a well-segmented large one.
- **Good for:** Multi-focal disease, where whole-volume Dice is dominated by the largest lesion.
- **Pitfalls:**
    - The dilation-based matching can merge nearby lesions.
    - The 374 mm penalty is specific to the atlas space. For other fields of view, use the image diagonal and document it.
    - The volume threshold removes tiny lesions from the denominator.
- **References:** BraTS 2023 metrics code; Moawad et al. (2023) (BraTS-METS); Kazerooni et al. (2024) (BraTS-PEDs).

### 5.3 Free-response ROC (FROC), FPs per scan and CPM

- **Definition.** Sweep a detection-confidence threshold $t$. At each $t$ plot lesion sensitivity $\mathrm{LTPR}(t)$ against the mean number of false positives per image $\overline{\mathrm{FP}}(t)$. The LUNA16 *competition performance metric* (CPM) is the mean sensitivity at 7 predefined FP rates:

    $$
    \mathrm{CPM}=\frac17\sum_{f\in\{\frac18,\frac14,\frac12,1,2,4,8\}}\mathrm{LTPR}\big(\overline{\mathrm{FP}}=f\big)
    $$

    Related measures are JAFROC (Chakraborty & Berbaum, 2004) and partial area under the FROC curve up to a maximum number of FPs per scan.

- **Range / direction:** Sensitivity in $[0,1]$ at a given FP/scan, higher is better.
- **Plain English:** "How many real lesions does the algorithm find if we tolerate *x* false alarms per scan?"
- **Good for:** Screening and detection tasks (nodules, metastases, microbleeds) with lesion-level confidence scores. Metrics Reloaded lists FROC as a multi-threshold detection metric.
- **Pitfalls:** It requires per-lesion confidence scores, whereas binary masks yield a single operating point. It depends on the matching criterion. FP-per-scan normalization differs from FP-per-image in 2D.
- **References:** Bunch et al. (1978); Chakraborty & Berbaum (2004); Setio et al. (2017); Maier-Hein et al. (2024).

### 5.4 Panoptic Quality (PQ), segmentation quality (SQ) and recognition quality (RQ)

- **Equation** (Kirillov et al., 2019). Predicted instance $p$ matches reference instance $g$ if $\mathrm{IoU}(p,g)>0.5$. This guarantees a unique matching. Then:

    $$
    \mathrm{PQ}=\frac{\sum_{(p,g)\in\mathrm{TP}}\mathrm{IoU}(p,g)}{\lvert\mathrm{TP}\rvert+\frac12\lvert\mathrm{FP}\rvert+\frac12\lvert\mathrm{FN}\rvert}
    =\underbrace{\frac{\sum_{(p,g)\in\mathrm{TP}}\mathrm{IoU}(p,g)}{\lvert\mathrm{TP}\rvert}}_{\mathrm{SQ}}\times\underbrace{\frac{\lvert\mathrm{TP}\rvert}{\lvert\mathrm{TP}\rvert+\frac12\lvert\mathrm{FP}\rvert+\frac12\lvert\mathrm{FN}\rvert}}_{\mathrm{RQ}}
    $$

- **Range / direction:** $[0,1]$, higher is better. RQ is the detection F1, and SQ is the mean IoU of matched pairs.
- **Edge cases:** Undefined when there are no instances in either mask (0/0). Conventions set it to 1 or NaN.
- **Plain English:** One number combining "did we find each lesion?" (RQ) with "how well did we outline the ones we found?" (SQ).
- **Good for:** Instance segmentation (multiple lesions, cells, vertebrae). Metrics Reloaded recommends PQ as especially suited to instance segmentation. Variants in `panoptica` replace IoU with DSC or ASSD inside SQ and allow a lower matching threshold with Hungarian assignment (MONAI uses Munkres for thresholds < 0.5).
- **Pitfalls:**
    - With IoU > 0.5, small lesions whose IoU falls just below 0.5 count as both a FN *and* a FP.
    - It is sensitive to instance-split and instance-merge errors.
    - Pooling over a dataset vs averaging per image gives different results.
- **References:** Kirillov et al. (2019); Kofler et al. (2023, panoptica); Maier-Hein et al. (2024).

### 5.5 Connected-component split/merge errors and lesion count error

- **Definitions.** Build the bipartite overlap graph between reference components $\{g_i\}$ and predicted components $\{p_j\}$, with an edge where $\lvert g_i\cap p_j\rvert>0$ (or above a threshold):
    - **Split error:** a reference component overlapped by ≥ 2 predicted components. The count is $\sum_i \max(0,\deg(g_i)-1)$.
    - **Merge error:** a predicted component overlapping ≥ 2 reference components. The count is $\sum_j \max(0,\deg(p_j)-1)$.
    - **Lesion count error:** $\lvert\, \#\{p_j\}-\#\{g_i\}\,\rvert$, or signed.
    - The information-theoretic counterpart is *split/merge variation of information* (Nunez-Iglesias et al., 2013; Arganda-Carreras et al., 2015):

        $$
        \mathrm{VI}_{\text{split}}=H(P\mid G),\qquad \mathrm{VI}_{\text{merge}}=H(G\mid P),\qquad \mathrm{VI}=\mathrm{VI}_{\text{split}}+\mathrm{VI}_{\text{merge}}
        $$

    These are computed on the joint label distribution of the two instance labelings. In the connectomics convention, $H(P\mid G)$ is labelled the split term and $H(G\mid P)$ the merge term.

- **Range / direction:** Counts and VI are $\ge0$, lower is better.
- **Plain English:** Did the algorithm break one lesion into several, or fuse several lesions into one?
- **Good for:** Lesion counting (MS lesion load, metastasis count), vertebra and tooth instance labelling, and neuron or vessel reconstruction.
- **Pitfalls:** The result depends on CCL connectivity and on overlap thresholds. Merges are invisible to voxel-level Dice.
- **References:** Arganda-Carreras et al. (2015); Nunez-Iglesias et al. (2013); Meilă (2007); Kofler et al. (2023).

---

## 6. Calibration and probabilistic metrics

These metrics apply to *soft* outputs ($p_i\in[0,1]$, or softmax vectors $\mathbf p_i$) and are often neglected in segmentation toolkits.

### 6.1 Expected calibration error (ECE) and maximum calibration error (MCE)

- **Equation** (Naeini et al., 2015; Guo et al., 2017). Partition voxels into $M$ equal-width confidence bins $B_m$:

    $$
    \mathrm{ECE}=\sum_{m=1}^{M}\frac{\lvert B_m\rvert}{n}\,\big\lvert\mathrm{acc}(B_m)-\mathrm{conf}(B_m)\big\rvert,\qquad \mathrm{MCE}=\max_m\big\lvert\mathrm{acc}(B_m)-\mathrm{conf}(B_m)\big\rvert
    $$

    $\mathrm{conf}(B_m)$ is the mean predicted confidence in the bin and $\mathrm{acc}(B_m)$ the fraction correct. For binary foreground calibration, use $\mathrm{conf}=\bar p$ and $\mathrm{acc}=\bar y$ in each bin (reliability diagram). Variants:

    - Class-wise ECE (Kull et al., 2019).
    - Adaptive / equal-mass binning (Nixon et al., 2019).
- **Range / direction:** $[0,1]$, lower is better.
- **Plain English:** When the model says "80% sure this voxel is tumour", is it right about 80% of the time?
- **Good for:** Checking whether probability maps can be trusted for uncertainty-aware decisions, thresholding or active learning. Mehrtash et al. (2020) show that Dice-loss-trained segmentation CNNs are poorly calibrated and that ensembling improves calibration.
- **Pitfalls:**
    - Whole-volume ECE is dominated by trivially correct background voxels with $p\approx0$, which makes it look excellent. Common practice is to restrict computation to a region of interest (for example a dilated union of $G$ and $P$) and to report that region.
    - The number of bins $M$ and the binning scheme change the value.
    - It is not a proper scoring rule, so a model can have low ECE and poor discrimination.
- **References:** Naeini et al. (2015); Guo et al. (2017); Mehrtash et al. (2020); Kull et al. (2019); Nixon et al. (2019).

### 6.2 Brier score

- **Equation** (Brier, 1950):

    $$
    \mathrm{BS}=\frac1n\sum_{i=1}^n (p_i-y_i)^2\quad\text{(binary)},\qquad \mathrm{BS}=\frac1n\sum_{i=1}^n\sum_{c=1}^{C}(p_{ic}-y_{ic})^2\quad\text{(multi-class)}
    $$

- **Range / direction:** $[0,1]$ binary ($[0,2]$ multi-class), lower is better.
- **Plain English:** The mean squared difference between predicted probabilities and the true 0/1 labels.
- **Good for:** A proper scoring rule combining calibration and discrimination. Mehrtash et al. (2020) use it for segmentation.
- **Pitfalls:** The same background-domination issue as ECE, so compute it in an ROI or per class. Its absolute values are hard to interpret without a reference, and a Brier skill score against a baseline helps.
- **References:** Brier (1950); Mehrtash et al. (2020).

### 6.3 Negative log-likelihood (NLL, cross-entropy)

- **Equation:** $\;\mathrm{NLL}=-\frac1n\sum_i\big[y_i\log p_i+(1-y_i)\log(1-p_i)\big]$ (binary). The multi-class form is $-\frac1n\sum_i\log p_{i,y_i}$.
- **Range / direction:** $[0,\infty)$, lower is better.
- **Plain English:** How surprised the model is by the true labels. Confident mistakes are heavily penalized.
- **Good for:** Proper scoring of probabilistic outputs and comparing calibration methods such as temperature scaling (Guo et al., 2017; Mehrtash et al., 2020).
- **Pitfalls:** It is unbounded, so clip $p$ with $\epsilon$. It is dominated by a few confident errors and by the background.
- **References:** Guo et al. (2017); Mehrtash et al. (2020).

### 6.4 Probabilistic distance (PBD)

- **Equation** (Gerig et al., 2001; Taha & Hanbury, 2015, Eq. 43). $f_G$ and $f_P$ are (fuzzy or probabilistic) membership maps:

    $$
    \mathrm{PBD}(G,P)=\frac{\sum_x\lvert f_G(x)-f_P(x)\rvert}{2\sum_x f_G(x)\,f_P(x)}
    $$

    *Verification caveat:* in the machine-extracted text of Taha & Hanbury's Eq. 43 we could not tell whether the "2" is a squared exponent on the numerator or a factor in the denominator. The form above ($L_1$ difference over twice the joint probability) follows Gerig et al.'s VALMET definition as we understand it. SegEvalKit should confirm it against the typeset PDF and test against EvaluateSegmentation's output before release.

- **Range / direction:** $[0,\infty)$, lower is better. It is undefined when $\sum f_Gf_P=0$ (no overlap).
- **Plain English:** Compares probability maps directly, without thresholding them.
- **Good for:** Fuzzy or probabilistic references (for example STAPLE maps and multi-rater averages).
- **Pitfalls:** It becomes infinite with no overlap. It is rarely used, so give it low priority.
- **References:** Gerig et al. (2001); Taha & Hanbury (2015).

### 6.5 Uncertainty–error overlap (UEO)

- **Equation** (Jungo & Reyes, 2019). With a voxel error map $E=G\,\triangle\,P$ and a thresholded uncertainty map $U_t=\{i: u_i>t\}$:

    $$
    \mathrm{UEO}=\max_t\ \mathrm{DSC}(U_t, E)
    $$

    The value at a fixed $t$ may also be reported.

- **Range / direction:** $[0,1]$, higher is better.
- **Plain English:** Does the model's uncertainty map highlight the regions where it actually made mistakes?
- **Good for:** Evaluating voxel-wise uncertainty estimates (MC dropout, ensembles, test-time augmentation) for human-in-the-loop correction.
- **Pitfalls:** The max over $t$ is optimistic, so choose $t$ on validation data. Errors concentrate at boundaries, which rewards trivial "boundary-uncertainty" maps.
- **References:** Jungo & Reyes (2019); Mehrtash et al. (2020).

---

## 7. Information-theoretic, clustering and agreement metrics

These treat the two segmentations as partitions (clusterings) of $\Omega$. Taha & Hanbury (2015) give binary-case closed forms. Below, $p_{jk}$ denotes the joint proportion of voxels in reference class $j$ and predicted class $k$ (for binary: $p_{11}=TP/N$, $p_{10}=FN/N$, $p_{01}=FP/N$, $p_{00}=TN/N$), and $H(\cdot)$ is Shannon entropy.

### 7.1 Mutual information (MI)

- **Equation** (Taha & Hanbury, 2015, Eq. 38):

    $$
    \mathrm{MI}(G,P)=H(G)+H(P)-H(G,P),\quad H(G)=-\sum_j p_{j\cdot}\log p_{j\cdot},\quad H(G,P)=-\sum_{j,k}p_{jk}\log p_{jk}
    $$

- **Range / direction:** $[0,\min(H(G),H(P))]$ (in bits or nats), higher is better. Normalized variants (NMI) map to $[0,1]$.
- **Plain English:** How much knowing one segmentation tells you about the other.
- **Good for:** Taha & Hanbury (2015) note that it rewards recall. It also applies to multi-class partitions.
- **Pitfalls:** It is TN-influenced and depends on field of view. Unnormalized values are not comparable across images, and there is no intuitive clinical reading.
- **References:** Taha & Hanbury (2015); Cover & Thomas (2006).

### 7.2 Variation of information (VOI)

- **Equation** (Meilă, 2007; Taha & Hanbury, 2015, Eq. 39):

    $$
    \mathrm{VOI}(G,P)=H(G)+H(P)-2\,\mathrm{MI}(G,P)=H(G\mid P)+H(P\mid G)
    $$

- **Range / direction:** $[0,\log N]$, lower is better. It is a true metric on partitions.
- **Plain English:** How much information is lost and gained when switching from one segmentation to the other.
- **Good for:** Multi-label and instance partitions (it decomposes into split and merge terms, §5.5) and topology-aware benchmarks (Hu et al., 2019).
- **Pitfalls:** Same as MI. It depends on the number of voxels and on field of view.
- **References:** Meilă (2007); Taha & Hanbury (2015).

### 7.3 Global consistency error (GCE)

- **Equation** (Martin et al., 2001). Taha & Hanbury (2015, Eq. 19) give the binary form:

    $$
    \mathrm{GCE}=\frac1N\min\left\{\frac{FN(FN+2TP)}{TP+FN}+\frac{FP(FP+2TN)}{TN+FP},\ \frac{FP(FP+2TP)}{TP+FP}+\frac{FN(FN+2TN)}{TN+FN}\right\}
    $$

    This comes from the local refinement error $E(S_1,S_2,x)=\frac{\lvert R(S_1,x)\setminus R(S_2,x)\rvert}{\lvert R(S_1,x)\rvert}$, where $R(S,x)$ is the region of $S$ containing voxel $x$.

- **Range / direction:** $[0,1]$, lower is better.
- **Plain English:** Measures how far one segmentation is from being a refinement of the other. It forgives one segmentation being a finer version of the other.
- **Pitfalls:** The refinement tolerance means trivial segmentations (everything one region, or every voxel its own region) score 0. It is rarely appropriate for binary medical segmentation. Include it only for completeness and comparability with EvaluateSegmentation and pymia.
- **References:** Martin et al. (2001); Taha & Hanbury (2015).

### 7.4 Rand index (RI) and adjusted Rand index (ARI)

- **Equation.** Contingency-table form (Hubert & Arabie, 1985). With $n_{jk}$ the joint counts, marginals $a_j=\sum_k n_{jk}$ and $b_k=\sum_j n_{jk}$, and $\binom{\cdot}{2}$ pair counts:

    $$
    \mathrm{ARI}=\frac{\sum_{jk}\binom{n_{jk}}{2}-\Big[\sum_j\binom{a_j}{2}\sum_k\binom{b_k}{2}\Big]\Big/\binom{N}{2}}{\frac12\Big[\sum_j\binom{a_j}{2}+\sum_k\binom{b_k}{2}\Big]-\Big[\sum_j\binom{a_j}{2}\sum_k\binom{b_k}{2}\Big]\Big/\binom{N}{2}}
    $$

    Taha & Hanbury (2015, Eq. 32) express this with pair-agreement counts $a,b,c,d$ as $\mathrm{ARI}=\frac{2(ad-bc)}{c^2+b^2+2ad+(a+d)(c+b)}$. The Rand index is $\mathrm{RI}=(a+d)/(a+b+c+d)$ (Rand, 1971).

- **Range / direction:** RI is in $[0,1]$. ARI is in $[-1,1]$ with 0 at chance. Higher is better.
- **Plain English:** Of all pairs of voxels, how often the two segmentations agree on whether the pair belongs together, corrected for chance (ARI).
- **Good for:** Instance or cluster partitions (cell and neuron segmentation) and topology benchmarks. For binary 3D masks it is TN-dominated.
- **Pitfalls:** It involves $O(N^2)$ pair counts, so use int64 or float for 3D volumes, where $N^2\sim10^{15}$. Its interpretation is weak for binary tasks.
- **References:** Rand (1971); Hubert & Arabie (1985); Taha & Hanbury (2015).

---

## 8. Statistical aggregation, uncertainty and ranking

This section covers how per-case metric values become study-level conclusions. Metrics Reloaded recommends computing metrics **per image (case) first, then aggregating over images**, respecting the hierarchical data structure (patients, then scans, then lesions, then classes).

### 8.1 Per-case aggregation and reporting

- **Recommended statistics:**
    - Mean ± SD *and* median with interquartile range (IQR).
    - The full distribution (box, violin or strip plots).
    - The number of failures or undefined cases, reported explicitly.
- **Empty and undefined values:** NaN must not be silently dropped. The handling (dropping vs. substituting worst or best values) must be stated. MetricsReloaded's code comments say, for example, "set to worst case in aggregation" for HD with one empty mask, and "set to 1" for DSC with both empty.
- **Multi-class:** Report per class. Class-averaged scores hide failures on small classes (Maier-Hein et al., 2024).
- **Pitfall:** Pooling voxels across the dataset before computing DSC ("global Dice") weights big patients more and differs from mean per-case DSC. State which one is reported.

### 8.2 Bootstrap confidence intervals

- **Procedure** (Efron, 1979). Resample the $n$ cases with replacement $B$ times (for example $B=1000$–$10{,}000$), recompute the aggregate $\hat\theta^{*b}$ each time, and report the percentile interval $[\hat\theta^*_{(\alpha/2)},\hat\theta^*_{(1-\alpha/2)}]$ (or BCa). For paired method comparisons, resample the *per-case differences*.
- **Pitfall:** Resample at the patient level when patients contribute several scans (cluster bootstrap).
- **References:** Efron (1979); Wiesenfarth et al. (2021).

### 8.3 Paired hypothesis tests (Wilcoxon signed-rank) and multiplicity

- **Wilcoxon signed-rank test** (Wilcoxon, 1945). With per-case differences $\delta_j=m_A(j)-m_B(j)$, rank $\lvert\delta_j\rvert$ (dropping zeros) and compute

    $$
    W^+=\sum_{j:\delta_j>0}\mathrm{rank}(\lvert\delta_j\rvert)
    $$

    Compare $W^+$ with its null distribution (exact, or normal approximation with tie correction). This is the default for paired, non-normal, bounded metrics such as DSC.

- **Multiple comparisons:** Use Holm (1979) step-down or Benjamini–Hochberg when comparing several methods, metrics or classes. For more than 2 methods across many cases, use a Friedman test with post-hoc Nemenyi (Demšar, 2006).
- **Pitfalls:**
    - Statistical significance is not clinical relevance, so report effect sizes (median difference with bootstrap CI).
    - Metric saturation near 1 compresses differences.
- **References:** Wilcoxon (1945); Holm (1979); Demšar (2006).

### 8.4 Ranking schemes and ranking robustness

- **Schemes** (Maier-Hein et al., 2018; Wiesenfarth et al., 2021):
    - *Aggregate-then-rank*: rank methods by mean or median metric.
    - *Rank-then-aggregate*: rank methods per case, then average the ranks.
    - *Test-based (significance) ranking*: count significant pairwise wins, for example one-sided Wilcoxon with Holm adjustment.
- **Robustness.** Bootstrap the test cases and recompute rankings. Summarize with Kendall's $\tau$ between the full-data ranking and bootstrap rankings, and visualize with blob plots or ranking heatmaps (challengeR).
- **Key finding:** Maier-Hein et al. (2018) showed that challenge rankings are often *not robust* to the choice of test data, ranking scheme, metric, aggregation operator and annotator. Winners can change with these choices, so rankings must be accompanied by uncertainty analysis.
- **Missing values:** Rankings must define how failed or missing predictions are treated (usually worst rank or worst metric value).
- **References:** Maier-Hein et al. (2018); Wiesenfarth et al. (2021).

---

## 9. Empty-mask and edge-case conventions

This table summarizes each library's native behaviour, from source code inspected in September 2026 (MetricsReloaded `main`, MONAI `dev`, DeepMind `master`, BraTS-2023-Metrics `main`). "Worst" and "best" are the aggregation-time recommendations in MetricsReloaded comments.

| Case | DSC / IoU | HD / HD95 / ASSD | NSD / BIoU / clDice |
|---|---|---|---|
| $G=\emptyset, P=\emptyset$ | MetricsReloaded: NaN, set to 1 when aggregating. MONAI (`ignore_empty=True`): NaN, excluded. BraTS 2023: 1.0 | MetricsReloaded: NaN ("set to 0"). BraTS 2023: 0. DeepMind: `inf`/NaN | MetricsReloaded: NaN, set to best |
| $G=\emptyset, P\neq\emptyset$ | 0. MONAI with `ignore_empty=True`: NaN | MetricsReloaded: NaN, set to worst. DeepMind: `inf`. MONAI: NaN/`inf` with warning. BraTS: 374 | 0, or NaN depending on tool |
| $G\neq\emptyset, P=\emptyset$ | 0 | As above (worst). BraTS: 374 | 0 |

**Recommendation for SegEvalKit:**

1. Return a structured result (`value`, `defined: bool`, `reason`), never a silent NaN.
2. Provide named policies:
    - `"nan"` (default for per-case output).
    - `"metrics_reloaded"` (best or worst as above).
    - `"brats2023"` (DSC 1/0, HD95 0/374).
    - `"penalty=image_diagonal_mm"`.
3. Always report counts of undefined cases per metric.

---

## 10. Metric selection guide

Metrics Reloaded selects metrics from a **problem fingerprint**. The steps are: first map the task to a *problem category* (image-level classification, object detection, semantic segmentation or instance segmentation), then answer fingerprint questions. Examples of fingerprint items:

- Domain interest in boundaries (FP2.5.x): whether errors should be counted by existence or by distance, and whether annotation imprecision should be compensated (FP2.5.7).
- Size of structures relative to the grid (FP3.1).
- Tubular shape (FP3.3).
- Noisy reference (FP4.3.1).
- Possibility of empty references.
- Unequal FP/FN costs.

The general rule is to **combine one overlap-based metric with one boundary-based metric**, and for instance problems to add **detection metrics** plus per-instance segmentation metrics.

### 10.1 Decision table

| Scenario / structure | Primary overlap | Boundary / distance | Additional | Rationale and key fingerprint items |
|---|---|---|---|---|
| **Large, compact organ** (liver, kidney, lung, brain) | DSC (or IoU) | NSD@τ (τ from inter-rater variability) + HD95 | RVD (signed) | DSC is informative for large objects. NSD tolerates annotation imprecision (FP2.5.7). HD95 catches gross leakage (Maier-Hein et al., 2024). |
| **Small lesions / structures** (few voxels relative to grid, e.g. small metastases, lymph nodes) | DSC reported with caution; $F_\beta$ if costs are asymmetric | ASSD/MASD or NSD, **not** HD alone | Lesion-wise F1 or LTPR/LFPR; centroid distance | DSC is unstable for small objects (FP3.1). Taha & Hanbury (2015) recommend distance-based metrics for small segments. Metrics Reloaded warns against DSC when structures are consistently small *and* the reference is noisy. |
| **Tubular / curvilinear** (vessels, airways, nerves, pancreatic duct) | **clDice** (+ DSC) | NSD or MASD | Betti errors ($\beta_0,\beta_1$), Betti matching error, Euler error | Tubular fingerprint (FP3.3) selects clDice. Topology correctness (connectivity) matters clinically (Shit et al., 2021; Stucki et al., 2023). |
| **Multi-instance lesions** (MS lesions, metastases, nodules) | Per-lesion DSC for matched pairs | Per-lesion HD95 / ASSD | **PQ** (SQ, RQ), lesion-wise F1, FROC/CPM if scores exist, split/merge counts, lesion-count error, BraTS-style LesionDice/LesionHD95 | Treat as instance segmentation: detection metrics first, then per-instance segmentation metrics. Declare the localization criterion and matching strategy (Maier-Hein et al., 2024; Kirillov et al., 2019). |
| **Boundary-critical: radiotherapy OAR / target contouring** | DSC | **NSD@τ** (organ-specific τ), **surface overlap** in both directions, HD95 (and HD for safety) | Added path length (not covered here), $F_\beta$ (target coverage) | NSD approximates clinical editing effort (Nikolov et al., 2021). HD/HD95 capture worst-case geometric error relevant to dose. Directed surface overlap separates under- from over-contouring. |
| **Volumetry** (tumour burden, atrophy, organ volume) | DSC (for sanity) | – | **AVD (mL)**, signed RVD, **Bland–Altman** bias and LoA, **ICC(2,1)**, VS | Volume is the endpoint. VS and RVD ignore position, so pair them with DSC (Taha & Hanbury, 2015; Bland & Altman, 1986). |
| **Uncertainty / probabilistic outputs** | Soft DSC (optional) | – | **ECE** (in an ROI, with reliability diagram), **Brier**, **NLL**, **UEO** | Calibration determines whether probabilities are trustworthy (Guo et al., 2017; Mehrtash et al., 2020; Jungo & Reyes, 2019). |
| **Possibly empty references** (e.g. enhancing tumour absent; post-op) | DSC with explicit empty policy; image-level detection (TP/FP/TN/FN of *presence*) | HD95 with penalty policy | Case-level sensitivity and specificity for presence | DSC and HD are undefined when the reference is empty. Metrics Reloaded and BraTS define explicit policies. Report presence detection separately. |
| **Inter-rater / agreement studies** | DSC, Cohen's κ (multi-class) | NSD, HD95 | ICC (volumes), Bland–Altman | These are agreement rather than accuracy questions. Report rater-vs-rater baselines as the ceiling. |
| **Benchmark / challenge ranking** | As above per task | As above | Bootstrap CIs, Wilcoxon + Holm, ranking robustness (Kendall τ, blob plots) | Rankings are fragile (Maier-Hein et al., 2018). Use challengeR-style analysis (Wiesenfarth et al., 2021). |

### 10.2 Metrics to avoid as primary endpoints in 3D segmentation

- **Accuracy, specificity and voxel-wise ROC-AUC.** They are dominated by true negatives and depend on field of view.
- **VS or RVD alone.** They are blind to position.
- **HD (max) alone.** It is dictated by a single outlier.
- **GCE and RI.** Their interpretation is weak for binary tasks.
- **Mean DSC across structures of very different sizes.** It is not comparable. Report per class.

---

## 11. Survey of existing tools and gaps for SegEvalKit

| Tool | Language / form | Coverage (segmentation-relevant) | Spacing / 3D | Empty-case handling | Notable limitations |
|---|---|---|---|---|---|
| **MetricsReloaded** (Project-MONAI/MetricsReloaded) | Python | Reference implementation of the Metrics Reloaded recommendations: DSC, IoU, Fβ, sensitivity, PPV, MCC, Cohen's κ, Youden, expected cost, net benefit, centreline DSC (clDice), Boundary IoU, NSD, HD, HD-percentile, ASSD, MASD, centre-of-mass distance, absolute volume difference ratio; detection and instance processes (matching, FROC, AP, PQ); calibration measures; aggregation utilities | Spacing via `pixdim`; voxel-border boundaries | Returns NaN with warnings; the `_nanrep` aggregation columns replace NaN with the best value (Dice, NSD) or the image diagonal (all distance metrics, even when both masks are empty) | Calibration metrics (ECE, class-wise ECE, MCE, Brier, root Brier, log score, NLL, kernel calibration error) exist in `calibration_measures.py`, but the evaluation pipeline applies them only to image-level classification; no Betti or topology errors; no volume-agreement statistics (ICC, Bland–Altman); boundary-voxel (not surfel) NSD; heavy per-pair object model; limited NIfTI I/O and CLI |
| **MONAI** `monai.metrics` | Python / PyTorch | Dice (with `ignore_empty`), Mean IoU, Generalized Dice, confusion-matrix metrics (sensitivity, specificity, precision, F1, MCC, …), HD / HD-percentile (directed option), surface distance (ASSD), surface Dice (NSD, optional sub-voxels), Panoptic Quality (2D/3D, Munkres for IoU < 0.5), FROC, ROC-AUC, AP, absolute volume difference, calibration (ECE via binning) | Spacing argument supported; GPU tensors | NaN / `inf` with warnings; `not_nans` tracking | Batch-tensor API (not file-based); no clDice or topology; no lesion-wise matching pipeline beyond PQ; no statistics or ranking |
| **seg-metrics** (Jia, Staring & Stoel, 2024) | Python, PyPI; `write_metrics` evaluates files or folders into a CSV (no CLI entry point); many file formats | Dice, Jaccard, precision, recall, FPR, FNR, volume similarity, HD, HD95, mean / median / SD surface distance | Reads spacing from images (SimpleITK) | All distances (HD, HD95, MSD) are 0 whenever either mask is empty, because an empty mask gets an all-zero distance map | No NSD, clDice, instance, calibration or statistics |
| **surface-distance** (google-deepmind) | Python | `compute_surface_distances` (surfel-area weighted), average surface distance (both directed), robust Hausdorff (percentile), surface overlap at tolerance, surface Dice at tolerance, Dice | Correct anisotropic spacing via surfel areas; 3D | `inf` distances when a mask is empty | Distance metrics only; unmaintained or low activity; no overlap suite, instances or statistics |
| **MedPy** `medpy.metric.binary` | Python | dc, jc, precision, recall / sensitivity, specificity, TPR / TNR, hd, hd95 (pooled percentile), asd (directed), assd (pooled mean of both distance sets in 0.5.2; mean of directed means up to 0.4.0), ravd, volume correlation, volume-change correlation, object-wise ASD / ASSD / TPR / FPR | `voxelspacing` argument | Raises `RuntimeError` for empty objects in distance metrics; `dc` returns 1 for two empty masks (0 up to 0.4.0) | Pooled HD95 differs from the MetricsReloaded, MONAI and BraTS convention; naming ambiguity (asd vs assd) and a version-dependent `assd`; no NSD, clDice or calibration |
| **panoptica** (BrainLesion; Kofler et al., 2023) | Python | Instance approximation (CCL), instance matching (naive, max-bipartite / Hungarian, IoU or DSC thresholds), instance evaluation: PQ, SQ, RQ with IoU, DSC, ASSD, clDice, HD, HD95, NSD, RVD, RVAE and centre distance per instance and globally; semantic and instance maps in 2D/3D | 3D, spacing-aware for distances | Configurable | Focused on instance evaluation; no calibration or volumetry statistics; no ranking |
| **EvaluateSegmentation** (Taha & Hanbury, VISCERAL) | C++ command-line tool (ITK) | 20 metrics from Taha & Hanbury (2015): DICE, JAC, TPR, TNR, FPR, FNR, PPV, F-measure, GCE, VS, RI, ARI, MI, VOI, ICC, PBD, KAP, AUC, HD (with quantile), AVD, MHD; fuzzy inputs; efficient HD | 3D, spacing-aware | Tool-specific | C++ binary, not a Python API, one image pair per call; no NSD, clDice or calibration metrics; lesion evaluation only as a separate detection mode (`-det`) on coordinate files; no statistics |
| **pymia** (Jungo et al., 2021) | Python | Evaluation module mirrors Taha & Hanbury: Dice, Jaccard, sensitivity, specificity, precision, F-measure, accuracy, fallout, FNR, AUC, Cohen's κ, GCE, VS, RI, ARI, MI, VOI, ICC, PBD, Mahalanobis, HD (percentile), average distance, SurfaceDiceOverlap, SurfaceOverlap, reference / prediction volume; writers (CSV, statistics) | SimpleITK images with spacing | Warnings (`NotComputableMetricWarning`) | No clDice, topology, instance or PQ, or calibration; basic statistics only |
| **torchmetrics** (Detlefsen et al., 2022) | Python / PyTorch | `segmentation`: DiceScore, GeneralizedDiceScore, MeanIoU, HausdorffDistance; `classification`: calibration error (ECE / MCE), MCC, Cohen's κ, AUROC, Fβ; `detection`: PanopticQuality | Tensor-based; spacing support limited | Configurable reductions | Not NIfTI-aware; few surface metrics (no NSD or ASSD); no lesion-wise analysis |
| *(Related)* **challengeR** (Wiesenfarth et al., 2021) | R | Ranking schemes, bootstrap ranking robustness, significance maps, visualizations | – | – | Operates on metric tables, not images; R only |

### 11.1 Gaps SegEvalKit can fill

1. **One NIfTI-native pipeline.** It would read image pairs (or folders and CSV manifests), check that affines and shapes match, handle multi-label maps, and compute every family in this document with one configuration. No single existing Python tool covers overlap, surface, topology, instance, calibration and statistics together.
2. **Explicit, switchable conventions**, with a reference test suite that pins them down:
    - Empty-mask policies (NaN, Metrics Reloaded, BraTS 2023, image-diagonal penalty).
    - HD-percentile mode (directed-max vs pooled).
    - Boundary weighting (voxel vs surfel area).
    - CCL connectivity.
3. **Cross-library conformance tests.** Numerical agreement with MetricsReloaded, MONAI, DeepMind, MedPy and BraTS 2023 under each convention, with documented differences. This is a real gap, since users currently get different HD95 and NSD values from different tools without knowing why.
4. **Lesion-wise evaluation.** Configurable matching (IoU, dilated overlap as in BraTS, centroid distance, Boundary IoU; greedy or Hungarian; minimum lesion volume in mm³). Outputs would include PQ, SQ, RQ, lesion F1, split and merge counts, lesion-count error, LesionDice and LesionHD95, and FROC/CPM when lesion confidences are available.
5. **Topology metrics for 3D.** clDice with isotropic resampling option, Betti numbers (β0/β1/β2) and errors, Euler-characteristic error, and optional Betti matching through the authors' implementation.
6. **Calibration for segmentation.** ROI-restricted ECE and MCE (equal-width and equal-mass bins), Brier, NLL, reliability-diagram data and UEO. Segmentation-focused tools offer at most binned ECE (MONAI 1.6.1 `CalibrationErrorMetric`, TorchMetrics `CalibrationError`); MetricsReloaded implements ECE, Brier and NLL but applies them only to image-level classification.
7. **Volumetry agreement.** AVD in mL, signed RVD, Bland–Altman (bias, LoA with CIs) and ICC(2,1) and ICC(3,1) at dataset level. No surveyed Python segmentation tool provides these.
8. **Statistics and ranking.** Per-case, then patient-cluster, aggregation; bootstrap CIs; paired Wilcoxon with Holm or BH correction; effect sizes; and challengeR-style ranking robustness (Kendall τ over bootstrap) in Python.
9. **A metric-selection helper.** It would take a Metrics Reloaded-style fingerprint (structure size, tubularity, instances, boundary importance, empty references, probabilistic output) and propose the metric set in §10.1, citing each recommendation.
10. **Clinician-readable reports.** HTML or Markdown per-case and per-class tables, with the plain-English explanations from this document attached to each metric.

---

## 12. References

1. Antonelli M, Reinke A, Bakas S, et al. The Medical Segmentation Decathlon. *Nature Communications* 13, 4128 (2022). https://doi.org/10.1038/s41467-022-30695-9
2. Arganda-Carreras I, Turaga SC, Berger DR, et al. Crowdsourcing the creation of image segmentation algorithms for connectomics. *Frontiers in Neuroanatomy* 9, 142 (2015). https://doi.org/10.3389/fnana.2015.00142
3. Beneš M, Zitová B. Performance evaluation of image segmentation algorithms on microscopic image data. *Journal of Microscopy* 257(1), 65–85 (2015).
4. Bland JM, Altman DG. Statistical methods for assessing agreement between two methods of clinical measurement. *The Lancet* 327(8476), 307–310 (1986). https://doi.org/10.1016/S0140-6736(86)90837-8
5. Brier GW. Verification of forecasts expressed in terms of probability. *Monthly Weather Review* 78(1), 1–3 (1950). [doi:10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2](https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2)
6. Bunch PC, Hamilton JF, Sanderson GK, Simmons AH. A free-response approach to the measurement and characterization of radiographic-observer performance. *Journal of Applied Photographic Engineering* 4(4), 166–171 (1978).
7. BraTS 2023 lesion-wise metrics code (Saluja R., et al.). https://github.com/rachitsaluja/BraTS-2023-Metrics (inspected Sept 2026).
8. Carass A, Roy S, Jog A, et al. Longitudinal multiple sclerosis lesion segmentation: Resource and challenge. *NeuroImage* 148, 77–102 (2017). https://doi.org/10.1016/j.neuroimage.2016.12.064
9. Cárdenes R, de Luis-García R, Bach-Cuadra M. A multidimensional segmentation evaluation for medical image data. *Computer Methods and Programs in Biomedicine* 96(2), 108–124 (2009).
10. Chakraborty DP, Berbaum KS. Observer studies involving detection and localization: modeling, analysis, and validation. *Medical Physics* 31(8), 2313–2330 (2004). https://doi.org/10.1118/1.1769352
11. Cheng B, Girshick R, Dollár P, Berg AC, Kirillov A. Boundary IoU: Improving object-centric image segmentation evaluation. *CVPR* 2021, 15334–15342. https://arxiv.org/abs/2103.16562
12. Chicco D, Jurman G. The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation. *BMC Genomics* 21, 6 (2020). https://doi.org/10.1186/s12864-019-6413-7
13. Cohen J. A coefficient of agreement for nominal scales. *Educational and Psychological Measurement* 20(1), 37–46 (1960). https://doi.org/10.1177/001316446002000104
14. Commowick O, Istace A, Kain M, et al. Objective evaluation of multiple sclerosis lesion segmentation using a data management and processing infrastructure. *Scientific Reports* 8, 13650 (2018). https://doi.org/10.1038/s41598-018-31911-7
15. Cover TM, Thomas JA. *Elements of Information Theory*, 2nd ed. Wiley (2006).
16. Crum WR, Camara O, Hill DLG. Generalized overlap measures for evaluation and validation in medical image analysis. *IEEE Transactions on Medical Imaging* 25(11), 1451–1461 (2006). https://doi.org/10.1109/TMI.2006.880587
17. Demšar J. Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research* 7, 1–30 (2006).
18. Detlefsen NS, Borovec J, Schock J, et al. TorchMetrics – Measuring reproducibility in PyTorch. *Journal of Open Source Software* 7(70), 4101 (2022). https://doi.org/10.21105/joss.04101
19. Dice LR. Measures of the amount of ecologic association between species. *Ecology* 26(3), 297–302 (1945). https://doi.org/10.2307/1932409
20. Efron B. Bootstrap methods: Another look at the jackknife. *Annals of Statistics* 7(1), 1–26 (1979). https://doi.org/10.1214/aos/1176344552
21. Gerig G, Jomier M, Chakos M. Valmet: A new validation tool for assessing and improving 3D object segmentation. *MICCAI 2001*, LNCS 2208, 516–523 (2001).
22. Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. *ICML 2017*, PMLR 70, 1321–1330. https://arxiv.org/abs/1706.04599
23. Heimann T, van Ginneken B, Styner MA, et al. Comparison and evaluation of methods for liver segmentation from CT datasets. *IEEE Transactions on Medical Imaging* 28(8), 1251–1265 (2009). https://doi.org/10.1109/TMI.2009.2013851
24. Holm S. A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics* 6(2), 65–70 (1979).
25. Hu X, Li F, Samaras D, Chen C. Topology-preserving deep image segmentation. *NeurIPS 2019* (Advances in Neural Information Processing Systems 32). https://arxiv.org/abs/1906.05404
26. Hubert L, Arabie P. Comparing partitions. *Journal of Classification* 2, 193–218 (1985). https://doi.org/10.1007/BF01908075
27. Huttenlocher DP, Klanderman GA, Rucklidge WJ. Comparing images using the Hausdorff distance. *IEEE Transactions on Pattern Analysis and Machine Intelligence* 15(9), 850–863 (1993). https://doi.org/10.1109/34.232073
28. Jaccard P. The distribution of the flora in the alpine zone. *New Phytologist* 11(2), 37–50 (1912). https://doi.org/10.1111/j.1469-8137.1912.tb05611.x
29. Jia J, Staring M, Stoel BC. seg-metrics: a Python package to compute segmentation metrics. medRxiv / arXiv:2403.07884 (2024). https://arxiv.org/abs/2403.07884
30. Jungo A, Reyes M. Assessing reliability and challenges of uncertainty estimations for medical image segmentation. *MICCAI 2019*, LNCS 11765, 48–56. https://arxiv.org/abs/1907.03338
31. Jungo A, Scheidegger O, Reyes M, Balsiger F. pymia: A Python package for data handling and evaluation in deep learning-based medical image analysis. *Computer Methods and Programs in Biomedicine* 198, 105796 (2021). https://doi.org/10.1016/j.cmpb.2020.105796
32. Kazerooni AF, Khalili N, Liu X, et al. BraTS-PEDs: Results of the multi-consortium international pediatric brain tumor segmentation challenge 2023. arXiv:2407.08855 (2024). https://arxiv.org/abs/2407.08855
33. Kirillov A, He K, Girshick R, Rother C, Dollár P. Panoptic segmentation. *CVPR* 2019, 9404–9413. https://arxiv.org/abs/1801.00868
34. Kofler F, Möller H, Buchner JA, et al. Panoptica – instance-wise evaluation of 3D semantic and instance segmentation maps. arXiv:2312.02608 (2023). https://arxiv.org/abs/2312.02608 ; code: https://github.com/BrainLesion/panoptica
35. Koo TK, Li MY. A guideline of selecting and reporting intraclass correlation coefficients for reliability research. *Journal of Chiropractic Medicine* 15(2), 155–163 (2016). https://doi.org/10.1016/j.jcm.2016.02.012
36. Kull M, Perello-Nieto M, Kängsepp M, et al. Beyond temperature scaling: Obtaining well-calibrated multiclass probabilities with Dirichlet calibration. *NeurIPS 2019*. https://arxiv.org/abs/1910.12656
37. Mahalanobis PC. On the generalised distance in statistics. *Proceedings of the National Institute of Sciences of India* 2(1), 49–55 (1936).
38. Maier-Hein L, Eisenmann M, Reinke A, et al. Why rankings of biomedical image analysis competitions should be interpreted with care. *Nature Communications* 9, 5217 (2018). https://doi.org/10.1038/s41467-018-07619-7
39. Maier-Hein L, Reinke A, Godau P, et al. Metrics reloaded: recommendations for image analysis validation. *Nature Methods* 21(2), 195–212 (2024). https://doi.org/10.1038/s41592-023-02151-z (PMC11182665; arXiv:2206.01653)
40. Martin D, Fowlkes C, Tal D, Malik J. A database of human segmented natural images and its application to evaluating segmentation algorithms and measuring ecological statistics. *ICCV 2001*, vol. 2, 416–423. https://doi.org/10.1109/ICCV.2001.937655
41. Matthews BW. Comparison of the predicted and observed secondary structure of T4 phage lysozyme. *Biochimica et Biophysica Acta – Protein Structure* 405(2), 442–451 (1975). https://doi.org/10.1016/0005-2795(75)90109-9
42. Mehrtash A, Wells WM, Tempany CM, Abolmaesumi P, Kapur T. Confidence calibration and predictive uncertainty estimation for deep medical image segmentation. *IEEE Transactions on Medical Imaging* 39(12), 3868–3878 (2020). https://doi.org/10.1109/TMI.2020.3006437
43. Meilă M. Comparing clusterings—an information based distance. *Journal of Multivariate Analysis* 98(5), 873–895 (2007). https://doi.org/10.1016/j.jmva.2006.11.013
44. Moawad AW, Janas A, Baid U, et al. The Brain Tumor Segmentation (BraTS-METS) Challenge 2023: Brain metastasis segmentation on pre-treatment MRI. arXiv:2306.00838 (2023). https://arxiv.org/abs/2306.00838
45. Naeini MP, Cooper GF, Hauskrecht M. Obtaining well calibrated probabilities using Bayesian binning. *Proceedings of the AAAI Conference on Artificial Intelligence* 29 (2015).
46. Nikolov S, Blackwell S, Zverovitch A, et al. Clinically applicable segmentation of head and neck anatomy for radiotherapy: Deep learning algorithm development and validation study. *Journal of Medical Internet Research* 23(7), e26151 (2021). https://doi.org/10.2196/26151 ; code: https://github.com/google-deepmind/surface-distance
47. Nixon J, Dusenberry MW, Zhang L, Jerfel G, Tran D. Measuring calibration in deep learning. *CVPR Workshops* 2019. https://arxiv.org/abs/1904.01685
48. Nunez-Iglesias J, Kennedy R, Parag T, Shi J, Chklovskii DB. Machine learning of hierarchical clustering to segment 2D and 3D images. *PLoS ONE* 8(8), e71715 (2013). https://doi.org/10.1371/journal.pone.0071715
49. Rand WM. Objective criteria for the evaluation of clustering methods. *Journal of the American Statistical Association* 66(336), 846–850 (1971). https://doi.org/10.1080/01621459.1971.10482356
50. Reinke A, Tizabi MD, Baumgartner M, et al. Understanding metric-related pitfalls in image analysis validation. *Nature Methods* 21(2), 182–194 (2024). https://doi.org/10.1038/s41592-023-02150-0 (PMC11181963)
51. Salehi SSM, Erdogmus D, Gholipour A. Tversky loss function for image segmentation using 3D fully convolutional deep networks. *MLMI 2017*, LNCS 10541, 379–387. https://arxiv.org/abs/1706.05721
52. Setio AAA, Traverso A, de Bel T, et al. Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in computed tomography images: The LUNA16 challenge. *Medical Image Analysis* 42, 1–13 (2017). https://doi.org/10.1016/j.media.2017.06.015
53. Shit S, Paetzold JC, Sekuboyina A, et al. clDice – A novel topology-preserving loss function for tubular structure segmentation. *CVPR* 2021, 16560–16569. https://arxiv.org/abs/2003.07311
54. Shrout PE, Fleiss JL. Intraclass correlations: Uses in assessing rater reliability. *Psychological Bulletin* 86(2), 420–428 (1979). https://doi.org/10.1037/0033-2909.86.2.420
55. Sørensen T. A method of establishing groups of equal amplitude in plant sociology based on similarity of species content. *Kongelige Danske Videnskabernes Selskab, Biologiske Skrifter* 5(4), 1–34 (1948).
56. Stucki N, Paetzold JC, Shit S, Menze B, Bauer U. Topologically faithful image segmentation via induced matching of persistence barcodes. *ICML 2023*, PMLR 202. https://proceedings.mlr.press/v202/stucki23a.html ; code: https://github.com/nstucki/Betti-matching
57. Sudre CH, Li W, Vercauteren T, Ourselin S, Cardoso MJ. Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations. *DLMIA 2017*, LNCS 10553, 240–248. https://arxiv.org/abs/1707.03237
58. Taha AA, Hanbury A. Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool. *BMC Medical Imaging* 15, 29 (2015). https://doi.org/10.1186/s12880-015-0068-x ; tool: https://github.com/Visceral-Project/EvaluateSegmentation
59. Tversky A. Features of similarity. *Psychological Review* 84(4), 327–352 (1977). https://doi.org/10.1037/0033-295X.84.4.327
60. Wiesenfarth M, Reinke A, Landman BA, et al. Methods and open-source toolkit for analyzing and visualizing challenge results. *Scientific Reports* 11, 2369 (2021). https://doi.org/10.1038/s41598-021-82017-6 ; code: https://github.com/wiesenfa/challengeR
61. Wilcoxon F. Individual comparisons by ranking methods. *Biometrics Bulletin* 1(6), 80–83 (1945). https://doi.org/10.2307/3001968
62. Yeghiazaryan V, Voiculescu I. Family of boundary overlap metrics for the evaluation of medical image segmentation. *Journal of Medical Imaging* 5(1), 015006 (2018). https://doi.org/10.1117/1.JMI.5.1.015006

**Software (inspected source, Sept 2026):**

- MetricsReloaded: https://github.com/Project-MONAI/MetricsReloaded
- MONAI (Cardoso MJ et al., arXiv:2211.02701, 2022): https://github.com/Project-MONAI/MONAI
- MedPy: https://github.com/loli/medpy
- pymia: https://github.com/rundherum/pymia
- torchmetrics: https://github.com/Lightning-AI/torchmetrics
- seg-metrics: https://github.com/Jingnan-Jia/segmentation_metrics
