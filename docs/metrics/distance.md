# Distance & boundary metrics

Distance metrics ask **where** the errors are and **how far** they reach, in millimetres; overlap metrics count
wrong voxels but cannot tell a rim error from one 5 cm away, and distances say little about wrong volume. Metrics
Reloaded therefore recommends **one overlap plus one boundary metric** per task. All distances use the physical
spacing: on 0.8 × 0.8 × 5 mm MR a one-slice error is 5 mm.

## At a glance

| Metric | Key | Summary of \(D_{P\to G}, D_{G\to P}\) | Unit | Outlier sensitivity | Parameter |
|---|---|---|---|---|---|
| [HD](#hd) | `hd` | maximum | mm | extreme: one voxel sets it | – |
| [HD\(_q\)](#hd_percentile) | `hd_percentile` | \(q\)-th percentile | mm | moderate | `q`, `mode` |
| [HD95](#hd95) | `hd95` | 95th percentile | mm | moderate | `mode` |
| [ASSD](#assd) | `assd` | pooled mean | mm | low | – |
| [MASD](#masd) | `masd` | mean of directed means | mm | low | – |
| [NSD](#nsd) | `nsd` | fraction \(\le \tau\) | – | none beyond \(\tau\) | `tolerance_mm` |
| [Boundary IoU](#boundary_iou) | `boundary_iou` | IoU of boundary bands | – | low | `width_mm` |
| [Centroid distance](#centroid_distance) | `centroid_distance` | centres of mass | mm | low (shape-blind) | – |

## Notation

\(\partial X\) is the **surface** of mask \(X\): foreground voxels with at least one 6-connected background
neighbour (the image border counts as background), each with unit weight. With
\(d(x, S) = \min_{y\in S}\lVert x-y\rVert_2\) in mm, the two directed distance sets are

\[
D_{P\to G} = \{\, d(p, \partial G) : p\in\partial P \,\},\qquad
D_{G\to P} = \{\, d(g, \partial P) : g\in\partial G \,\}.
\]

\(D_{P\to G}\) measures over-segmentation (prediction straying from the reference), \(D_{G\to P}\)
under-segmentation. \(P_q(\cdot)\) is the \(q\)-th percentile (linear interpolation). Distances are exact
Euclidean: an EDT on CPU (`scipy.ndimage.distance_transform_edt`), a chunked nearest-neighbour search on GPU
(`device="cuda"`), equal to about \(10^{-6}\) mm.

## Conventions

| Convention | SegEvalKit | Other tools |
|---|---|---|
| HD95 / HD\(_q\) | max of the two directed percentiles (MetricsReloaded, MONAI, DeepMind, BraTS) | `mode="pooled"`: percentile of pooled distances (MedPy); never larger |
| ASSD vs MASD | `assd`: mean of pooled distances (MONAI; MedPy ≥ 0.5.2); `masd`: mean of the two directed means (MetricsReloaded) | MedPy ≤ 0.5.1 called MASD "assd" |
| NSD surface weight | voxel counting (MONAI, FLARE / PanTS) | DeepMind weights by surface area; differs on anisotropic grids |
| One mask empty | distances = `one_empty_distance`: image diagonal (`"worst"`, default), NaN, or a number; NSD, BIoU = 0 | BraTS 2023: 374 mm (`EmptyPolicy.preset("brats2023")`) |
| Both masks empty | distances 0, NSD / BIoU 1 (`"best"`), or NaN (`"nan"`) | see [empty masks](index.md#empty-masks) |

## Metrics

<div class="sek-metric" markdown>

### Hausdorff distance (HD) {#hd}

<div class="sek-meta"><span class="sek-chip">hd · hausdorff · hd100</span><span class="sek-chip">[0, ∞)</span><span class="sek-chip teal">↓ lower is better</span><span class="sek-chip orange">mm</span></div>

\[
\mathrm{HD} = \max\Big(\max D_{P\to G},\; \max D_{G\to P}\Big)
\]

**In words** The single worst boundary error between the two surfaces.

**Use when** One far-off error matters clinically (a safety bound for organs at risk), always next to a robust summary.

**Watch out** One stray voxel sets it ([outliers](../guide/pitfalls.md#outliers)); resolution-dependent and not comparable across structure sizes.

??? info "Details: empty masks, parameters, reference"
    **Example.** One voxel about 54 mm from a perfect 20 mm cube moves HD from 0 to 53.7 mm while Dice stays 0.9999 (Reinke et al. 2024, P2).

    **Empty masks.** Both empty: 0 (`"best"`) or NaN. One empty: `one_empty_distance` (image diagonal by default).

    **Reference.** Huttenlocher DP et al. *IEEE TPAMI* 15(9):850–863, 1993. [doi:10.1109/34.232073](https://doi.org/10.1109/34.232073)

</div>

<div class="sek-metric" markdown>

### Percentile Hausdorff distance (HDq) {#hd_percentile}

<div class="sek-meta"><span class="sek-chip">hd_percentile</span><span class="sek-chip">[0, ∞)</span><span class="sek-chip teal">↓ lower is better</span><span class="sek-chip orange">mm</span></div>

\[
\begin{aligned}
\mathrm{HD}_q &= \max\Big(P_q(D_{P\to G}),\; P_q(D_{G\to P})\Big)
\\
\mathrm{HD}_q^{\text{pooled}} &= P_q\big(D_{P\to G}\cup D_{G\to P}\big)\quad(\texttt{mode="pooled"})
\end{aligned}
\]

**In words** A near-worst boundary error that ignores the most extreme \((100-q)\) % of surface points.

**Use when** You need a percentile other than 95; for 95, `hd95` is equivalent and takes the same `mode`.

**Watch out** The two modes differ, so state `q` and `mode`; on small structures any percentile approaches the maximum.

??? info "Details: empty masks, parameters, reference"
    **Parameters.** `q = 95.0`, `mode = "directed"` (or `"pooled"`). **Empty masks.** As [HD](#hd).

    **Reference.** Huttenlocher et al. 1993, [doi:10.1109/34.232073](https://doi.org/10.1109/34.232073); directed convention of DeepMind [`surface-distance`](https://github.com/google-deepmind/surface-distance) and [MONAI](https://github.com/Project-MONAI/MONAI).

</div>

<div class="sek-metric" markdown>

### 95th-percentile Hausdorff distance (HD95) {#hd95}

<div class="sek-meta"><span class="sek-chip">hd95</span><span class="sek-chip">[0, ∞)</span><span class="sek-chip teal">↓ lower is better</span><span class="sek-chip orange">mm</span></div>

\[
\begin{aligned}
\mathrm{HD}_{95} &= \max\Big(P_{95}(D_{P\to G}),\; P_{95}(D_{G\to P})\Big)
\\
\mathrm{HD}_{95}^{\text{pooled}} &= P_{95}\big(D_{P\to G}\cup D_{G\to P}\big)\quad(\texttt{mode="pooled"})
\end{aligned}
\]

**In words** 95 % of each mask's surface lies within this distance of the other surface.

**Use when** You need a robust worst-case boundary error: BraTS, KiTS, TopCoW, radiotherapy contouring; in the `default` set.

**Watch out** Still sensitive to outlier clusters above 5 % of the surface, and silent on how much surface is acceptable ([NSD](#nsd)).

??? info "Details: empty masks, parameters, reference"
    **Parameters.** `mode = "directed"` (default) or `"pooled"`; `{"hd95": {"mode": "pooled"}}` reproduces MedPy. They differ (7.14 mm vs 7.07 mm for a 10-voxel cube inside a 20-voxel cube), so state the mode.

    **Empty masks.** As [HD](#hd); 374 mm for one empty mask under the `brats2023` preset.

    **Reference.** Huttenlocher et al. 1993, [doi:10.1109/34.232073](https://doi.org/10.1109/34.232073). Bakas S et al. BRATS challenge, 2018. [arXiv:1811.02629](https://arxiv.org/abs/1811.02629)

</div>

<div class="sek-metric" markdown>

### Average symmetric surface distance (ASSD) {#assd}

<div class="sek-meta"><span class="sek-chip">assd · asd</span><span class="sek-chip">[0, ∞)</span><span class="sek-chip teal">↓ lower is better</span><span class="sek-chip orange">mm</span></div>

\[
\mathrm{ASSD} = \frac{\sum D_{P\to G} + \sum D_{G\to P}}{\lvert\partial P\rvert + \lvert\partial G\rvert}
\]

**In words** The typical boundary error: the mean distance over all surface points of both masks.

**Use when** You want average contour accuracy in mm (SLIVER07 and other liver benchmarks); in the `default` set.

**Watch out** Averaging hides local large errors, the larger surface dominates ([MASD](#masd) weighs both equally), and tool names vary.

??? info "Details: empty masks, parameters, reference"
    **Naming.** MedPy's `asd` is a *directed* mean; always state the formula. **Empty masks.** As [HD](#hd).

    **Reference.** Heimann T et al. *IEEE TMI* 28(8):1251–1265, 2009. [doi:10.1109/TMI.2009.2013851](https://doi.org/10.1109/TMI.2009.2013851). Yeghiazaryan V, Voiculescu I. *J Med Imaging* 5(1):015006, 2018. [doi:10.1117/1.JMI.5.1.015006](https://doi.org/10.1117/1.JMI.5.1.015006)

</div>

<div class="sek-metric" markdown>

### Mean average surface distance (MASD) {#masd}

<div class="sek-meta"><span class="sek-chip">masd</span><span class="sek-chip">[0, ∞)</span><span class="sek-chip teal">↓ lower is better</span><span class="sek-chip orange">mm</span></div>

\[
\mathrm{MASD} = \tfrac{1}{2}\big(\overline{D_{P\to G}} + \overline{D_{G\to P}}\big)
= \frac12\left(\frac{\sum D_{P\to G}}{\lvert\partial P\rvert} + \frac{\sum D_{G\to P}}{\lvert\partial G\rvert}\right)
\]

**In words** Like ASSD, but each surface gets equal weight regardless of its size.

**Use when** The two surfaces can differ a lot in size (small structures, severe under- or over-segmentation).

**Watch out** Easily confused with ASSD; the two are equal only when \(\lvert\partial P\rvert = \lvert\partial G\rvert\).

??? info "Details: empty masks, parameters, reference"
    **Example.** A 10-voxel cube inside a 20-voxel reference cube: ASSD 5.63 mm, MASD 5.39 mm. **Empty masks.** As [HD](#hd).

    **Reference.** Yeghiazaryan & Voiculescu 2018, [doi:10.1117/1.JMI.5.1.015006](https://doi.org/10.1117/1.JMI.5.1.015006). Maier-Hein L et al. Metrics reloaded. *Nature Methods* 21:195–212, 2024. [doi:10.1038/s41592-023-02151-z](https://doi.org/10.1038/s41592-023-02151-z)

</div>

<div class="sek-metric" markdown>

### Normalized surface Dice (NSD) {#nsd}

<div class="sek-meta"><span class="sek-chip">nsd · surface_dice · normalized_surface_dice</span><span class="sek-chip">[0, 1]</span><span class="sek-chip teal">↑ higher is better</span></div>

\[
\mathrm{NSD}_\tau = \frac{\lvert\{d\in D_{P\to G}: d\le\tau\}\rvert + \lvert\{d\in D_{G\to P}: d\le\tau\}\rvert}{\lvert\partial P\rvert + \lvert\partial G\rvert}
\]

**In words** The fraction of both surfaces within tolerance \(\tau\) mm of the other: the contour needing no manual correction.

**Use when** Boundaries matter but annotation noise should be forgiven; Metrics Reloaded's default boundary metric, paired with Dice; in the `default` set.

**Watch out** Strongly \(\tau\)-dependent ([NSD tolerance](../guide/pitfalls.md#nsd-tolerance)), so fix and report \(\tau\); errors beyond \(\tau\) are not graded by size.

??? info "Details: empty masks, parameters, reference"
    **Parameters.** `tolerance_mm = 2.0`. Choose \(\tau\) per structure from inter-rater variability or a clinical margin, before evaluation.

    **Example.** A 24-voxel cube shifted 2 voxels at 1 mm: NSD 0.64 at \(\tau = 0.5\) mm, 0.69 at 1 mm, 1.00 at 2 mm.

    **Empty masks.** Both empty: 1 (`"best"`) or NaN. One empty: 0 (independent of `one_empty_distance`).

    **Reference.** Nikolov S et al. *J Med Internet Res* 23(7):e26151, 2021. [doi:10.2196/26151](https://doi.org/10.2196/26151)

</div>

<div class="sek-metric" markdown>

### Boundary IoU (BIoU) {#boundary_iou}

<div class="sek-meta"><span class="sek-chip">boundary_iou</span><span class="sek-chip">[0, 1]</span><span class="sek-chip teal">↑ higher is better</span></div>

\[
\mathrm{BIoU}_d = \frac{\lvert (P_d\cap P)\cap(G_d\cap G)\rvert}{\lvert (P_d\cap P)\cup(G_d\cap G)\rvert},
\qquad X_d \cap X = \partial X \,\cup\, \{\, x\in X : \mathrm{EDT}_X(x) \le d \,\}
\]

**In words** IoU measured only in an inner band of width \(d\) along each boundary, rewarding precise contours over bulk overlap.

**Use when** Comparing boundary quality of large objects, where Dice saturates near 1; less outlier-sensitive than HD.

**Watch out** Depends on `width_mm`; objects thinner than about \(2d\) approach plain IoU; state the width with the spacing.

??? info "Details: empty masks, parameters, reference"
    **Band.** \(\mathrm{EDT}_X(x)\) is the distance in mm to the nearest background voxel. The band is never empty for a non-empty mask: at 1 mm spacing \(d = 2\) mm is the two outer layers; for \(d\) below the spacing it is the surface alone. On anisotropic grids it is thicker (in voxels) along the fine axes.

    **Parameters.** `width_mm = 2.0`. **Empty masks.** Both empty: 1 (`"best"`) or NaN. One empty: 0.

    **Reference.** Cheng B et al. Boundary IoU. *CVPR 2021*, 15334–15342. [arXiv:2103.16562](https://arxiv.org/abs/2103.16562)

</div>

<div class="sek-metric" markdown>

### Centroid distance (CD) {#centroid_distance}

<div class="sek-meta"><span class="sek-chip">centroid_distance</span><span class="sek-chip">[0, ∞)</span><span class="sek-chip teal">↓ lower is better</span><span class="sek-chip orange">mm</span></div>

\[
\mathrm{CD} = \lVert \bar{x}_P - \bar{x}_G \rVert_2, \qquad \bar{x}_X = \frac{1}{\lvert X\rvert}\sum_{x\in X} x \ \ (\text{coordinates in mm})
\]

**In words** How far apart the centres of mass of the two masks are.

**Use when** Localisation is the question, or as a small-lesion matching criterion (listed by Metrics Reloaded).

**Watch out** Blind to shape and size (a sphere and a concentric shell score 0); concave objects can have an outside centroid.

??? info "Details: empty masks, parameters, reference"
    **Empty masks.** As [HD](#hd).

    **Reference.** Reinke A et al. Understanding metric-related pitfalls. *Nature Methods* 21:182–194, 2024. [doi:10.1038/s41592-023-02150-0](https://doi.org/10.1038/s41592-023-02150-0)

</div>

## Code

The `distance` set is `hd`, `hd95`, `assd`, `masd`, `nsd`, `boundary_iou`; distances are computed once per pair and
shared. Pass `device="cuda"` for GPU surface distances (same values).

```python
import numpy as np
from segevalkit.metrics import compute_metrics

ref = np.zeros((40, 40, 40), bool); ref[8:32, 8:32, 8:32] = True
pred = np.roll(ref, 2, axis=0); pred[36, 36, 36] = True   # 2-voxel shift + one stray voxel
scores = compute_metrics(pred, ref, ["distance", "hd_percentile", "centroid_distance"],
    spacing=(1.0, 1.0, 1.0),
    params={"nsd": {"tolerance_mm": 1.0}, "boundary_iou": {"width_mm": 2.0},
            "hd_percentile": {"q": 95, "mode": "pooled"}})  # MedPy convention
# hd 8.6603  hd95 2.0000  assd 0.6713  masd 0.6713  nsd 0.6926
# boundary_iou 0.4979  hd_percentile 2.0000  centroid_distance 2.0010
```

The stray voxel sets HD but leaves HD95 at the 2 mm shift. To exclude missed or hallucinated structures from
aggregates instead of penalising them, use `EmptyPolicy("best", "nan")` and report how many were excluded.
