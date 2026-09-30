# Distance & boundary metrics

Distance metrics ask **where** the errors are and **how far** they reach, in millimetres. They complement
overlap metrics, which count wrong voxels but do not see whether a wrong voxel sits on the rim or 5 cm away.
The reverse also holds: distance metrics say little about how much volume is wrong. Metrics Reloaded therefore
recommends reporting **one overlap metric and one boundary metric** for every segmentation task.

All distances use the physical voxel spacing. On anisotropic MR (for example 0.8 × 0.8 × 5 mm) a one-slice error
is 5 mm, not "1"; ignoring the spacing gives errors of several-fold.

## Conventions

SegEvalKit fixes the conventions that differ silently between tools:

Surface voxels
:   \(\partial X\) is the set of foreground voxels of \(X\) with at least one **6-connected** (face) background
    neighbour. The image border counts as background, so an object touching the border still has a surface
    there. Each surface voxel has unit weight (voxel counting).

Directed distances
:   For every surface voxel of one mask, the exact Euclidean distance in mm (spacing applied) to the nearest
    surface voxel of the other mask. On the CPU this is an exact Euclidean distance transform
    (`scipy.ndimage.distance_transform_edt`); on a GPU it is an exact chunked nearest-neighbour search. Both give
    the same values to about \(10^{-6}\) mm.

Percentile Hausdorff
:   `hd95` is the maximum of the two directed 95th percentiles (MetricsReloaded, MONAI, DeepMind
    `surface-distance`, BraTS). `hd_percentile` with `mode="pooled"` gives the MedPy convention (percentile of the
    pooled distances), which is never larger than the directed form.

ASSD vs MASD
:   `assd` is the mean over the **pooled** distances of both surfaces (MONAI; MedPy ≥ 0.5.2). `masd` is the mean
    of the **two directed means** (MetricsReloaded's MASD; MedPy ≤ 0.5.1 called this "assd").

NSD
:   Voxel counting, as in MONAI and the FLARE / PanTS evaluation code. DeepMind's reference implementation weights
    each surface element by its area; the two agree closely on smooth, well-sampled surfaces and can differ on
    strongly anisotropic grids.

Empty masks
:   Both masks empty: every distance is 0 and NSD / Boundary IoU are 1 under `EmptyPolicy(both_empty="best")`,
    or NaN under `"nan"`. Exactly one mask empty: HD, HD\(_q\), HD95, ASSD, MASD and centroid distance return the
    `one_empty_distance` penalty: the image diagonal in mm (`"worst"`, the default), NaN (`"nan"`), or a fixed
    number (374 mm under the `brats2023` preset). NSD and Boundary IoU return 0. See
    [empty masks](index.md#empty-masks).

## Notation

With \(d(x, S) = \min_{y\in S}\lVert x-y\rVert_2\) the distance in mm from a voxel centre to a set of voxel
centres, the two directed distance sets are

\[
D_{P\to G} = \{\, d(p, \partial G) : p\in\partial P \,\},\qquad
D_{G\to P} = \{\, d(g, \partial P) : g\in\partial G \,\}.
\]

\(D_{P\to G}\) measures how far the prediction strays from the reference (over-segmentation, false positives);
\(D_{G\to P}\) how much of the reference the prediction fails to reach (under-segmentation). Every metric on this
page except Boundary IoU and centroid distance is a summary of these two sets. \(P_q(\cdot)\) denotes the
\(q\)-th percentile with linear interpolation.

## At a glance

| Metric | Key | Summary of \(D_{P\to G}, D_{G\to P}\) | Unit | Outlier sensitivity | Parameter |
|---|---|---|---|---|---|
| [HD](#hd) | `hd` | maximum | mm | extreme: one voxel sets it | – |
| [HD\(_q\)](#hd_percentile) | `hd_percentile` | \(q\)-th percentile | mm | moderate | `q`, `mode` |
| [HD95](#hd95) | `hd95` | max of directed 95th percentiles | mm | moderate | – |
| [ASSD](#assd) | `assd` | pooled mean | mm | low | – |
| [MASD](#masd) | `masd` | mean of directed means | mm | low | – |
| [NSD](#nsd) | `nsd` | fraction \(\le \tau\) | – | none beyond \(\tau\) | `tolerance_mm` |
| [Boundary IoU](#boundary_iou) | `boundary_iou` | IoU of inner bands | – | low | `width_mm` |
| [Centroid distance](#centroid_distance) | `centroid_distance` | centres of mass | mm | low (shape-blind) | – |

## Metrics

<div class="sek-metric" markdown>

### Hausdorff distance (HD) {#hd}

<div class="sek-meta"><span class="sek-chip">key: <code>hd</code></span><span class="sek-chip">aliases: <code>hausdorff</code>, <code>hd100</code></span><span class="sek-chip">range [0, ∞)</span><span class="sek-chip teal">lower is better</span><span class="sek-chip orange">mm</span><span class="sek-chip">needs spacing</span></div>

\[
\mathrm{HD} = \max\Big(\max D_{P\to G},\; \max D_{G\to P}\Big)
\]

**In plain words:** the single worst boundary error: the largest distance from any surface point of one mask to
the nearest surface point of the other.

**Use it when:** a single far-off error matters clinically, for example as a safety bound for radiotherapy
organs at risk, and always next to a robust summary.

**Watch out for:** *spatial outliers* (Reinke et al. 2024, category P2). One stray false-positive voxel far from
the object sets the value: adding a single voxel about 54 mm from a perfect prediction of a 20 mm cube moves HD from 0 to 53.7 mm while Dice stays at 0.9999
(see [Pitfalls](../guide/pitfalls.md#outliers)). It depends on resolution and is not comparable across structures
of different size.

**Empty masks:** both empty: 0 (`"best"`) or NaN. Exactly one empty: the `one_empty_distance` penalty (image
diagonal in mm by default).

**Parameters:** none.

**Reference:** Huttenlocher DP, Klanderman GA, Rucklidge WJ. Comparing images using the Hausdorff distance.
*IEEE TPAMI* 15(9), 850–863 (1993). [doi:10.1109/34.232073](https://doi.org/10.1109/34.232073)

</div>

<div class="sek-metric" markdown>

### Percentile Hausdorff distance (HDq) {#hd_percentile}

<div class="sek-meta"><span class="sek-chip">key: <code>hd_percentile</code></span><span class="sek-chip">range [0, ∞)</span><span class="sek-chip teal">lower is better</span><span class="sek-chip orange">mm</span><span class="sek-chip">needs spacing</span></div>

\[
\mathrm{HD}_q = \max\Big(P_q(D_{P\to G}),\; P_q(D_{G\to P})\Big)
\qquad\text{or, with } \texttt{mode="pooled"},\qquad
\mathrm{HD}_q^{\text{pooled}} = P_q\big(D_{P\to G}\cup D_{G\to P}\big)
\]

**In plain words:** a near-worst-case boundary error that ignores the most extreme \((100-q)\) % of surface
points.

**Use it when:** you need a percentile other than 95, or you must reproduce numbers from MedPy (`mode="pooled"`).

**Watch out for:** the two modes give different numbers; always state `q` and `mode`. The directed form (default)
matches MetricsReloaded, MONAI, DeepMind and BraTS. For small structures with few surface voxels, any
percentile approaches the maximum.

**Empty masks:** as [HD](#hd).

**Parameters:** `q = 95.0`, `mode = "directed"` (or `"pooled"`).

**Reference:** Huttenlocher et al. 1993, [doi:10.1109/34.232073](https://doi.org/10.1109/34.232073); directed
convention of DeepMind `surface-distance` ([code](https://github.com/google-deepmind/surface-distance)) and
[MONAI](https://github.com/Project-MONAI/MONAI).

</div>

<div class="sek-metric" markdown>

### 95th-percentile Hausdorff distance (HD95) {#hd95}

<div class="sek-meta"><span class="sek-chip">key: <code>hd95</code></span><span class="sek-chip">range [0, ∞)</span><span class="sek-chip teal">lower is better</span><span class="sek-chip orange">mm</span><span class="sek-chip">needs spacing</span></div>

\[
\mathrm{HD}_{95} = \max\Big(P_{95}(D_{P\to G}),\; P_{95}(D_{G\to P})\Big)
\]

**In plain words:** 95 % of the surface points of each mask lie within this distance of the other mask's surface.

**Use it when:** you need a robust worst-case boundary error; it is the boundary metric of BraTS, KiTS, TopCoW and
many radiotherapy auto-contouring studies, and part of the `default` metric set.

**Watch out for:** it is still sensitive to clusters of outliers larger than 5 % of the surface, and it tells
nothing about how much of the surface is acceptable (use [NSD](#nsd)). `hd95` always uses the directed
convention and takes no parameters; for the MedPy (pooled) convention use
`hd_percentile` with `{"q": 95, "mode": "pooled"}`.

**Empty masks:** as [HD](#hd). Under the `brats2023` preset, one empty mask gives 374 mm.

**Parameters:** none (fixed \(q = 95\), directed).

**Reference:** Huttenlocher et al. 1993, [doi:10.1109/34.232073](https://doi.org/10.1109/34.232073). Bakas S,
Reyes M, Jakab A, et al. Identifying the best machine learning algorithms for brain tumor segmentation,
progression assessment, and overall survival prediction in the BRATS challenge.
[arXiv:1811.02629](https://arxiv.org/abs/1811.02629) (2018).

</div>

<div class="sek-metric" markdown>

### Average symmetric surface distance (ASSD) {#assd}

<div class="sek-meta"><span class="sek-chip">key: <code>assd</code></span><span class="sek-chip">alias: <code>asd</code></span><span class="sek-chip">range [0, ∞)</span><span class="sek-chip teal">lower is better</span><span class="sek-chip orange">mm</span><span class="sek-chip">needs spacing</span></div>

\[
\mathrm{ASSD} = \frac{\sum D_{P\to G} + \sum D_{G\to P}}{\lvert\partial P\rvert + \lvert\partial G\rvert}
\]

**In plain words:** the typical boundary error: the mean distance over all surface points of both masks.

**Use it when:** you want the average contour accuracy in mm (SLIVER07 and other liver benchmarks). It is part of
the `default` metric set.

**Watch out for:** averaging hides localised large errors. The larger surface dominates the mean; use
[MASD](#masd) if both surfaces should weigh equally. Names are inconsistent across tools (MedPy's `asd` is a
*directed* mean), so always state the formula.

**Empty masks:** as [HD](#hd).

**Parameters:** none.

**Reference:** Heimann T, van Ginneken B, Styner MA, et al. Comparison and evaluation of methods for liver
segmentation from CT datasets. *IEEE TMI* 28(8), 1251–1265 (2009).
[doi:10.1109/TMI.2009.2013851](https://doi.org/10.1109/TMI.2009.2013851). Yeghiazaryan V, Voiculescu I. Family
of boundary overlap metrics for the evaluation of medical image segmentation. *J Med Imaging* 5(1), 015006
(2018). [doi:10.1117/1.JMI.5.1.015006](https://doi.org/10.1117/1.JMI.5.1.015006)

</div>

<div class="sek-metric" markdown>

### Mean average surface distance (MASD) {#masd}

<div class="sek-meta"><span class="sek-chip">key: <code>masd</code></span><span class="sek-chip">range [0, ∞)</span><span class="sek-chip teal">lower is better</span><span class="sek-chip orange">mm</span><span class="sek-chip">needs spacing</span></div>

\[
\mathrm{MASD} = \tfrac{1}{2}\big(\overline{D_{P\to G}} + \overline{D_{G\to P}}\big)
= \frac12\left(\frac{\sum D_{P\to G}}{\lvert\partial P\rvert} + \frac{\sum D_{G\to P}}{\lvert\partial G\rvert}\right)
\]

**In plain words:** like ASSD, but each surface gets equal weight regardless of its size.

**Use it when:** prediction and reference surfaces can differ a lot in size (small structures, severe under- or
over-segmentation) and you do not want the larger one to dominate.

**Watch out for:** it is easily confused with ASSD. The two are equal only when \(\lvert\partial P\rvert =
\lvert\partial G\rvert\). For a 10-voxel cube predicted inside a 20-voxel reference cube, ASSD = 5.63 mm and
MASD = 5.39 mm.

**Empty masks:** as [HD](#hd).

**Parameters:** none.

**Reference:** Yeghiazaryan & Voiculescu 2018,
[doi:10.1117/1.JMI.5.1.015006](https://doi.org/10.1117/1.JMI.5.1.015006). Maier-Hein L, Reinke A, Godau P, et
al. Metrics reloaded: recommendations for image analysis validation. *Nature Methods* 21, 195–212 (2024).
[doi:10.1038/s41592-023-02151-z](https://doi.org/10.1038/s41592-023-02151-z)

</div>

<div class="sek-metric" markdown>

### Normalized surface Dice (NSD) {#nsd}

<div class="sek-meta"><span class="sek-chip">key: <code>nsd</code></span><span class="sek-chip">aliases: <code>surface_dice</code>, <code>normalized_surface_dice</code></span><span class="sek-chip">range [0, 1]</span><span class="sek-chip teal">higher is better</span><span class="sek-chip orange">dimensionless</span><span class="sek-chip">needs spacing</span></div>

\[
\mathrm{NSD}_\tau = \frac{\lvert\{d\in D_{P\to G}: d\le\tau\}\rvert + \lvert\{d\in D_{G\to P}: d\le\tau\}\rvert}{\lvert\partial P\rvert + \lvert\partial G\rvert}
\]

\(\tau\) is the tolerance in mm (`tolerance_mm`).

**In plain words:** the fraction of both surfaces that lies within a clinically acceptable distance \(\tau\) of
the other surface, i.e. the part of the contour that would not need manual correction.

**Use it when:** boundaries matter and annotation imprecision should be forgiven. Metrics Reloaded makes NSD the
default boundary metric in that case and recommends pairing it with Dice. It is part of the `default` metric
set.

**Watch out for:** the result depends strongly on \(\tau\). For a 24-voxel cube shifted by 2 voxels at 1 mm spacing, NSD is 0.64 at
\(\tau = 0.5\) mm, 0.69 at 1 mm and 1.00 at 2 mm (see [Pitfalls](../guide/pitfalls.md#nsd-tolerance)). Choose
\(\tau\) per structure from inter-rater variability or a clinical margin, fix it before evaluation, and report it.
Errors beyond \(\tau\) are not graded by size. The voxel-counting form can differ from DeepMind's area-weighted
form on anisotropic grids.

**Empty masks:** both empty: 1 (`"best"`) or NaN. Exactly one empty: 0 (independent of `one_empty_distance`).

**Parameters:** `tolerance_mm = 2.0`.

**Reference:** Nikolov S, Blackwell S, Zverovitch A, et al. Clinically applicable segmentation of head and neck
anatomy for radiotherapy: deep learning algorithm development and validation study. *J Med Internet Res* 23(7),
e26151 (2021). [doi:10.2196/26151](https://doi.org/10.2196/26151)

</div>

<div class="sek-metric" markdown>

### Boundary IoU (BIoU) {#boundary_iou}

<div class="sek-meta"><span class="sek-chip">key: <code>boundary_iou</code></span><span class="sek-chip">range [0, 1]</span><span class="sek-chip teal">higher is better</span><span class="sek-chip orange">dimensionless</span><span class="sek-chip">needs spacing</span></div>

\[
\mathrm{BIoU}_d = \frac{\lvert (P_d\cap P)\cap(G_d\cap G)\rvert}{\lvert (P_d\cap P)\cup(G_d\cap G)\rvert}
\]

\(X_d \cap X\) is the inner boundary band of \(X\): the foreground voxels whose Euclidean distance (mm) to the
nearest background voxel is at most \(d\) (`width_mm`). At 1 mm isotropic spacing, \(d = 2\) mm is the two
outermost voxel layers.

**In plain words:** IoU measured only in a thin band along each boundary, so it rewards precise contours rather
than bulk overlap.

**Use it when:** comparing boundary quality of large objects, where Dice saturates near 1. It is less sensitive
to outliers than HD.

**Watch out for:** it depends on the band width. For objects thinner than about \(2d\) it approaches IoU. Choose
`width_mm` at least as large as the **largest** voxel spacing: a surface voxel whose only background neighbour
lies along a 5 mm axis is 5 mm from the background, and with `width_mm` below the smallest spacing both bands
are empty and SegEvalKit returns 1.0 whatever the masks.

**Empty masks:** both empty: 1 (`"best"`) or NaN. Exactly one empty: 0.

**Parameters:** `width_mm = 2.0`.

**Reference:** Cheng B, Girshick R, Dollár P, Berg AC, Kirillov A. Boundary IoU: improving object-centric image
segmentation evaluation. *CVPR 2021*, 15334–15342. [arXiv:2103.16562](https://arxiv.org/abs/2103.16562)

</div>

<div class="sek-metric" markdown>

### Centroid distance (CD) {#centroid_distance}

<div class="sek-meta"><span class="sek-chip">key: <code>centroid_distance</code></span><span class="sek-chip">range [0, ∞)</span><span class="sek-chip teal">lower is better</span><span class="sek-chip orange">mm</span><span class="sek-chip">needs spacing</span></div>

\[
\mathrm{CD} = \lVert \bar{x}_P - \bar{x}_G \rVert_2, \qquad \bar{x}_X = \frac{1}{\lvert X\rvert}\sum_{x\in X} x
\]

with voxel coordinates scaled by the spacing (mm).

**In plain words:** how far apart the centres of mass of the two masks are.

**Use it when:** localisation is the question (is the structure in the right place?), or as a localisation
criterion for small lesions, which Metrics Reloaded lists among its matching criteria.

**Watch out for:** it is blind to shape and size: a sphere and a thin shell with the same centre score 0.
Concave and multi-component objects can have a centroid outside the object.

**Empty masks:** as [HD](#hd).

**Parameters:** none.

**Reference:** Reinke A, Tizabi MD, Baumgartner M, et al. Understanding metric-related pitfalls in image analysis
validation. *Nature Methods* 21, 182–194 (2024).
[doi:10.1038/s41592-023-02150-0](https://doi.org/10.1038/s41592-023-02150-0)

</div>

## Code

The `distance` metric set contains `hd`, `hd95`, `assd`, `masd`, `nsd` and `boundary_iou`. Distances are computed
once per pair and shared by all of them.

```python
import numpy as np
from segevalkit.metrics import compute_metrics

ref = np.zeros((40, 40, 40), bool)
ref[8:32, 8:32, 8:32] = True
pred = np.roll(ref, 2, axis=0)           # 2-voxel shift along the first axis
pred[36, 36, 36] = True                  # plus one stray false-positive voxel

scores = compute_metrics(
    pred, ref,
    metrics=["distance", "hd_percentile", "centroid_distance"],
    spacing=(1.0, 1.0, 1.0),
    params={
        "nsd": {"tolerance_mm": 1.0},
        "hd_percentile": {"q": 95, "mode": "pooled"},   # MedPy convention
        "boundary_iou": {"width_mm": 2.0},
    },
)
for name, value in scores.items():
    print(f"{name:18s} {value:.4f}")
```

```text
hd                 8.6603
hd95               2.0000
assd               0.6713
masd               0.6713
nsd                0.6926
boundary_iou       0.4979
hd_percentile      2.0000
centroid_distance  2.0010
```

The stray voxel sets HD (8.66 mm) but leaves HD95 at the 2 mm shift. To use a GPU for the surface distances,
pass `device="cuda"`; the values are the same.

!!! tip "Empty-mask penalty"
    Distances for a missed or hallucinated structure follow the `EmptyPolicy`. To reproduce BraTS 2023, pass
    `empty=EmptyPolicy.preset("brats2023")` (374 mm). To exclude such cases from aggregates instead, use
    `EmptyPolicy("best", "nan")`, and report how many cases were excluded.
