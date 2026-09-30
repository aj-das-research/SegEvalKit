# Conventions & conformance

MONAI, MedPy, DeepMind's `surface-distance` and the BraTS toolkit can return four different HD95 values for the
same prediction. This page lists each implementation choice, the one SegEvalKit makes, and the tests that pin it
down.

## Surfaces and distances

| Choice | SegEvalKit | Alternatives in use |
|---|---|---|
| Surface of a mask | Foreground voxels with ≥ 1 background **face** neighbour (6-connectivity); the image border counts as background | Surfel meshes on a half-voxel grid (DeepMind); 26-connected borders |
| Distance | Exact Euclidean distance between voxel centres, in mm, with the header spacing | Voxel units; approximate chamfer transforms |
| Element weighting | Every surface voxel counts once | Each surface element weighted by its area (DeepMind, KiTS surface Dice) |
| Computation | EDT of the other surface on the union bounding box (CPU) or exact nearest-neighbour search (GPU): identical numbers | |

## Summary statistics of the distances

| Metric | SegEvalKit | Notes |
|---|---|---|
| **HD** | \(\max(\max D_{P\to G}, \max D_{G\to P})\) | Identical in every library |
| **HD95** | \(\max(P_{95}(D_{P\to G}), P_{95}(D_{G\to P}))\) (`mode="directed"`) | MetricsReloaded, MONAI, DeepMind, BraTS. MedPy uses the percentile of the **pooled** distances: `params={"hd95": {"mode": "pooled"}}`. The pooled form is never larger. |
| **ASSD** | Mean of the **pooled** distances \(\big(\sum D_{P\to G} + \sum D_{G\to P}\big) / \big(\lvert\partial P\rvert + \lvert\partial G\rvert\big)\) | MONAI (`symmetric=True`), MedPy ≥ 0.5.2 `assd`. |
| **MASD** | Mean of the two **directed means** \(\tfrac12(\bar D_{P\to G} + \bar D_{G\to P})\) | MetricsReloaded; MedPy ≤ 0.4.0 `assd` (0.5.0 and 0.5.1 raise an error when the surfaces differ in size). Differs from ASSD when the two surfaces have very different sizes. |
| **NSD** | Fraction of surface voxels within τ, voxel counting | MONAI `compute_surface_dice`, FLARE code. DeepMind / KiTS weight by surfel area. |

!!! warning "Name collisions"
    MedPy changed the meaning of `assd`: up to 0.4.0 it is the mean of directed means, in 0.5.2 the pooled mean. In SegEvalKit the names are fixed:
    [`assd`](../metrics/distance.md#assd) is the pooled mean and [`masd`](../metrics/distance.md#masd) the mean
    of directed means. Always report which one you use.

## Empty masks

| Case | SegEvalKit default | Other presets |
|---|---|---|
| Both empty | Ideal value (Dice 1, HD 0, NSD 1) | `nan`: excluded |
| One empty, overlap metrics | 0 | – |
| One empty, distance metrics | Image diagonal in mm | `brats2023`: 374 mm; `nan`: excluded; any number |
| Precision with empty prediction, recall with empty reference | NaN (undefined) | – |

See [Configuration → Empty masks](../getting-started/configuration.md#empty-masks).

## Instances and lesions

| Choice | SegEvalKit | Alternatives |
|---|---|---|
| Component connectivity | 26 (`connectivity=` 6, 18, 26) | 6 or 18 in some challenge code |
| Minimum component size | 0 (`min_lesion_voxels=`) | BraTS 2023 drops lesions ≤ 50 mm³ |
| Matching for detection | Any overlap, many-to-many (`criterion="overlap"`) | IoU > 0.5 one-to-one (`criterion="iou"`); BraTS dilates the reference first |
| Panoptic quality | Hungarian IoU matching, IoU > 0.5 | Kirillov et al. 2019 |

## Failure thresholds

`plotting.failure_quadrants` flags a case when it is **worse than a uniform boundary error of `error_mm`**, the
largest error accepted for its structure class: its Dice is below the Dice such an error produces, or its HD95 is
above `error_mm` (a uniform error of *e* mm has an HD95 of about *e* mm). Both limits therefore describe the same
physical error, so the two axes flag consistently.

| Class (matched from the label name) | `error_mm` | Dice below | HD95 above | Dice measured at `error_mm` on PanTS |
|---|---|---|---|---|
| Large organ (liver, spleen, stomach, lung) | 5 | 0.85 | 5 mm | liver 0.860 |
| Compact organ (kidney, bladder, heart) | 5 | 0.65 | 5 mm | kidney 0.687 |
| Elongated organ (pancreas, duodenum, bowel) | 5 | 0.50 | 5 mm | pancreas 0.546 |
| Small organ (gallbladder, adrenal, ducts) | 5 | 0.35 | 5 mm | gallbladder 0.363 |
| Vessel (aorta, IVC, arteries) | 3 | 0.75 | 3 mm | aorta 0.755 |
| Small vessel (veins, portal and mesenteric veins) | 3 | 0.45 | 3 mm | veins 0.462 |
| Lesion / tumour | 3 | 0.65 | 3 mm | pancreatic lesion 0.673 |

The Dice column is the median, over real PanTS masks, of the mean Dice after eroding and after dilating by
`error_mm` ([sensitivity study](sensitivity-study.md)), rounded down to 0.05. The same physical error costs a
small structure far more Dice than a large one, which is why one Dice threshold for all structures flags small
organs that are fine and misses large ones that are not. `error_mm` (5 mm for organs, 3 mm for vessels and
lesions) is a SegEvalKit convention, not a published clinical standard: replace it with your clinical tolerance,
for example from inter-rater variability ([Nikolov et al. 2021](../metrics/distance.md)).

```python
P.failure_quadrants(res, "pancreas")                                          # class defaults
P.failure_quadrants(res, "pancreas", thresholds={"pancreas": {"dice": 0.7, "hd95": 10}})
P.failure_thresholds("gall_bladder")   # {'error_mm': 5.0, 'dice': 0.35, 'hd95': 5.0, 'class': 'small_organ'}
```

Unknown names use the compact-organ row. The thresholds and the rule are printed in the plot legend.

## Conformance tests

`tests/test_reference_implementations.py` runs on every change and asserts:

| Library (version) | Quantity | Agreement |
|---|---|---|
| MONAI 1.5.1 | Dice, HD, HD95, ASSD (symmetric), surface Dice at 2 mm, isotropic and anisotropic spacing | Relative 1e-5 |
| MedPy 0.5.2 | Dice, Jaccard, precision, recall, HD, HD95 (pooled mode), ASSD, RVD, isotropic and anisotropic spacing | 1e-6 |
| DeepMind surface-distance 0.1 | HD | Exact |
| DeepMind surface-distance 0.1 | HD95, surface Dice at 1 mm on smooth shapes | Within 1 mm and 0.06: a documented surfel-weighting difference |
| SegEvalKit CPU vs GPU | Every distance and overlap metric | Relative 1e-6 |

!!! note "Why DeepMind differs"
    DeepMind weights each surface element between voxels by its area, so an isolated voxel contributes six faces
    and speckle noise pulls its percentiles up more than voxel counting does. HD, a maximum, matches exactly.
