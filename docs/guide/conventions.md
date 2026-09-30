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
| **MASD** | Mean of the two **directed means** \(\tfrac12(\bar D_{P\to G} + \bar D_{G\to P})\) | MetricsReloaded; MedPy ≤ 0.5.1 `assd`. Differs from ASSD when the two surfaces have very different sizes. |
| **NSD** | Fraction of surface voxels within τ, voxel counting | MONAI `compute_surface_dice`, FLARE code. DeepMind / KiTS weight by surfel area. |

!!! warning "Name collisions"
    MedPy renamed the meaning of `assd` between 0.5.1 and 0.5.2. In SegEvalKit the names are fixed:
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
