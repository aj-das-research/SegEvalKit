# Data & output format

SegEvalKit aims to be the **standard interface** between a segmentation model and its evaluation: any model that
writes NIfTI files in one of three layouts can be evaluated on any dataset, and every evaluation produces the same
documented results folder.

## Input layouts

=== "flat"

    One multi-label file per case directly in the folder. Used by nnU-Net, MSD, AMOS, BTCV, KiTS, FLARE.

    ```text
    labels/
    ├── case_001.nii.gz
    └── case_002.nii.gz
    ```

=== "folder"

    One multi-label file inside a folder per case.

    ```text
    labels/
    ├── case_001/
    │   └── combined_labels.nii.gz
    └── case_002/
        └── combined_labels.nii.gz
    ```

=== "per_structure"

    One binary file per structure, optionally in a sub-folder. Used by TotalSegmentator, AbdomenAtlas, PanTS.

    ```text
    labels/
    ├── case_001/
    │   └── segmentations/
    │       ├── liver.nii.gz
    │       └── pancreas.nii.gz
    └── case_002/ ...
    ```

`layout="auto"` (the default) detects which one a folder uses. Prediction and reference **do not need the same
layout**: a model writing `pancreas = 7` in a multi-label map can be scored against `segmentations/pancreas.nii.gz`.

| Format | Extension | Requirement |
|---|---|---|
| NIfTI | `.nii`, `.nii.gz` | built in (nibabel) |
| MetaImage / NRRD | `.mha`, `.mhd`, `.nrrd` | `pip install segevalkit[sitk]` |
| NumPy | `.npy`, `.npz` | `.npz` may carry `spacing`; `.npy` needs `spacing=` |

## Labels

Labels map a structure name to how to find it on each side:

```yaml
labels:
  liver: 1                      # same id in prediction and reference
  kidney: [2, 3]                # region = union of ids (left + right)
  pancreas: {ref: 1, pred: 7}   # different ids per side
  lesion: {ref_file: pancreatic_lesion.nii.gz, pred: 28}
```

An MSD / nnU-Net `dataset.json` `labels` block can be passed as is. Without labels, SegEvalKit evaluates every
non-zero id (multi-label references) or every structure file (per-structure references).

## Probability maps

Calibration metrics need foreground probabilities. Pass a folder with one float NIfTI per case and structure,
`probs/<case>/<structure>.nii.gz`, values in [0, 1], on the reference grid:

```python
ev.evaluate("predictions/", "labels/", prob="probs/")
```

## Geometry checks

Before comparing masks SegEvalKit checks that prediction and reference share **shape, spacing and affine**.

| `alignment=` | Behaviour |
|---|---|
| `"strict"` (default) | a mismatch fails the case (recorded in `meta.json`, the run continues) |
| `"resample"` | the prediction is resampled onto the reference grid (nearest neighbour) |
| `"ignore"` | only equal shapes are required; use when headers are known to be unreliable |

Spacing always comes from the **reference** header, so distances are in millimetres and volumes in millilitres.

## The results folder

```text
eval/
├── per_case.csv
├── per_case_wide.csv
├── lesions.csv
├── summary.csv
└── meta.json
```

| File | Content |
|---|---|
| `per_case.csv` | Tidy table: `case_id, label, metric, value`. One row per number. Descriptive per-case flags start with `_`: `_ref_empty`, `_pred_empty`, `_ref_volume_ml`, `_pred_volume_ml`. |
| `per_case_wide.csv` | One row per `(case_id, label)`; one column per metric plus `ref_empty`, `pred_empty`, `ref_volume_ml`, `pred_volume_ml`. |
| `lesions.csv` | One row per reference lesion (`kind = ref`) and per unmatched predicted component (`kind = pred_fp`): `volume_ml`, `detected`, `dice`, `iou`, `n_touching`. |
| `summary.csv` | Per `(label, metric)`: `n, n_nan, mean, std, median, q1, q3, min, max, ci_low, ci_high` (95 % percentile bootstrap of the mean). |
| `meta.json` | `results_format`, `segevalkit_version`, timestamp, Python version, the complete configuration (metrics, parameters, empty policy, connectivity, alignment, device), paths and layouts, labels, missing and extra predictions, per-case errors, runtime. |

Load a folder back with `segevalkit.load_results("eval/")`. The format is versioned (`results_format: 1`).

!!! tip "Why a long table?"
    One row per number means new metrics never change the schema, NaNs are explicit, and the table drops
    straight into pandas, R, or any plotting library. `per_case_wide.csv` is a convenience view of the same data.
