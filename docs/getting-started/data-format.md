# Data & output format

Any model that writes masks in one of three layouts can be evaluated on any dataset, and every evaluation writes
the same documented results folder.

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

`layout="auto"` (default) detects the layout. Prediction and reference **may differ**: a multi-label map with
`pancreas = 7` can be scored against `segmentations/pancreas.nii.gz`.

| Format | Extension | Requirement |
|---|---|---|
| NIfTI | `.nii`, `.nii.gz` | built in (nibabel) |
| MetaImage / NRRD | `.mha`, `.mhd`, `.nrrd` | `pip install segevalkit[sitk]` |
| NumPy | `.npy`, `.npz` | `.npz` may carry `spacing`; `.npy` needs `spacing=` |

## Labels

Each structure name maps to how it is found on each side:

```yaml
labels:
  liver: 1                      # same id in prediction and reference
  kidney: [2, 3]                # region = union of ids (left + right)
  pancreas: {ref: 1, pred: 7}   # different ids per side
  lesion: {ref_file: pancreatic_lesion.nii.gz, pred: 28}
```

An MSD / nnU-Net `dataset.json` `labels` block works as is. Without labels, every non-zero id (multi-label
references) or every structure file (per-structure references) is evaluated.

## Probability maps

Calibration metrics need foreground probabilities: one float NIfTI per case and structure,
`probs/<case>/<structure>.nii.gz`, values in [0, 1], on the reference grid.

```python
ev.evaluate("predictions/", "labels/", prob="probs/")
```

## Geometry checks

Prediction and reference must share **shape, spacing and affine**:

| `alignment=` | Behaviour |
|---|---|
| `"strict"` (default) | a mismatch fails the case (recorded in `meta.json`, the run continues) |
| `"resample"` | the prediction is resampled onto the reference grid (nearest neighbour) |
| `"ignore"` | only equal shapes are required (for known-unreliable headers) |

Spacing always comes from the **reference** header; distances are in mm, volumes in mL.

## The results folder

`per_case.csv`
:   Tidy table `case_id, label, metric, value`, one row per number. Descriptive per-case flags start with `_`:
    `_ref_empty`, `_pred_empty`, `_ref_volume_ml`, `_pred_volume_ml`.

`per_case_wide.csv`
:   One row per `(case_id, label)`; one column per metric plus `ref_empty`, `pred_empty`, `ref_volume_ml`,
    `pred_volume_ml`.

`lesions.csv`
:   One row per reference lesion (`kind = ref`) and per unmatched predicted component (`kind = pred_fp`):
    `volume_ml`, `detected`, `dice`, `iou`, `n_touching`.

`summary.csv`
:   Per `(label, metric)`: `n, n_nan, mean, std, median, q1, q3, min, max, ci_low, ci_high` (95 % percentile
    bootstrap of the mean).

`meta.json`
:   `results_format`, `segevalkit_version`, timestamp, Python version, full configuration (metrics, parameters,
    empty policy, connectivity, alignment, device), paths, layouts, labels, missing/extra predictions, per-case
    errors, runtime.

Load it back with `segevalkit.load_results("eval/")`. The format is versioned (`results_format: 1`).

!!! tip "Why a long table?"
    New metrics never change the schema, NaNs are explicit, and the table drops straight into pandas, R or any
    plotting library. `per_case_wide.csv` is a convenience view of the same data.
