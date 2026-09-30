# Configuration

Every option that can change a reported number is an explicit argument and is written to `meta.json`. Nothing
depends on a third-party library's hidden defaults.

## Evaluator options

| Option | Default | Meaning |
|---|---|---|
| `labels` | all ids / files | Structures to evaluate ([forms](data-format.md#labels)). |
| `metrics` | `"default"` | Names, aliases (`dsc`, `jaccard`, `hausdorff`...), sets (`default`, `overlap`, `distance`, `volume`, `topology`, `detection`, `calibration`, `agreement`) or `"all"`. |
| `params` | `{}` | Per-metric keyword overrides, e.g. `{"nsd": {"tolerance_mm": 1.0}, "hd_percentile": {"q": 99}}`. |
| `empty` | `EmptyPolicy()` | Empty-mask policy ([below](#empty-masks)). |
| `device` | `"cpu"` | `"cuda"` / `"cuda:1"` for GPU surface distances and confusion counts. |
| `connectivity` | `26` | Connectivity of lesion components (6, 18, 26). |
| `min_lesion_voxels` | `0` | Components smaller than this are ignored by detection metrics. |
| `alignment` | `"strict"` | `strict`, `resample` or `ignore` ([geometry checks](data-format.md#geometry-checks)). |
| `missing_pred` | `"empty"` | `empty`: score a missing prediction as an empty mask; `skip`: drop the case. |
| `lesion_table` | `True` | Collect per-lesion rows when detection metrics run. |

Per-structure parameters override global ones — tolerances *should* depend on the structure:

```python
ev = sek.Evaluator(
    labels={
        "liver": {"values": [1], "params": {"nsd": {"tolerance_mm": 3.0}}},
        "tumour": {"values": [2], "params": {"nsd": {"tolerance_mm": 1.0}}},
    },
    metrics=["dice", "nsd"],
)
```

## Empty masks

Both masks empty means the structure was correctly called absent; with one empty, most metrics are undefined.
The choice is an explicit preset:

| Preset | Both empty | Distance with one empty | Used by |
|---|---|---|---|
| `segevalkit` (default) | ideal value (Dice 1, HD 0) | image diagonal in mm | this library |
| `brats2023` | ideal value | 374 mm | BraTS 2023 lesion-wise code |
| `metrics_reloaded` | ideal value | worst (image diagonal) | Metrics Reloaded aggregation advice |
| `nan` / `nnunet` | NaN (excluded) | NaN (excluded) | nnU-Net, MONAI `ignore_empty` |
| `topcow` | ideal value | 90 mm | TopCoW 2024 HD95 cap |

```python
from segevalkit import EmptyPolicy
ev = sek.Evaluator(..., empty=EmptyPolicy.preset("brats2023"))
ev = sek.Evaluator(..., empty=EmptyPolicy(both_empty="nan", one_empty_distance=100.0))
```

Precision with an empty prediction and recall with an empty reference are NaN, never a silent 0. Per-case
`ref_empty` / `pred_empty` flags let you analyse these cases separately.

## Configuration file

The same options in YAML for the command line:

```yaml title="eval.yaml"
pred: /data/predictions/nnunet
ref: /data/PanTS/LabelTe
images: /data/PanTS/ImageTe
name: nnunet-pants
out: eval/nnunet
labels:
  pancreas: {pred: [17, 18, 19, 20, 21, 28], ref_file: pancreas.nii.gz+pancreatic_lesion.nii.gz}
  pancreatic_lesion: {pred: 28, ref_file: pancreatic_lesion.nii.gz}
metrics: [default, detection, cldice]
params:
  nsd: {tolerance_mm: 2.0}
empty_policy: segevalkit
connectivity: 26
min_lesion_voxels: 10
alignment: resample
device: cuda
workers: 4
report: true
```

```console
$ segevalkit evaluate --config eval.yaml
```

Command-line flags override the file, except `--connectivity` and `--min-lesion-voxels`, where a value in the
file wins.
