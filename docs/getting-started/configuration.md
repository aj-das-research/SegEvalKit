# Configuration

Every option that can change a reported number is an explicit argument, and all of them are written to
`meta.json` with the results. Nothing depends on hidden defaults of a third-party library.

## Evaluator options

| Option | Default | Meaning |
|---|---|---|
| `labels` | all ids / files | Structures to evaluate ([forms](data-format.md#labels)). |
| `metrics` | `"default"` | Names, aliases (`dsc`, `jaccard`, `hausdorff`...), sets (`default`, `overlap`, `distance`, `volume`, `topology`, `detection`, `calibration`, `agreement`) or `"all"`. |
| `params` | `{}` | Per-metric keyword overrides, e.g. `{"nsd": {"tolerance_mm": 1.0}, "hd_percentile": {"q": 99}}`. |
| `empty` | `EmptyPolicy()` | Empty-mask policy (see below). |
| `device` | `"cpu"` | `"cuda"` / `"cuda:1"` for GPU surface distances and confusion counts. |
| `connectivity` | `26` | Connectivity of lesion components (6, 18, 26). |
| `min_lesion_voxels` | `0` | Components smaller than this are ignored by detection metrics. |
| `alignment` | `"strict"` | `strict`, `resample` or `ignore` ([geometry checks](data-format.md#geometry-checks)). |
| `missing_pred` | `"empty"` | `empty` scores a missing prediction as an empty mask; `skip` drops the case. |
| `lesion_table` | `True` | Collect per-lesion rows when detection metrics run. |

Per-structure parameters override global ones. Tolerances *should* depend on the structure:

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

When both masks are empty the structure was correctly called absent. When one is empty most metrics are
undefined. SegEvalKit makes the choice explicit:

| Preset | Both empty | Distance with one empty | Used by |
|---|---|---|---|
| `segevalkit` (default) | ideal value (Dice 1, HD 0) | image diagonal in mm | this library |
| `brats2023` | ideal value | 374 mm | BraTS 2023 lesion-wise code |
| `metrics_reloaded` | ideal value | worst (image diagonal) | Metrics Reloaded aggregation advice |
| `nan` / `nnunet` | NaN (excluded) | NaN (excluded) | nnU-Net, MONAI `ignore_empty` |

```python
from segevalkit import EmptyPolicy
ev = sek.Evaluator(..., empty=EmptyPolicy.preset("brats2023"))
ev = sek.Evaluator(..., empty=EmptyPolicy(both_empty="nan", one_empty_distance=100.0))
```

Precision with an empty prediction and recall with an empty reference are *undefined* (NaN), never a silent 0.
Every per-case row carries `ref_empty` / `pred_empty` flags so you can analyse these cases separately.

## Configuration file

The same options in YAML, for the command line:

```yaml title="eval.yaml"
pred: /data/predictions/nnunet
ref: /data/PanTS/LabelTe
images: /data/PanTS/ImageTe
name: nnunet-pants
out: eval/nnunet
labels:
  pancreas: {pred: [17, 18, 19, 20, 21, 28], ref_file: pancreas.nii.gz}
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

Command-line flags override the file.
