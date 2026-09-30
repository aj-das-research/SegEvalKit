# Adding a dataset preset

A preset reproduces a benchmark's **official protocol** behind one flag:

```console
$ segevalkit evaluate --dataset my_dataset --pred preds/ --ref my_dataset/labels --out eval/
```

## 1. Collect the protocol

From the challenge paper or official evaluation code, collect:

| Item | Example (KiTS23) |
|---|---|
| Label ids and evaluation regions | kidney = 1, tumour = 2, cyst = 3; HECs: kidney+masses = {1,2,3}, masses = {2,3}, tumour = {2} |
| Reference layout | `case_xxxxx/segmentation.nii.gz` (folder layout) |
| Official metrics and parameters | Dice and surface Dice (tolerance per HEC) |
| Empty-mask convention | how cases without a tumour are scored |
| Licence, URL, citation | CC BY-NC-SA 4.0, kits-challenge.org, Heller et al. 2023 |

## 2. Register it

```python title="src/segevalkit/datasets.py"
register_dataset(DatasetPreset(
    key="my_dataset",
    title="My Dataset: Liver Lesion Segmentation",
    modality="CT",
    anatomy="Liver and liver lesions",
    labels={"liver": 1, "lesion": 2, "liver_with_lesion": [1, 2]},
    metrics=["dice", "nsd", "hd95", "lesion_f1", "lesion_recall"],
    params={"nsd": {"tolerance_mm": 2.0}},
    empty_policy="segevalkit",
    ref_layout="flat",
    cases="200 train / 50 test",
    url="https://example.org/my-dataset",
    license="CC BY 4.0",
    citation="Author et al. 2026, Journal",
    notes="Lesions < 10 voxels are ignored by the official code: "
          "use --min-lesion-voxels 10.",
))
```

For one file per structure, set `ref_layout="per_structure"`, `ref_subdir="segmentations"` and `labels=None` (all
files) or a list of structure names.

## 3. Verify against the official code

Evaluate a few cases with both the official script and the preset and compare per-case numbers. Record any deviation
(e.g. surfel-weighted surface Dice) in `notes`. The [presets page](../datasets/presets.md) is generated from the
registry.
