# Quickstart

Evaluate a prediction folder against a reference folder, read the results, plot, and write a report — about
five minutes.

## 1. Your data

```text
labelsTr/                 predictions/
├── case_001.nii.gz       ├── case_001.nii.gz      # multi-label maps, same ids
├── case_002.nii.gz       ├── case_002.nii.gz
└── ...                   └── ...
```

Case ids come from file names (nnU-Net's `_0000` channel suffix is stripped) or folder names. The **reference**
defines the case list: a missing prediction is scored as an empty mask, never skipped. Other layouts:
[Data & output format](data-format.md).

## 2. Evaluate

=== "Python"

    ```python
    import segevalkit as sek

    ev = sek.Evaluator(
        # regions are unions of ids
        labels={"liver": 1, "tumour": 2, "liver_incl_tumour": [1, 2]},
        metrics=["default", "detection"],
        params={"nsd": {"tolerance_mm": 2.0}},
    )
    res = ev.evaluate("predictions/", "labelsTr/",
                      n_workers=8, out_dir="eval/", name="my-model")
    print(res)
    # EvaluationResult(name='my-model', cases=131,
    #                  labels=['liver', 'tumour', 'liver_incl_tumour'], metrics=18)
    ```

=== "Command line"

    ```console
    $ segevalkit evaluate --pred predictions/ --ref labelsTr/ \
        --labels liver=1,tumour=2,liver_incl_tumour=1+2 \
        --metrics default,detection --nsd-tolerance 2 \
        --out eval/ --workers 8 --name my-model
    ```

`"default"` = Dice, IoU, NSD, HD95, ASSD, precision, recall, relative volume difference. Unsure what to report?

```console
$ segevalkit recommend --structure small_lesion --multi-instance --volumetry
```

## 3. Read the results

```python
res.summary()      # per label & metric: n, mean, std, median, IQR, 95 % bootstrap CI
res.wide()         # one row per (case, label), one column per metric
res.worst_cases("dice", "tumour", k=5)
res.lesions        # one row per reference lesion / false-positive component
```

The same tables are written to `eval/` as plain CSV and JSON (`per_case.csv`, `per_case_wide.csv`,
`lesions.csv`, `summary.csv`, `meta.json`) — see [the results folder](data-format.md#the-results-folder).

## 4. Plot

```python
from segevalkit import plotting as P

P.metric_distribution(res, "dice").savefig("dice.png")             # raincloud per structure
P.metric_vs_size(res, "dice", label="tumour").savefig("size.png")  # size bias
P.failure_quadrants(res, "liver").savefig("failures.png")          # Dice vs HD95
P.detection_by_size(res, label="tumour").savefig("detect.png")
```

All plots: [plot gallery](../analysis/plots.md).

## 5. Look at the errors

```python
from segevalkit import viz
from segevalkit.io import load_volume

img = load_volume("imagesTr/case_007_0000.nii.gz", kind="image")
ref = load_volume("labelsTr/case_007.nii.gz")
pred = load_volume("predictions/case_007.nii.gz")
fig = viz.triplanar(img.data, pred.data == 2, ref.data == 2,
                    affine=ref.affine, window="liver")
fig.savefig("case_007.png")
```

Violet = agreement, orange = missed tissue, teal = added tissue.

## 6. Compare two models

```python
from segevalkit.stats import compare

a, b = sek.load_results("eval_modelA/"), sek.load_results("eval_modelB/")
# paired Wilcoxon, Holm-corrected, with effect sizes
cmp = compare(a, b, metrics=["dice", "nsd", "hd95"])
P.comparison_forest(cmp, name_a="A", name_b="B")
```

## 7. HTML report

```console
$ segevalkit report eval/ --images imagesTr/
```

Configuration, warnings, summary tables, figures, a worst-case gallery and a metric glossary in one file. See
[HTML report](../analysis/report.md).
