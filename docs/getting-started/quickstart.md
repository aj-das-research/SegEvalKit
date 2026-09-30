# Quickstart

This page evaluates a folder of predictions against a folder of references, reads the results, draws the key
plots and writes a report. It takes about five minutes.

## 1. Your data

SegEvalKit reads the layouts used by almost every dataset and model output (details in
[Data & output format](data-format.md)):

```text
labelsTr/                 predictions/
├── case_001.nii.gz       ├── case_001.nii.gz      # multi-label maps, same ids
├── case_002.nii.gz       ├── case_002.nii.gz
└── ...                   └── ...
```

Case ids come from file names (nnU-Net's `_0000` channel suffix is stripped) or from folder names. The
**reference** defines the case list: a missing prediction is scored as an empty mask, never silently skipped.

## 2. Evaluate

=== "Python"

    ```python
    import segevalkit as sek

    ev = sek.Evaluator(
        labels={"liver": 1, "tumour": 2, "liver_incl_tumour": [1, 2]},   # regions are unions of ids
        metrics=["default", "detection"],
        params={"nsd": {"tolerance_mm": 2.0}},
    )
    res = ev.evaluate("predictions/", "labelsTr/", n_workers=8, out_dir="eval/", name="my-model")
    print(res)
    # EvaluationResult(name='my-model', cases=131, labels=['liver', 'tumour', 'liver_incl_tumour'], metrics=18)
    ```

=== "Command line"

    ```console
    $ segevalkit evaluate --pred predictions/ --ref labelsTr/ \
        --labels liver=1,tumour=2,liver_incl_tumour=1+2 \
        --metrics default,detection --nsd-tolerance 2 \
        --out eval/ --workers 8 --name my-model
    ```

`"default"` is a compact, broadly recommended set: Dice, IoU, NSD, HD95, ASSD, precision, recall and relative
volume difference. Not sure what to report? Ask the recommender:

```console
$ segevalkit recommend --structure small_lesion --multi-instance --volumetry
```

## 3. Read the results

```python
res.summary()                         # per label & metric: n, mean, std, median, IQR, 95 % bootstrap CI
res.wide()                            # one row per (case, label), one column per metric
res.worst_cases("dice", "tumour", k=5)
res.lesions                           # one row per reference lesion / false-positive component
```

Everything is also on disk in `eval/` as plain CSV and JSON, readable without SegEvalKit:

```text
eval/
├── per_case.csv          case_id, label, metric, value      (tidy / long)
├── per_case_wide.csv     one row per (case, label)
├── lesions.csv           per-lesion table
├── summary.csv           descriptive statistics with CIs
└── meta.json             versions, full configuration, paths, missing and failed cases
```

## 4. Plot

```python
from segevalkit import plotting as P

P.metric_distribution(res, "dice").savefig("dice.png")           # raincloud per structure
P.metric_vs_size(res, "dice", label="tumour").savefig("size.png")   # size bias
P.failure_quadrants(res, "liver").savefig("failures.png")         # Dice vs HD95
P.detection_by_size(res, label="tumour").savefig("detect.png")
```

See the [plot gallery](../analysis/plots.md) for all of them.

## 5. Look at the errors

```python
from segevalkit import viz
from segevalkit.io import load_volume

img = load_volume("imagesTr/case_007_0000.nii.gz", kind="image")
ref = load_volume("labelsTr/case_007.nii.gz")
pred = load_volume("predictions/case_007.nii.gz")
viz.triplanar(img.data, pred.data == 2, ref.data == 2, affine=ref.affine, window="liver").savefig("case_007.png")
```

Violet is agreement, orange is missed tissue, teal is added tissue.

## 6. Compare two models

```python
from segevalkit.stats import compare

a, b = sek.load_results("eval_modelA/"), sek.load_results("eval_modelB/")
compare(a, b, metrics=["dice", "nsd", "hd95"])      # paired Wilcoxon, Holm-corrected, effect sizes
P.comparison_forest(compare(a, b), name_a="A", name_b="B")
```

## 7. One HTML report

```console
$ segevalkit report eval/ --images imagesTr/
```

The report embeds the configuration, warnings, summary tables, figures, a worst-case gallery and a glossary of
every metric used. See [HTML report](../analysis/report.md).
