# Quickstart

Evaluate a folder of predictions, read the results, find the failures, compare models, plot and write a report.
Every output on this page is real: three official models (nnU-Net ResEnc-M, MedFormer and TotalSegmentator, run
from their own code and checkpoints) on 8 PanTS test CTs chosen to span tumour sizes: 2 large, 2 medium, 2 small
and 2 tumour-free.

!!! info "Real outputs, not a benchmark"
    Eight cases illustrate the library; they are too few to rank models. Preparing your own data?
    Start with [What you need](prepare.md). No data at hand? `segevalkit view --demo` opens the
    [interactive viewer](../analysis/viewer.md) on a synthetic phantom ([live demo](../assets/viewer/demo_phantom.html){ target="_blank" }).

## 1. Describe what to evaluate

Structures are named once; each side says where to find them. Here the PanTS reference has one file per structure
(`LabelTe/<case>/segmentations/<name>.nii.gz`) and nnU-Net writes one multi-label map per case. The reference
defines the case list: a missing prediction is scored as an empty mask, never skipped. Other layouts:
[Data & output format](data-format.md).

=== "Python"

    ```python
    import segevalkit as sek

    ev = sek.Evaluator(
        labels={
            "pancreas": {"ref_file": "pancreas.nii.gz+pancreatic_lesion.nii.gz",
                         "pred": [17, 18, 19, 20, 21, 28]},
            "pancreatic_lesion": {"ref_file": "pancreatic_lesion.nii.gz", "pred": 28,
                                  "metrics": ["default", "detection"]},
            "liver": {"ref_file": "liver.nii.gz", "pred": 14},
        },
        metrics="default",
        params={"nsd": {"tolerance_mm": 2.0}},
        min_lesion_voxels=10,
    )
    ```

=== "Configuration file"

    ```yaml title="eval_nnunet.yaml"
    pred: predictions/nnunet
    ref: PanTS/LabelTe
    images: PanTS/ImageTe
    out: eval/nnunet
    labels:
      pancreas:
        ref_file: pancreas.nii.gz+pancreatic_lesion.nii.gz
        pred: [17, 18, 19, 20, 21, 28]
      pancreatic_lesion:
        ref_file: pancreatic_lesion.nii.gz
        pred: 28
        metrics: [default, detection]
      liver: {ref_file: liver.nii.gz, pred: 14}
    metrics: [default]
    params: {nsd: {tolerance_mm: 2.0}}
    min_lesion_voxels: 10
    ```

The pancreas is scored as pancreas ∪ lesion because PanTS places the lesion inside the pancreas mask in some cases
and outside it in others ([pitfall](../guide/pitfalls.md#annotation-conventions)). `"default"` is Dice, IoU, NSD,
HD95, ASSD, precision, recall and relative volume difference; unsure what to report? Run
`segevalkit recommend` ([decision guide](../guide/choosing.md)).

## 2. Evaluate and summarise

One call scores every case; `summary()` gives mean, median and a 95 % bootstrap confidence interval per structure
and metric. The same tables are written to `eval/nnunet/` as CSV and JSON
([the results folder](data-format.md#the-results-folder)).

=== "Python"

    ```python
    res = ev.evaluate("predictions/nnunet/", "PanTS/LabelTe/",
                      n_workers=8, out_dir="eval/nnunet")
    res.summary().query("metric in ['dice', 'nsd', 'hd95']")
    ```

=== "Command line"

    ```console
    $ segevalkit evaluate --config eval_nnunet.yaml
    ```

```text
            label metric  n    mean  median  ci_low  ci_high
         pancreas   dice  8   0.889   0.905   0.840    0.927
         pancreas    nsd  8   0.843   0.887   0.757    0.915
         pancreas   hd95  8   5.925   2.890   2.575   10.228
pancreatic_lesion   dice  8   0.350   0.204   0.097    0.619
pancreatic_lesion    nsd  8   0.311   0.196   0.075    0.561
pancreatic_lesion   hd95  8 290.878 252.815 100.865  481.879
            liver   dice  8   0.981   0.981   0.978    0.985
            liver    nsd  8   0.937   0.948   0.915    0.958
            liver   hd95  8   6.264   2.394   2.032   11.556
```

The lesion rows show why the [empty-mask policy](configuration.md#empty-masks) matters: in tumour-free patients a
spurious lesion prediction scores Dice 0 and the image-diagonal HD95 penalty, which drags the lesion HD95 to
hundreds of millimetres.

## 3. Find the failures

```python
res.worst_cases("dice", "pancreas", k=3)
```

```text
       case_id  dice
PanTS_00009746 0.733
PanTS_00009287 0.862
PanTS_00009322 0.881
```

`res.wide()` has one row per case and structure, one column per metric.

## 4. Look at lesions, not voxels

The lesion table shows what the Dice average hides: nnU-Net detected 3 of the 7 reference lesions in these cases
and missed a 23.7 mL tumour entirely.

```python
res.lesions.query("kind == 'ref'")[["case_id", "volume_ml", "detected", "dice"]]
```

```text
       case_id  volume_ml  detected  dice
PanTS_00009287      0.145     False 0.000
PanTS_00009287      0.103     False 0.000
PanTS_00009760     23.745     False 0.000
PanTS_00009027      0.181     False 0.000
PanTS_00009152     15.251      True 0.778
PanTS_00009329      2.187      True 0.702
PanTS_00009544      9.765      True 0.666
```

## 5. Compare and rank models

Paired tests (Wilcoxon, Holm-corrected) and challenge-style rankings
([statistics](../analysis/statistics.md)):

```python
from segevalkit.stats import compare, rank_methods

compare(res_nnunet, res_medformer, metrics=["dice", "nsd", "hd95"], labels=["pancreas"])
rank_methods({"nnU-Net": res_nnunet, "MedFormer": res_medformer, "TotalSegmentator": res_ts},
             "dice", label="pancreas")
```

```text
   label metric  n  mean_a  mean_b  mean_diff  frac_a_better  p_adjusted
pancreas   dice  8   0.889   0.894     -0.005          0.250       0.445
pancreas    nsd  8   0.843   0.858     -0.016          0.125       0.445
pancreas   hd95  8   5.925   5.785      0.139          0.250       0.445

          method  mean  rank
       MedFormer 0.894   1.0
nnU-Net ResEnc-M 0.889   2.0
TotalSegmentator 0.863   3.0
```

With 8 cases no difference is significant (adjusted p = 0.445): exactly what the paired test is for.

## 6. Plot and look

```python
from segevalkit import plotting as P

P.metric_distribution({"nnU-Net": r1, "MedFormer": r2, "TotalSegmentator": r3}, "dice",
                      labels=organs)
```

<figure class="sk-fig sk-fig--wide" markdown>
[![Dice per structure and model](../assets/showcase/dist_dice.png)](../assets/showcase/dist_dice.png)
<figcaption>Raincloud of per-case Dice: shape, median and IQR, and every case. Every figure uses one
colour-vision-safe palette. All plots: <a href="../../analysis/plots/">plot gallery</a>.</figcaption>
</figure>

```python
from segevalkit import viz

viz.triplanar(ct, pred_pancreas, ref_pancreas, affine=ref.affine, window="pancreas")
```

=== "Three models"

    <figure class="sk-fig sk-fig--wide" markdown>
    [![Three models on one slice](../assets/showcase/model_comparison.png)](../assets/showcase/model_comparison.png)
    <figcaption>One slice of PanTS_00009152, three models: pancreas ∪ lesion (top) and lesion (bottom). Violet
    agreement, orange missed, teal added. TotalSegmentator has no lesion class.</figcaption>
    </figure>

=== "Tri-planar"

    <figure class="sk-fig sk-fig--wide" markdown>
    [![Tri-planar error view of the pancreas](../assets/showcase/triplanar_pancreas.png)](../assets/showcase/triplanar_pancreas.png)
    <figcaption>nnU-Net pancreas on the same case, through the region of largest error in each plane.</figcaption>
    </figure>

More views: [qualitative visualisation](../analysis/qualitative.md).

## 7. Report

One self-contained HTML file with provenance, tables, figures, the worst cases and a metric glossary:
[open the sample report](../assets/showcase/report_nnunet.html){ target="_blank" }.

=== "Python"

    ```python
    sek.report.build_report(res, "eval/nnunet/report.html", image_source="PanTS/ImageTe/")
    ```

=== "Command line"

    ```console
    $ segevalkit evaluate --config eval_nnunet.yaml --report
    $ segevalkit compare eval/nnunet eval/medformer --out comparison/
    ```

See [HTML report](../analysis/report.md) and the [command-line reference](cli.md).
