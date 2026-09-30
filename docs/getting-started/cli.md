# Command line

```console
$ segevalkit --help
usage: segevalkit [-h] [--version] {evaluate,report,compare,rank,metrics,recommend,datasets,visualize} ...
```

## `evaluate`

Evaluate a prediction folder against a reference folder and write the [results folder](data-format.md#the-results-folder).

```console
$ segevalkit evaluate --pred preds/ --ref labelsTr/ --labels liver=1,tumour=2 --out eval/
$ segevalkit evaluate --config eval.yaml
$ segevalkit evaluate --dataset kits23 --pred preds/ --ref kits23/cases --out eval/ --workers 8 --device cuda
```

| Flag | Meaning |
|---|---|
| `--pred`, `--ref` | prediction and reference folders |
| `--labels` | `liver=1,tumour=2`, `kidney=2+3` (region), `liver,pancreas` (per-structure files) or a JSON/YAML file |
| `--metrics` | comma-separated names or sets |
| `--dataset` | a [dataset preset](../datasets/presets.md) (labels, layout, official metrics and parameters) |
| `--config` | YAML/JSON configuration ([Configuration](configuration.md)) |
| `--prob` | folder of probability maps |
| `--nsd-tolerance` | NSD tolerance in mm |
| `--empty-policy` | `segevalkit`, `brats2023`, `metrics_reloaded`, `nan` |
| `--alignment` | `strict`, `resample`, `ignore` |
| `--connectivity`, `--min-lesion-voxels` | lesion component definition |
| `--workers`, `--device` | parallel processes; `cpu` / `cuda` |
| `--cases` | text file with the case ids to evaluate |
| `--report`, `--images` | also write `report.html`, with overlays on these images |
| `--out`, `--name` | output folder, method name |

## `report`

```console
$ segevalkit report eval/ --images imagesTr/ --gallery-metric nsd --window liver
```

## `compare`

Paired comparison of two results folders on their common cases: mean difference with bootstrap CI, fraction
of cases where A is better, rank-biserial effect size, Wilcoxon / t / permutation p-values with Holm or BH
correction, and a forest plot.

```console
$ segevalkit compare eval_A/ eval_B/ --metrics dice,nsd,hd95 --out cmp/
```

## `rank`

```console
$ segevalkit rank eval_A/ eval_B/ eval_C/ --metric dice --label pancreas --out ranking.png
```

Prints aggregate-then-rank and rank-then-aggregate rankings and the bootstrap ranking stability (median Kendall τ),
and draws the blob plot.

## `recommend`

```console
$ segevalkit recommend --structure tubular --boundary-critical --tolerance 1
$ segevalkit recommend --structure small_lesion --multi-instance --probabilistic --markdown
```

## `visualize`

```console
$ segevalkit visualize --pred p.nii.gz --ref g.nii.gz --image ct.nii.gz --label 2 --kind triplanar --out fig.png
$ segevalkit visualize --pred p.nii.gz --ref g.nii.gz --label 1 --kind surface --out surface.html
```

`--kind`: `slice`, `triplanar`, `montage`, `projection`, `surface` (`.png` static or `.html` interactive).

## `metrics` and `datasets`

```console
$ segevalkit metrics -v                 # every metric with its direction, unit and one-line summary
$ segevalkit metrics --family topology
$ segevalkit datasets                   # dataset presets
$ segevalkit datasets brats2023
```
