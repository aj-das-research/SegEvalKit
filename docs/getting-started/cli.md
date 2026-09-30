# Command line

```console
$ segevalkit --help
usage: segevalkit [-h] [--version]
                  {evaluate,report,compare,rank,metrics,recommend,datasets,visualize} ...
```

## `evaluate`

Score a prediction folder against a reference folder and write the [results folder](data-format.md#the-results-folder).

```console
$ segevalkit evaluate --pred preds/ --ref labelsTr/ --labels liver=1,tumour=2 --out eval/
$ segevalkit evaluate --config eval.yaml
$ segevalkit evaluate --dataset kits23 --pred preds/ --ref kits23/cases \
    --out eval/ --workers 8 --device cuda
```

| Flag | Meaning |
|---|---|
| `--pred`, `--ref` | prediction and reference folders |
| `--labels` | `liver=1,tumour=2`, `kidney=2+3` (region), `liver,pancreas` (per-structure files) or a JSON/YAML file |
| `--metrics` | comma-separated names or sets |
| `--dataset` | a [dataset preset](../datasets/presets.md) (labels, layout, official metrics and parameters) |
| `--config` | YAML/JSON [configuration file](configuration.md#configuration-file) |
| `--prob` | folder of probability maps |
| `--nsd-tolerance` | NSD tolerance in mm |
| `--empty-policy` | `segevalkit`, `brats2023`, `metrics_reloaded`, `nan`, `nnunet`, `topcow` |
| `--alignment` | `strict`, `resample`, `ignore` |
| `--connectivity`, `--min-lesion-voxels` | lesion component definition |
| `--workers`, `--device` | parallel processes; `cpu`, `cuda` or `cuda:N` |
| `--cases` | text file with the case ids to evaluate |
| `--report`, `--images` | also write `report.html`, with overlays on these images |
| `--out`, `--name` | output folder, method name |

## Other commands

`report RESULTS`
:   HTML report from a results folder. Options: `--out`, `--images`, `--gallery-metric` (default `dice`),
    `--window` (default `abdomen`).

`compare A B`
:   Paired comparison on common cases: mean difference with bootstrap CI, fraction of cases where A is better,
    rank-biserial effect size, corrected p-values, forest plot. Options: `--metrics`,
    `--test {wilcoxon,ttest,permutation}`, `--correction {holm,bh,none}`, `--name-a`, `--name-b`, `--out`.

`rank R1 R2 ...`
:   Aggregate-then-rank and rank-then-aggregate rankings, bootstrap stability (median Kendall τ) and a blob plot.
    Options: `--metric` (default `dice`), `--label`, `--n-boot` (default 1000), `--out`.

`recommend`
:   Metrics to report for a problem fingerprint. Options:
    `--structure {large_organ,small_structure,small_lesion,large_lesion,tubular,hollow}`, `--multi-instance`,
    `--boundary-critical`, `--volumetry`, `--empty-references`, `--probabilistic`, `--asymmetric`,
    `--noisy-reference`, `--ranking`, `--tolerance MM`, `--markdown`.

`visualize`
:   Qualitative error figure for one case. Options: `--pred`, `--ref`, `--image`, `--label`,
    `--kind {slice,triplanar,montage,projection,surface}`, `--view`, `--window`, `--title`, `--out`
    (`.png` static; `.html` interactive for `surface`).

`metrics`, `datasets [NAME]`
:   List metrics (`-v` adds direction, unit and summary; `--family` filters) or dataset presets.

```console
$ segevalkit report eval/ --images imagesTr/ --gallery-metric nsd --window liver
$ segevalkit compare eval_A/ eval_B/ --metrics dice,nsd,hd95 --out cmp/
$ segevalkit rank eval_A/ eval_B/ eval_C/ --metric dice --label pancreas --out ranking.png
$ segevalkit recommend --structure tubular --boundary-critical --tolerance 1
$ segevalkit visualize --pred p.nii.gz --ref g.nii.gz --image ct.nii.gz --label 2 \
    --kind triplanar --out fig.png
$ segevalkit visualize --pred p.nii.gz --ref g.nii.gz --label 1 \
    --kind surface --out surface.html
$ segevalkit metrics --family topology
$ segevalkit datasets brats2023
```
