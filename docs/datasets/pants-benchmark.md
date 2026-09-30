# PanTS benchmark

Three official models, run here from their official code and checkpoints, on the **901 PanTS test CTs**, evaluated with SegEvalKit at every level: per case, pooled over the test set, by patient cohort and per patient.

!!! info "Evaluation still running"
    Evaluated so far: TotalSegmentator (901 cases). nnU-Net ResEnc-M and MedFormer are still running; the tables gain a column as each evaluation finishes.

## Protocol

| | |
|---|---|
| Reference | per-structure masks `LabelTe/<case>/segmentations/*.nii.gz`; pancreas = pancreas ∪ lesion |
| Geometry | voxel correspondence (`alignment="ignore"`): 227 of 901 cases carry label headers that disagree with the CT ([pitfall](../guide/pitfalls.md#header-mismatch)) |
| Metrics | Dice, NSD at 2 mm, HD95, ASSD, volume errors; lesion-wise detection for lesion models |
| Missing predictions | scored as empty |
| Statistics | 95 % bootstrap CIs over cases; cohort tests Kruskal–Wallis / Mann–Whitney, Holm-corrected |

## Organ segmentation (mean over cases)

| Structure | DSC TotalSeg. | NSD TotalSeg. | HD95 TotalSeg. |
|---|---:|---:|---:|
| Pancreas | .823 | .788 | 12.2 |
| Liver | .963 | .898 | 6.2 |
| Spleen | .946 | .938 | 7.6 |
| Kidney (L) | .890 | .844 | 16.9 |
| Kidney (R) | .905 | .853 | 13.3 |
| Stomach | .871 | .795 | 14.7 |
| Gallbladder | .755 | .755 | 48.6 |
| Duodenum | .737 | .669 | 18.1 |
| Aorta | .881 | .884 | 19.9 |
| IVC | .817 | .804 | 24.3 |
| Adrenal (L) | .758 | .901 | 10.5 |
| Adrenal (R) | .695 | .883 | 7.7 |
| Colon | .856 | .780 | 22.7 |

DSC and NSD: higher is better; HD95 in mm, lower is better.

## Macro and pooled Dice

The macro value averages per-case Dice (every patient weighs the same); the pooled value sums voxels over the test set (every voxel weighs the same). See [evaluation levels](../analysis/levels.md).

| Structure | TotalSeg. macro | TotalSeg. pooled |
|---|---:|---:|
| Pancreas | .823 | .817 |
| Liver | .963 | .966 |
| Spleen | .946 | .956 |
| Kidney (L) | .890 | .897 |
| Kidney (R) | .905 | .916 |
| Stomach | .871 | .873 |
| Gallbladder | .755 | .831 |
| Duodenum | .737 | .744 |
| Aorta | .881 | .881 |
| IVC | .817 | .832 |
| Adrenal (L) | .758 | .746 |
| Adrenal (R) | .695 | .710 |
| Colon | .856 | .864 |

## Cohorts: pancreas Dice by subgroup

| Factor | Group | n | TotalSeg. |
|---|---|---:|---:|
| CT phase | Arterial | 162 | .801 |
|  | Delay | 12 | .833 |
|  | Non-contrast | 134 | .781 |
|  | Venous | 593 | .839 |
|  | *adjusted p* |  | 0.007 |
| Scanner | GE | 42 | .694 |
|  | Philips | 11 | .781 |
|  | Siemens | 334 | .819 |
|  | *adjusted p* |  | 0.007 |
| Country | CH | 88 | .824 |
|  | CN | 60 | .803 |
|  | DE | 182 | .851 |
|  | FR | 19 | .865 |
|  | TR | 5 | .886 |
|  | US | 444 | .813 |
|  | multinational | 103 | .818 |
|  | *adjusted p* |  | < 0.001 |
| Slice thickness | ≤ 1.25 mm | 246 | .779 |
|  | 1.25–3 mm | 461 | .840 |
|  | > 3 mm | 194 | .841 |
|  | *adjusted p* |  | < 0.001 |
| Sex | F | 222 | .816 |
|  | M | 269 | .817 |
|  | *adjusted p* |  | 0.249 |
| Age | < 50 | 105 | .836 |
|  | 50–65 | 209 | .816 |
|  | > 65 | 165 | .821 |
|  | *adjusted p* |  | 1.000 |
| Tumour | no tumour | 750 | .826 |
|  | tumour | 151 | .811 |
|  | *adjusted p* |  | 0.152 |

Subgroup differences are observational: a scanner or site may differ in case mix (phase, tumour size), so read them as hypotheses.

## Reproduce

```console
$ sbatch benchmarks/pants/run_{nnunet,medformer,totalseg}.sbatch   # official inference
$ sbatch --export=ALL,MODELS=totalseg slurm/pants_eval.sbatch        # evaluation + report
$ python benchmarks/pants/write_results.py                          # these tables
```

MedFormer runs the official R-Super script with one upstream fix: its 3D padding swapped the slice and width axes, which made post-processing fail on 25 scans shorter than the training patch (`benchmarks/pants/patch_medformer.py`).
