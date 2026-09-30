<p align="center">
  <img src="docs/assets/icon.svg" width="88" alt="SegEvalKit logo">
</p>

<h1 align="center">SegEvalKit</h1>

<p align="center">
  <b>Holistic, literature-grounded evaluation of volumetric (CT / MR) medical image segmentation.</b><br>
  50+ metrics · explicit conventions · GPU surface distances · statistics & ranking · purple-themed plots, overlays and reports
</p>

<p align="center">
  <a href="https://aj-das-research.github.io/SegEvalKit/"><img alt="docs" src="https://img.shields.io/badge/docs-SegEvalKit-6d3fd6?style=flat-square"></a>
  <img alt="python" src="https://img.shields.io/badge/python-3.9%2B-7c4ee4?style=flat-square">
  <img alt="license" src="https://img.shields.io/badge/license-Apache--2.0-9a72ee?style=flat-square">
  <img alt="tests" src="https://img.shields.io/badge/conformance-MONAI%20%C2%B7%20MedPy%20%C2%B7%20DeepMind-5a2fb8?style=flat-square">
</p>

---

A Dice score alone does not tell you whether a model finds small tumours, breaks vessels, draws clinically
acceptable boundaries, outputs trustworthy probabilities, or really beats the baseline. Answering those questions
takes many metrics, and the existing tools disagree on how to compute them: HD95 has two definitions in use,
"ASSD" means different things in different libraries, and empty masks are handled in at least four ways.
**SegEvalKit** brings the metrics together behind one standard input/output interface, makes every convention
explicit and configurable, tests them against the reference implementations, and adds the statistics and figures
needed to report results you can defend.

## Highlights

| | |
|---|---|
| **Metrics** | 51 metrics in 7 families: overlap, volume, surface distance, topology, lesion-wise detection, calibration, agreement. One cached computation context per pair. |
| **Standard I/O** | Reads the three layouts used by virtually every dataset and model (flat, folder, per-structure NIfTI), checks geometry, writes one documented results format (`per_case.csv`, `summary.csv`, `lesions.csv`, `meta.json`). |
| **Conventions** | Explicit empty-mask policies (incl. BraTS 2023 / Metrics Reloaded presets), directed vs pooled HD95, ASSD vs MASD, connectivity, NSD tolerance, all recorded with the results. |
| **Conformance** | Test-suite checks agreement with MONAI, MedPy and DeepMind `surface-distance`, and CPU ≡ GPU. |
| **Statistics** | Bootstrap CIs, paired Wilcoxon/t/permutation tests with Holm/BH correction and effect sizes, challenge-style rankings with bootstrap stability, Bland–Altman, ICC, size stratification, patient-level presence detection. |
| **Figures** | Raincloud plots, ECDFs, heatmaps, metric-vs-size, metric correlation, volume agreement, comparison forests, ranking blob plots, reliability diagrams, detection-by-size, failure quadrants. |
| **Qualitative** | Colour-vision-safe TP/FN/FP overlays, contours, tri-planar views, montages, 3D error projections, surface-distance meshes (static and interactive), worst-case galleries. |
| **Guidance** | A Metrics-Reloaded-style recommender (`segevalkit recommend`), a pitfalls catalogue, and a sensitivity study on real CT anatomy. |
| **Datasets** | Presets with official protocols for PanTS, TotalSegmentator, BTCV, AMOS, FLARE22, MSD, KiTS23, BraTS 2023, ISLES'22, autoPET, TopCoW. |

## Install

```bash
pip install "segevalkit[all] @ git+https://github.com/aj-das-research/SegEvalKit.git"
```

## A walkthrough on real data

Every output below is real: three official models (nnU-Net, MedFormer, TotalSegmentator, run from their own code and
checkpoints) on 8 PanTS test CTs chosen to span tumour sizes and orientations. Eight cases illustrate the library;
they are not a benchmark.

**1. Describe what to evaluate.** Structures are named once; each side says where to find them. Here the reference
has one file per structure and nnU-Net writes one multi-label map.

```python
import segevalkit as sek

ev = sek.Evaluator(
    labels={
        "pancreas":          {"ref_file": "pancreas.nii.gz+pancreatic_lesion.nii.gz", "pred": [17, 18, 19, 20, 21, 28]},
        "pancreatic_lesion": {"ref_file": "pancreatic_lesion.nii.gz", "pred": 28, "metrics": ["default", "detection"]},
        "liver":             {"ref_file": "liver.nii.gz", "pred": 14},
    },
    metrics="default", params={"nsd": {"tolerance_mm": 2.0}}, min_lesion_voxels=10,
)
```

**2. Evaluate a folder, then summarise.** One call scores every case; `summary()` gives mean, median and a 95 %
bootstrap confidence interval per structure and metric.

```python
res = ev.evaluate("predictions/nnunet/", "PanTS/LabelTe/", n_workers=8, out_dir="eval/nnunet")
res.summary().query("metric in ['dice', 'nsd', 'hd95']")
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

The lesion rows show why the empty-mask policy matters: in tumour-free patients a spurious lesion prediction scores
Dice 0 and the image-diagonal HD95 penalty, which is what drags the lesion HD95 to hundreds of millimetres.

**3. Find the failures.** Per-case values and the worst cases are one call away.

```python
res.worst_cases("dice", "pancreas", k=3)
```

```text
       case_id  dice
PanTS_00009746 0.733
PanTS_00009287 0.862
PanTS_00009322 0.881
```

**4. Look at lesions, not voxels.** The lesion table shows what the Dice average hides: nnU-Net detected 3 of the
7 reference lesions in these cases and missed a 23.7 mL tumour entirely.

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

**5. Compare and rank models** with paired tests (Holm-corrected) and challenge-style rankings.

```python
from segevalkit.stats import compare, rank_methods
compare(res_nnunet, res_medformer, metrics=["dice", "nsd", "hd95"], labels=["pancreas"])
rank_methods({"nnU-Net": res_nnunet, "MedFormer": res_medformer, "TotalSegmentator": res_ts}, "dice", label="pancreas")
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

**6. Plot and look.** Figures use one colour-vision-safe palette: violet agreement, orange missed, teal added.

```python
sek.plotting.metric_distribution({"nnU-Net": r1, "MedFormer": r2, "TotalSegmentator": r3}, "dice", labels=organs)
sek.viz.triplanar(ct, pred_pancreas, ref_pancreas, affine=ref.affine, window="pancreas")
```

<p align="center">
  <img src="docs/assets/showcase/model_comparison.png" width="92%" alt="Three models on one slice">
</p>
<p align="center">
  <img src="docs/assets/showcase/dist_dice.png" width="92%" alt="Dice per structure and model">
</p>
<p align="center">
  <img src="docs/assets/showcase/triplanar_pancreas.png" width="92%" alt="Tri-planar error view">
</p>

**7. Report.** One self-contained HTML file with provenance, tables, figures, the worst cases and a metric glossary:
[open the sample report](https://aj-das-research.github.io/SegEvalKit/assets/showcase/report_nnunet.html).

```python
sek.report.build_report(res, "eval/nnunet/report.html", image_source="PanTS/ImageTe/")
```

The same workflow from the command line:

```console
$ segevalkit evaluate --config eval_nnunet.yaml --report     # the labels above, as YAML
$ segevalkit compare eval/nnunet eval/medformer --out comparison/
$ segevalkit recommend --structure small_lesion --multi-instance
```

<p align="center"><img src="docs/assets/terminal/recommend.svg" width="92%" alt="segevalkit recommend"></p>

## Documentation

Full documentation, including the metric catalogue with equations and plain-language explanations, the decision
guide, the pitfalls catalogue, the conformance report and the benchmark results, is at
**https://aj-das-research.github.io/SegEvalKit/** (source in [`docs/`](docs)).

The two literature studies behind the library are in [`research/`](research):
[metrics review](research/metrics_literature_review.md) · [datasets & benchmarks survey](research/datasets_benchmarks_survey.md).

## Repository layout

```text
src/segevalkit/     library: metrics/, io/, stats/, plotting/, viz/, report/, guide/, datasets.py, synthetic.py, cli.py
tests/              unit, conformance (MONAI / MedPy / DeepMind) and end-to-end tests
docs/               MkDocs Material site (purple theme)
benchmarks/         reproducible studies: metric sensitivity on real anatomy, PanTS benchmark with official models
research/           literature review and dataset survey
slurm/, scripts/    cluster launchers and environment setup
```

## Citation

```bibtex
@software{das2026segevalkit,
  title  = {SegEvalKit: Holistic Evaluation of Volumetric Medical Image Segmentation},
  author = {Das, Abhijit},
  year   = {2026},
  url    = {https://github.com/aj-das-research/SegEvalKit}
}
```

Please also cite the original papers of the metrics and datasets you use; each metric's documentation page lists them.

## License

Apache-2.0. Dataset presets describe public datasets; each dataset keeps its own licence.
