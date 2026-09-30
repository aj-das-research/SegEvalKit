<p align="center">
  <img src="docs/assets/logo.svg" width="96" alt="SegEvalKit logo">
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

## Quick start

```python
import segevalkit as sek

ev = sek.Evaluator(
    labels={"liver": 1, "tumour": 2},
    metrics=["default", "detection"],          # Dice, IoU, NSD, HD95, ASSD, PPV, TPR, RVD + lesion-wise
    params={"nsd": {"tolerance_mm": 2.0}},
)
res = ev.evaluate("predictions/", "labelsTr/", n_workers=8, out_dir="eval/")

res.summary()                                   # mean, median, IQR, 95 % bootstrap CI per structure and metric
sek.plotting.metric_distribution(res, "dice").savefig("dice.png")
sek.report.build_report(res, "eval/report.html", image_source="imagesTr/")
```

```console
$ segevalkit evaluate --pred predictions/ --ref labelsTr/ --labels liver=1,tumour=2 --out eval/ --report
$ segevalkit compare eval_A/ eval_B/ --out comparison/
$ segevalkit recommend --structure tubular --boundary-critical
$ segevalkit visualize --pred p.nii.gz --ref g.nii.gz --image ct.nii.gz --label 2 --kind triplanar --out case.png
```

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
