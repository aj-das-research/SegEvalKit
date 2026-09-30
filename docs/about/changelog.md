# Changelog

## Unreleased

- **Evaluation levels:** dataset-level pooled ("micro") metrics (`res.pooled()`, `pooled.csv`: aggregated Dice,
  pooled lesion sensitivity / precision / F1, false-positive lesions per scan) from raw per-case counts now kept
  in the results; cohort analysis from case metadata (`segevalkit.cohort.cohort_summary`, `cohort_tests`,
  `plotting.cohort_plot`, `segevalkit cohort`).
- Union of per-structure files in label specs (`"a.nii.gz+b.nii.gz"`).

## 0.1.0 (2026-09-30)

First public release. Install with
`pip install "segevalkit[all] @ git+https://github.com/aj-das-research/SegEvalKit.git@v0.1.0"`.
[GitHub release](https://github.com/aj-das-research/SegEvalKit/releases/tag/v0.1.0)

**Coming next:** the full PanTS benchmark (nnU-Net, MedFormer and TotalSegmentator on the 901-case test set)
and the accompanying technical report.

- 53 metrics in 7 families (overlap, volume, distance, topology, detection, calibration, agreement) with a
  shared cached computation context and explicit empty-mask policies (`segevalkit`, `brats2023`,
  `metrics_reloaded`, `topcow`, `nan` / `nnunet`).
- Directed and pooled HD-percentiles, ASSD and MASD, voxel NSD, boundary IoU, clDice with a symmetric-object
  skeleton fallback, Betti-0/1/2 and Euler errors, lesion-wise recall / precision / F1 / Dice, panoptic quality,
  split / merge counts, soft Dice, AUROC, AUPRC, Brier, NLL, ECE with a boundary-band ROI.
- PyTorch GPU backend for confusion counts and exact surface distances, identical to the CPU backend.
- `Evaluator` for flat, folder and per-structure layouts with geometry checks, optional resampling and parallel
  workers; standard results folder (`per_case.csv`, `per_case_wide.csv`, `lesions.csv`, `summary.csv`,
  `meta.json`).
- Statistics: bootstrap CIs, paired tests with Holm / BH, effect sizes, rankings with bootstrap stability,
  Bland–Altman, ICC, stratification, presence detection.
- Plots and qualitative visualisation in a colour-vision-checked purple theme; self-contained HTML report.
- Metric recommender, synthetic perturbations, dataset presets, command line with a rich terminal interface.
- Conformance tests against MONAI, MedPy and DeepMind `surface-distance`.
