# Testing & conformance

```console
$ pytest -q                        # everything (~1-2 min on 8 cores)
$ pytest -m reference              # conformance with MONAI / MedPy / DeepMind only
$ pytest -m "not reference"        # without the third-party reference libraries
$ pytest tests/test_backends.py    # CPU == PyTorch backend (GPU test runs when CUDA is present)
```

## Test layers

| File | What it guarantees |
|---|---|
| `test_overlap.py`, `test_distance.py`, `test_topology.py`, `test_detection.py`, `test_calibration.py` | analytic values on shapes with known answers (shifted cubes, concentric spheres, tori, hollow spheres, cut rings); empty-mask policies; monotonicity |
| `test_reference_implementations.py` | numerical agreement with MONAI 1.5, MedPy 0.5.2 and DeepMind `surface-distance`, isotropic and anisotropic spacing; pins down documented convention differences |
| `test_backends.py` | the PyTorch backend (CPU and CUDA) returns the NumPy/SciPy values to 1e-6 |
| `test_io_evaluator.py` | layouts, label forms, missing predictions, geometry checks and resampling, parallel = serial, per-label metrics, results round-trip |
| `test_stats_guide_synthetic.py` | multiple-testing corrections, CIs, paired comparison, rankings, ICC, Bland–Altman, presence detection, recommender logic, perturbations |
| `test_cli_plots.py` | every CLI command end to end, every plot and qualitative view renders |

## Writing tests for metrics

* Prefer inputs with a closed-form answer: a cube shifted by *k* voxels has HD = *k*·spacing; two concentric
  spheres have a known surface gap; a solid torus has Betti numbers (1, 1, 0).
* Always test both empty cases and the identical case.
* Test anisotropic spacing for anything in millimetres.
* When a convention differs from a reference library, assert the documented relationship (e.g. pooled HD95 ≤
  directed HD95) rather than loosening a tolerance until the test passes.

## Continuous integration

`.github/workflows/tests.yml` runs the suite on Python 3.10–3.12 with CPU PyTorch and the reference libraries on every
push and pull request. `.github/workflows/docs.yml` builds this site with `mkdocs build --strict` and deploys it.
