# Installation

SegEvalKit needs Python ≥ 3.9. The core depends only on NumPy, SciPy, nibabel, pandas, scikit-image,
matplotlib, PyYAML, tqdm and Jinja2.

=== "pip (from GitHub)"

    ```console
    $ pip install "segevalkit @ git+https://github.com/aj-das-research/SegEvalKit.git"
    ```

=== "With every extra"

    ```console
    $ pip install "segevalkit[all] @ git+https://github.com/aj-das-research/SegEvalKit.git"
    ```

=== "Development"

    ```console
    $ git clone https://github.com/aj-das-research/SegEvalKit.git
    $ cd SegEvalKit
    $ pip install -e ".[all,dev,docs]"
    $ pytest                       # ~70 tests, including conformance with MONAI / MedPy / DeepMind
    $ mkdocs serve                 # this documentation at http://127.0.0.1:8000
    ```

## Optional extras

| Extra | Adds | Enables |
|---|---|---|
| `gpu` | `torch` | `device="cuda"`: confusion counts and surface distances on the GPU |
| `sitk` | `SimpleITK` | `.mha`, `.mhd`, `.nrrd` inputs |
| `interactive` | `plotly` | interactive 3D surface-distance maps (`backend="plotly"`) |
| `all` | all of the above | |
| `dev` | pytest, ruff | the test-suite |
| `docs` | mkdocs-material, mkdocstrings | building this site |

!!! tip "PyTorch and CUDA"
    Install the PyTorch build that matches your driver *before* SegEvalKit, e.g.
    `pip install torch --index-url https://download.pytorch.org/whl/cu128`. SegEvalKit never pins a CUDA version.

`cc3d` (connected-components-3d) is used automatically when installed and makes lesion labelling 5–10× faster.

## Check the installation

```console
$ segevalkit --version
$ segevalkit metrics | head
$ python -c "import segevalkit as sek; print(len(sek.list_metrics()), 'metrics')"
```
