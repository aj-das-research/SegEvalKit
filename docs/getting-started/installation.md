# Installation

SegEvalKit needs Python ≥ 3.9. Core dependencies: NumPy, SciPy, nibabel, pandas, scikit-image, matplotlib,
PyYAML, tqdm, Jinja2 and rich.

=== "pip (from GitHub)"

    ```console
    $ pip install "segevalkit @ git+https://github.com/aj-das-research/SegEvalKit.git"
    ```

=== "With every extra"

    ```console
    $ pip install "segevalkit[all] @ git+https://github.com/aj-das-research/SegEvalKit.git@v0.1.0"
    ```

=== "Development"

    ```console
    $ git clone https://github.com/aj-das-research/SegEvalKit.git
    $ cd SegEvalKit
    $ pip install -e ".[all,dev,docs]"
    $ pytest                       # ~75 tests, incl. conformance with MONAI / MedPy / DeepMind
    $ mkdocs serve                 # this site at http://127.0.0.1:8000
    ```

## Optional extras

| Extra | Adds | Enables |
|---|---|---|
| `gpu` | `torch` | `device="cuda"`: confusion counts and surface distances on the GPU |
| `sitk` | `SimpleITK` | `.mha`, `.mhd`, `.nrrd` inputs |
| `interactive` | `plotly` | interactive 3D surface-distance maps (`backend="plotly"`) |
| `all` | all of the above | |
| `dev` | pytest, ruff | the test suite |
| `docs` | mkdocs-material, mkdocstrings | building this site |

!!! tip "PyTorch and CUDA"
    Install the PyTorch build matching your driver *before* SegEvalKit, e.g.
    `pip install torch --index-url https://download.pytorch.org/whl/cu128`. SegEvalKit never pins a CUDA version.

If `cc3d` (connected-components-3d) is installed it is used automatically; lesion labelling is 5–10× faster.

## Check the installation

```console
$ segevalkit --version
$ segevalkit metrics | head
$ python -c "import segevalkit as sek; print(len(sek.list_metrics()), 'metrics')"
```
