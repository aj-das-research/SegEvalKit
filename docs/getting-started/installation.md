# Installation

SegEvalKit needs Python ≥ 3.9 and is tested on **3.10, 3.11 and 3.12** (3.11 recommended). Core dependencies:
NumPy, SciPy, nibabel, pandas, scikit-image, matplotlib, PyYAML, tqdm, Jinja2 and rich.

## 1. Create a clean environment

Install into a fresh environment with a supported Python rather than into the system Python, and install with
`python -m pip` so that the package goes into the interpreter you just activated.

=== "conda"

    ```console
    $ conda create -n segevalkit python=3.11 pip -y
    $ conda activate segevalkit
    $ python -V                     # Python 3.11.x
    ```

=== "venv"

    ```console
    $ python3.11 -m venv .venv
    $ source .venv/bin/activate
    $ python -V                     # Python 3.11.x
    ```

=== "HPC cluster"

    Put the environment on a large project file system, not in a quota-limited home directory, and load the
    cluster's conda first. See [GPU & HPC](gpu-hpc.md#environment-on-a-cluster).

## 2. Install

=== "With every extra"

    ```console
    $ python -m pip install "segevalkit[all] @ git+https://github.com/aj-das-research/SegEvalKit.git@v0.1.0"
    ```

=== "Core only"

    ```console
    $ python -m pip install "segevalkit @ git+https://github.com/aj-das-research/SegEvalKit.git@v0.1.0"
    ```

=== "Over SSH"

    Where HTTPS to GitHub is blocked but an SSH key is set up (common on clusters):

    ```console
    $ python -m pip install "segevalkit[all] @ git+ssh://git@github.com/aj-das-research/SegEvalKit.git@v0.1.0"
    ```

=== "Development"

    ```console
    $ git clone https://github.com/aj-das-research/SegEvalKit.git
    $ cd SegEvalKit
    $ python -m pip install -e ".[all,dev,docs]"
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

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ERROR: Package 'segevalkit' requires a different Python` or `No matching distribution found` | Python older than 3.9 | Create an environment with Python 3.10–3.12 (step 1) |
| `pip` succeeds but `import segevalkit` fails | `pip` belongs to another interpreter | Use `python -m pip install ...` inside the activated environment; check `python -V` and `which python` |
| Long dependency resolution or a build of SciPy / scikit-image from source | Very new or very old Python without wheels | Use Python 3.10–3.12, which have prebuilt wheels |
| `git clone` hangs or `could not read Username` | HTTPS to GitHub blocked | Install over SSH (tab above) |
| `torch.cuda.is_available()` is `False` after installing `[gpu]` | CPU-only or wrong-CUDA PyTorch wheel | Install the PyTorch build for your driver first (tip above) |

