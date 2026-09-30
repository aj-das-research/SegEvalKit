# GPU & HPC

`device="cuda"` moves confusion counts and surface distances to the GPU; `n_workers` parallelises cases.

## What runs where

| Step | CPU | GPU (`device="cuda"`) |
|---|---|---|
| Reading NIfTI | ✓ (worker processes) | — |
| Confusion counts (all overlap / volume / agreement metrics) | ✓ | ✓ |
| Surface extraction | ✓ | — |
| Directed surface distances (HD, HD95, ASSD, MASD, NSD) | exact EDT on a crop | exact chunked nearest-neighbour search |
| Components, skeletons, topology | ✓ | — |

The GPU path gives **the same numbers** as the CPU path (tests assert 1e-6 relative agreement). Surface
distances are computed on the union bounding box of both masks plus one voxel, so a 512 × 512 × 600 CT with a
30 mL organ costs about as much as the organ's crop.

## Throughput tips

| Lever | Advice |
|---|---|
| Workers | Each worker loads and scores whole cases; `.nii.gz` decompression usually dominates, so use about one worker per core. With `device="cuda"`, 2–4 workers keep one GPU busy. |
| Metrics | Topology (skeletons, Betti numbers) is the most expensive family; request it only for tubular or hollow structures. |
| Memory | Peak ≈ two copies of the largest label volume per worker, plus crop-level distance maps. |

## Environment on a cluster

Create the environment once on a large project file system, with an explicit Python version, and install over
SSH if the cluster blocks HTTPS to GitHub. The first line loads the cluster's conda; its path is site-specific.

```console
$ source /apps/local/conda_init.sh              # your cluster's conda setup
$ conda create -p /path/to/envs/segevalkit python=3.11 pip -y
$ conda activate /path/to/envs/segevalkit
$ python -V                                     # check before installing
$ python -m pip install "segevalkit[all] @ git+ssh://git@github.com/aj-das-research/SegEvalKit.git@v0.1.0"
```

Install large environments (PyTorch wheels are several GB) from a CPU batch job rather than a login node, and
install the PyTorch build that matches the nodes' driver first (see [Installation](installation.md)).

## SLURM template

```bash title="evaluate.sbatch"
#!/bin/bash
#SBATCH -J segevalkit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 04:00:00
set -euo pipefail
export OMP_NUM_THREADS=1          # parallelism comes from the worker processes
python -c "import torch; assert torch.cuda.device_count() >= 1"
segevalkit evaluate --config eval.yaml --device cuda --workers 4
```

Use the devices the scheduler allocated (`cuda` / `cuda:0` inside the job); never set `CUDA_VISIBLE_DEVICES`
yourself.

## Reproducibility

`meta.json` records the SegEvalKit and Python versions and the full configuration. Bootstrap CIs and rankings
take a `seed` (default 0); perturbation studies are seeded per case and perturbation.
