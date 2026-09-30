# GPU & HPC

## What runs where

| Step | CPU | GPU (`device="cuda"`) |
|---|---|---|
| Reading NIfTI | ✓ (worker processes) | — |
| Confusion counts (all overlap / volume / agreement metrics) | ✓ | ✓ |
| Surface extraction | ✓ | — |
| Directed surface distances (HD, HD95, ASSD, MASD, NSD) | exact EDT on a crop | exact chunked nearest-neighbour search |
| Components, skeletons, topology | ✓ | — |

The GPU path computes **the same numbers** as the CPU path: the test-suite asserts agreement to 1e-6 relative.
Surface distances are always computed on the union bounding box of the two masks (plus one voxel), so a 512 × 512
× 600 CT with a 30 mL organ costs about as much as the organ's crop.

## Throughput tips

* **Workers.** `n_workers` processes each load and score whole cases. Decompressing `.nii.gz` usually dominates,
  so use about as many workers as cores. With `device="cuda"`, 2–4 workers keep one GPU busy.
* **Metric choice.** Topology metrics (skeletons, Betti numbers) are the most expensive family; request them only
  for structures where topology matters (tubular, hollow).
* **Memory.** Peak memory is roughly two copies of the largest label volume per worker plus the crop-level
  distance maps.

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

Always compute on the devices your scheduler allocated: use `cuda` / `cuda:0` inside the job and never set
`CUDA_VISIBLE_DEVICES` yourself.

## Reproducibility

`meta.json` records the SegEvalKit version, Python version and the complete configuration. Bootstrap CIs and
rankings take a `seed` (default 0). Perturbation studies are seeded per case and perturbation.
