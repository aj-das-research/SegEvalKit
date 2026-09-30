#!/usr/bin/env bash
# Build the SegEvalKit conda environment on NFS (never on the login node:
# submit with `sbatch -p cscc-cpu-p slurm/build_env.sbatch`).
#
#   /nfs-stor/$USER/envs/segevalkit   conda env (many small files -> NFS)
#
# torch is pinned to the cu128 wheel index because the DANA A100 nodes run
# driver 570 (CUDA <= 12.8); cu130 wheels import but fail at the first kernel.
set -euo pipefail

ENV_PREFIX="${SEGEVALKIT_ENV:-/nfs-stor/$USER/envs/segevalkit}"
REPO="${SEGEVALKIT_REPO:-$HOME/projects/SegEvalKit}"

source /apps/local/conda_init.sh
if [ ! -d "$ENV_PREFIX" ]; then
    conda create -y -p "$ENV_PREFIX" python=3.11
fi
conda activate "$ENV_PREFIX"

python -m pip install --upgrade pip
python -m pip install "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e "$REPO[all,dev,docs]"
# Reference implementations, used only by the cross-validation test-suite.
python -m pip install "monai==1.5.1" "medpy==0.5.2" "surface-distance-based-measures" || true

python - <<'PY'
import torch, segevalkit
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("segevalkit", segevalkit.__version__)
PY
