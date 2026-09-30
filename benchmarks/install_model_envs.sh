#!/bin/bash
# Install / repair the three baseline-model environments from a local wheelhouse.
# Some DANA compute nodes cannot reach download.pytorch.org, and an unpinned
# dependency (torchvision / timm / TotalSegmentator) otherwise pulls a CUDA 13
# torch from PyPI that cannot run on the A100 driver (CUDA <= 12.8).
# Wheels were fetched on the login node with `pip download --no-deps`.
# One environment per invocation, so the three can be built in parallel jobs.
set -euo pipefail
STORE=/l/users/$USER/SegEvalKit; TP=$STORE/third_party; ENVS=/nfs-stor/$USER/envs; W=$STORE/tmp/wheels
export TMPDIR=$STORE/tmp/${SLURM_JOB_ID:-x}; mkdir -p "$TMPDIR"; export PIP_CACHE_DIR=$STORE/tmp/pip-cache
source /apps/local/conda_init.sh
clean_torch() {  # remove any torch / CUDA-13 runtime pulled in earlier
    local py=$1
    "$py" -m pip uninstall -y -q torch torchvision triton cuda-toolkit cuda-bindings cuda-pathfinder \
        $("$py" -m pip list 2>/dev/null | grep -oE "^nvidia-[a-z0-9-]+" | tr "\n" " ") >/dev/null 2>&1 || true
}
MODEL=${1:?usage: install_model_envs.sh nnunet|totalseg|medformer}
case $MODEL in
nnunet)
# nnU-Net (py311, torch 2.5.1)
E=$ENVS/sek-nnunet/bin/python
clean_torch $E
$E -m pip install -q $W/torch-2.5.1+cu121-cp311-cp311-linux_x86_64.whl $W/torchvision-0.20.1+cu121-cp311-cp311-linux_x86_64.whl
$E -m pip install -q -c $W/constraints-cp311.txt -e "$TP/nnUNet-PanTS-regions" "timm>=1.0,<1.1" huggingface_hub
;;
totalseg)
# TotalSegmentator (py311, torch 2.5.1)
E=$ENVS/sek-totalseg/bin/python
[ -x "$E" ] || conda create -y -q -p "$ENVS/sek-totalseg" python=3.11
clean_torch $E
$E -m pip install -q $W/torch-2.5.1+cu121-cp311-cp311-linux_x86_64.whl $W/torchvision-0.20.1+cu121-cp311-cp311-linux_x86_64.whl
$E -m pip install -q -c $W/constraints-cp311.txt "TotalSegmentator==2.18.0"
TOTALSEG_HOME_DIR=$STORE/checkpoints/totalsegmentator "$ENVS/sek-totalseg/bin/totalseg_download_weights" -t total
;;
medformer)
# MedFormer (py310, torch 2.1.0)
E=$ENVS/sek-medformer/bin/python
[ -x "$E" ] || conda create -y -q -p "$ENVS/sek-medformer" python=3.10
clean_torch $E
$E -m pip install -q $W/torch-2.1.0+cu121-cp310-cp310-linux_x86_64.whl $W/torchvision-0.16.0+cu121-cp310-cp310-linux_x86_64.whl
$E -m pip install -q -c $W/constraints-cp310.txt -r "$TP/R-Super/rsuper_train/requirements.txt" "numpy<2"
;;
*) echo "unknown model $MODEL"; exit 2;;
esac
E=$ENVS/sek-$MODEL/bin/python
$E -c "import torch, torchvision; print('$MODEL', torch.__version__, torch.version.cuda, torchvision.__version__)"
