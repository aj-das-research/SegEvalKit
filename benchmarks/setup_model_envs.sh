#!/usr/bin/env bash
# Build isolated environments for the official baseline models and download
# their official checkpoints (pinned revisions). Run on a CPU node:
#   sbatch -p cscc-cpu-p --cpus-per-task=8 --mem=32G -t 04:00:00 benchmarks/setup_model_envs.sh
#
#   envs         /nfs-stor/$USER/envs/sek-{nnunet,medformer,totalseg}
#   code         /l/users/$USER/SegEvalKit/third_party/<repo>   (fresh clones)
#   checkpoints  /l/users/$USER/SegEvalKit/checkpoints/<model>
set -euo pipefail
STORE=/l/users/$USER/SegEvalKit
TP=$STORE/third_party
CK=$STORE/checkpoints
ENVS=/nfs-stor/$USER/envs
mkdir -p "$CK"
export TMPDIR=$STORE/tmp/${SLURM_JOB_ID:-setup}; mkdir -p "$TMPDIR"
export PIP_CACHE_DIR=$STORE/tmp/pip-cache
source /apps/local/conda_init.sh

mkenv() {  # name python
    [ -d "$ENVS/$1" ] || conda create -y -q -p "$ENVS/$1" python="$2"
}

# 1) nnU-Net v2 fork used for the PanTS leaderboard submission (edomerli/nnUNet-PanTS-regions)
mkenv sek-nnunet 3.11
"$ENVS/sek-nnunet/bin/pip" install -q "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
"$ENVS/sek-nnunet/bin/pip" install -q -e "$TP/nnUNet-PanTS-regions" huggingface_hub

# 2) MedFormer (official R-Super / CBIM code, MrGiovanni/R-Super)
mkenv sek-medformer 3.10
"$ENVS/sek-medformer/bin/pip" install -q "torch==2.1.0" --index-url https://download.pytorch.org/whl/cu121
"$ENVS/sek-medformer/bin/pip" install -q -r "$TP/R-Super/rsuper_train/requirements.txt" huggingface_hub "numpy<2"

# 3) TotalSegmentator (official pip package, wasserth/TotalSegmentator)
mkenv sek-totalseg 3.11
"$ENVS/sek-totalseg/bin/pip" install -q "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
"$ENVS/sek-totalseg/bin/pip" install -q "TotalSegmentator==2.18.0"

# Official checkpoints (pinned revisions)
"$ENVS/sek-nnunet/bin/python" - <<PY
from huggingface_hub import snapshot_download
snapshot_download("edomerli/nnUNet-PanTS-regions", revision="2f3e1b57f194d658d884ee9fd451b0af05255d64",
                  local_dir="$CK/nnunet_pants_regions",
                  allow_patterns=["dataset.json", "plans.json", "dataset_fingerprint.json", "fold_all/checkpoint_best.pth"])
snapshot_download("AbdomenAtlas/MedFormerPanTS", revision="b854c8248af1e21f729dc2dfd489bd400472cf65",
                  local_dir="$CK/medformer_pants")
PY
export TOTALSEG_HOME_DIR=$CK/totalsegmentator
"$ENVS/sek-totalseg/bin/totalseg_download_weights" -t total

for e in sek-nnunet sek-medformer sek-totalseg; do
    "$ENVS/$e/bin/python" -c "import torch; print('$e', torch.__version__, torch.version.cuda)"
done
du -sh "$CK"/*
rm -rf "$TMPDIR"
