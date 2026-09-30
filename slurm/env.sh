# Common environment for SegEvalKit jobs. Sourced, not executed.
#   code   $HOME/projects/SegEvalKit          (small, versioned)
#   env    /nfs-stor/$USER/envs/segevalkit     (conda: many small files -> NFS)
#   data   /l/users/$USER/SegEvalKit           (Lustre: data, outputs, logs)
export SEGEVALKIT_REPO="${SEGEVALKIT_REPO:-$HOME/projects/SegEvalKit}"
export SEGEVALKIT_ENV="${SEGEVALKIT_ENV:-/nfs-stor/$USER/envs/segevalkit}"
export SEGEVALKIT_STORE="${SEGEVALKIT_STORE:-/l/users/$USER/SegEvalKit}"
source /apps/local/conda_init.sh
conda activate "$SEGEVALKIT_ENV"
export HF_HOME=/l/users/$USER/vlm-eval-suit/.cache/huggingface
unset TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE PYTORCH_TRANSFORMERS_CACHE PYTORCH_PRETRAINED_BERT_CACHE
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1   # parallelism comes from worker processes
export TMPDIR="$SEGEVALKIT_STORE/tmp/${SLURM_JOB_ID:-local}"; mkdir -p "$TMPDIR"
