#!/bin/bash
# TotalSegmentator v2 (official package + official weights, task "total") on the PanTS test set.
#SBATCH -J sek-totalseg
#SBATCH -p cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH -t 24:00:00
#SBATCH --exclude=gpu-05,gpu-15,gpu-51,gpu-54
#SBATCH -o /l/users/%u/SegEvalKit/logs/%x-%j.out
set -euo pipefail
STORE=/l/users/$USER/SegEvalKit
ENV=/nfs-stor/$USER/envs/sek-totalseg
export TOTALSEG_HOME_DIR=$STORE/checkpoints/totalsegmentator
export TMPDIR=$STORE/tmp/$SLURM_JOB_ID; mkdir -p "$TMPDIR"
export OMP_NUM_THREADS=4 MPLCONFIGDIR=$TMPDIR/mpl
OUT=$STORE/outputs/predictions/totalsegmentator
N=${NGPU:-4}   # one SLURM step per allocated GPU
run_part() {  # relaunch a crashed step up to 5 times; every pipeline skips finished cases
  local i=$1 k
  for k in 1 2 3 4 5; do
    srun --exact -n1 --gres=gpu:1 --cpus-per-task=8 --mem=58G \
    "$ENV/bin/python" "$HOME/projects/SegEvalKit/benchmarks/pants/run_totalseg.py" \
      --images "$STORE/data/raw/PanTS/ImageTe" --out "$OUT" --parts $N --part $i \
      > "$STORE/logs/totalseg-part$i-$SLURM_JOB_ID-try$k.log" 2>&1 && return 0
    echo "part $i attempt $k failed; relaunching" >&2
  done
}
for i in $(seq 0 $((N-1))); do
  run_part $i &
done
wait
ls "$OUT" | wc -l
