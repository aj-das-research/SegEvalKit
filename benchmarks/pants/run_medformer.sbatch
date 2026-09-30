#!/bin/bash
# MedFormer (AbdomenAtlas/MedFormerPanTS, official checkpoint; official inference code from
# MrGiovanni/R-Super) on the PanTS test set, 4 GPUs = 4 SLURM steps over --parts/--current_part.
#SBATCH -J sek-medformer
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
ENV=/nfs-stor/$USER/envs/sek-medformer
CK=$STORE/checkpoints/medformer_pants
OUT=$STORE/outputs/predictions/medformer_pants
export TMPDIR=$STORE/tmp/$SLURM_JOB_ID; mkdir -p "$TMPDIR" "$OUT"
export OMP_NUM_THREADS=4 MPLCONFIGDIR=$TMPDIR/mpl
cd "$STORE/third_party/R-Super/rsuper_train"
# The official script needs an ids CSV (BDMAP_ID column) to read <case>/ct.nii.gz folders.
IDS=$STORE/data/pants_test_ids.csv
{ echo BDMAP_ID; ls "$STORE/data/raw/PanTS/ImageTe" | sort; } > "$IDS"
N=${NGPU:-4}   # one SLURM step per allocated GPU
run_part() {  # relaunch a crashed step up to 5 times; every pipeline skips finished cases
  local i=$1 k
  for k in 1 2 3 4 5; do
    srun --exact -n1 --gres=gpu:1 --cpus-per-task=8 --mem=58G \
    "$ENV/bin/python" predict_abdomenatlas.py --load "$CK/pants_pancreas_release/fold_0_latest.pth" \
      --img_path "$STORE/data/raw/PanTS/ImageTe" --class_list "$CK/labels_pants.yaml" \
      --save_path "$OUT" --ids "$IDS" --organ_mask_on_lesion --save_probabilities_lesions --organs_with_tumor pancreas \
      --parts $N --current_part $i --gpu 0 \
      > "$STORE/logs/medformer-part$i-$SLURM_JOB_ID-try$k.log" 2>&1 && return 0
    echo "part $i attempt $k failed; relaunching" >&2
  done
}
for i in $(seq 0 $((N-1))); do
  run_part $i &
done
wait
ls "$OUT" | wc -l
