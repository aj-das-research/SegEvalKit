#!/bin/bash
# nnU-Net ResEnc-M (edomerli/nnUNet-PanTS-regions, official PanTS leaderboard checkpoint)
# on the PanTS test set, 4 GPUs = 4 independent SLURM steps, each predicting 1/4 of the cases.
#SBATCH -J sek-nnunet
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
ENV=/nfs-stor/$USER/envs/sek-nnunet
export nnUNet_raw=$STORE/tmp/nnunet_raw nnUNet_preprocessed=$STORE/tmp/nnunet_pre nnUNet_results=$STORE/tmp/nnunet_res
export nnUNet_extTrainer=$STORE/third_party/nnUNet-PanTS-regions/nnunetv2/custom_code
export TMPDIR=$STORE/tmp/$SLURM_JOB_ID; mkdir -p "$TMPDIR" "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
export OMP_NUM_THREADS=4 MPLCONFIGDIR=$TMPDIR/mpl
OUT=$STORE/outputs/predictions/nnunet_pants_regions
mkdir -p "$OUT"
N=${NGPU:-4}   # one SLURM step per allocated GPU
# predict_from_modelfolder has no -num_parts: shard the (symlinked) inputs into N folders instead.
for i in $(seq 0 $((N-1))); do
  mkdir -p "$STORE/data/pants_nnunet/shard$i"
done
ls "$STORE/data/pants_nnunet/imagesTs" | sort | awk -v n=$N -v d="$STORE/data/pants_nnunet" \
  '{ print d "/imagesTs/" $0, d "/shard" (NR-1)%n "/" $0 }' | while read src dst; do
  [ -e "$dst" ] || ln -s "$(readlink -f "$src")" "$dst"
done
for i in $(seq 0 $((N-1))); do
  srun --exact -n1 --gres=gpu:1 --cpus-per-task=8 --mem=58G \
    "$ENV/bin/nnUNetv2_predict_from_modelfolder" -m "$STORE/checkpoints/nnunet_pants_regions" \
      -i "$STORE/data/pants_nnunet/shard$i" -o "$OUT" -f all -chk checkpoint_best.pth \
      -npp 3 -nps 3 --disable_progress_bar --continue_prediction \
      > "$STORE/logs/nnunet-part$i-$SLURM_JOB_ID.log" 2>&1 &
done
wait
ls "$OUT"/*.nii.gz | wc -l
