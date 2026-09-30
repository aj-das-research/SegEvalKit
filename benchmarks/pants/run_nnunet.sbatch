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
export OMP_NUM_THREADS=4
OUT=$STORE/outputs/predictions/nnunet_pants_regions
mkdir -p "$OUT"
N=4
for i in $(seq 0 $((N-1))); do
  srun --exact -n1 --gres=gpu:1 --cpus-per-task=8 --mem=58G \
    "$ENV/bin/nnUNetv2_predict_from_modelfolder" -m "$STORE/checkpoints/nnunet_pants_regions" \
      -i "$STORE/data/pants_nnunet/imagesTs" -o "$OUT" -f all -chk checkpoint_best.pth \
      -npp 3 -nps 3 -num_parts $N -part_id $i --disable_progress_bar --continue_prediction \
      > "$STORE/logs/nnunet-part$i-$SLURM_JOB_ID.log" 2>&1 &
done
wait
ls "$OUT"/*.nii.gz | wc -l
