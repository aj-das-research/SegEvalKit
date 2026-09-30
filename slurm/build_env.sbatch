#!/bin/bash
#SBATCH -J segevalkit-env
#SBATCH -p cscc-cpu-p
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o /l/users/%u/SegEvalKit/logs/%x-%j.out
set -euo pipefail
export TMPDIR=/l/users/$USER/SegEvalKit/tmp/$SLURM_JOB_ID; mkdir -p "$TMPDIR"
bash "$HOME/projects/SegEvalKit/scripts/setup_env.sh"
rm -rf "$TMPDIR"
