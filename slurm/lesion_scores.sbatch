#!/bin/bash
#SBATCH -J sek-lesion-scores
#SBATCH -p cscc-cpu-p
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH -t 04:00:00
#SBATCH -o /l/users/%u/SegEvalKit/logs/%x-%j.out
# Lesion-only re-evaluation with per-lesion confidence scores (FROC / CPM, PR curves).
set -euo pipefail
source "$HOME/projects/SegEvalKit/slurm/env.sh"
cd "$SEGEVALKIT_REPO/benchmarks/pants"
MODELS=${MODELS:-"medformer"}
python -W ignore lesion_scores.py --models $MODELS --workers 30
