#!/bin/bash
#SBATCH -J sek-pants-eval
#SBATCH -p cscc-cpu-p
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH -t 08:00:00
#SBATCH -o /l/users/%u/SegEvalKit/logs/%x-%j.out
# Evaluate the models whose predictions exist (all cases with a prediction), then build the report tables.
set -euo pipefail
source "$HOME/projects/SegEvalKit/slurm/env.sh"
cd "$SEGEVALKIT_REPO"
MODELS=${MODELS:-"nnunet medformer totalseg"}
python -W ignore benchmarks/pants/evaluate_models.py --models $MODELS --workers 30
python -W ignore benchmarks/pants/benchmark_report.py --models nnunet medformer totalseg
