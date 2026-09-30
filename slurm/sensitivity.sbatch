#!/bin/bash
#SBATCH -J sek-sensitivity
#SBATCH -p cscc-cpu-p
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH -o /l/users/%u/SegEvalKit/logs/%x-%j.out
set -euo pipefail
source "$HOME/projects/SegEvalKit/slurm/env.sh"
cd "$SEGEVALKIT_REPO"
python benchmarks/metric_sensitivity.py --ref-root data/raw/PanTS/LabelTe --out outputs/sensitivity \
    --n-cases 10 --workers 15
