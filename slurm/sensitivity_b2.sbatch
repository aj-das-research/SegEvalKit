#!/bin/bash
#SBATCH -J sek-sens-b2
#SBATCH -p cscc-cpu-p
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 06:00:00
#SBATCH -o /l/users/%u/SegEvalKit/logs/%x-%j.out
set -euo pipefail
source "$HOME/projects/SegEvalKit/slurm/env.sh"
cd "$SEGEVALKIT_REPO"
# Supplementary run: Betti-2 (cavity) error on the structures that get topology metrics.
python benchmarks/metric_sensitivity.py --ref-root data/raw/PanTS/LabelTe --out outputs/sensitivity \
    --n-cases 10 --workers 15 --only betti2_error \
    --structures aorta veins pancreatic_lesion kidney_left gall_bladder pancreas
