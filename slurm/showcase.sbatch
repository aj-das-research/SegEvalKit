#!/bin/bash
#SBATCH -J sek-showcase
#SBATCH -p cscc-cpu-p
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -o /l/users/%u/SegEvalKit/logs/%x-%j.out
set -euo pipefail
source "$HOME/projects/SegEvalKit/slurm/env.sh"
cd "$SEGEVALKIT_REPO"
python -W ignore benchmarks/showcase/make_showcase.py --out docs/assets/showcase
