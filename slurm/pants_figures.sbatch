#!/bin/bash
# Render the PanTS benchmark figures (docs PNG + paper PDF): sbatch slurm/pants_figures.sbatch
#SBATCH -J sek-figs
#SBATCH -p cscc-cpu-p
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -o /l/users/abhijit.das/SegEvalKit/logs/%x-%j.out
cd /home/abhijit.das/projects/SegEvalKit
export MPLCONFIGDIR=/l/users/abhijit.das/SegEvalKit/tmp/mpl OMP_NUM_THREADS=2
/nfs-stor/abhijit.das/envs/segevalkit/bin/python -W ignore benchmarks/pants/make_figures.py --workers 14 ${ARGS:-}
