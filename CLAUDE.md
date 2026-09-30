# SegEvalKit: working notes for Claude

Library for evaluating volumetric (CT/MR, NIfTI) segmentation: metrics, standard I/O, stats, plots,
qualitative viz, docs site, arXiv report. Package in `src/segevalkit`, docs in `docs/` (MkDocs Material,
**purple theme everywhere**), paper in `paper/` (nested Overleaf repo, gitignored).

## Paths
| What | Where |
|---|---|
| Code (git: `git@github.com:aj-das-research/SegEvalKit.git`, SSH only) | `~/projects/SegEvalKit` |
| Main env | `/nfs-stor/$USER/envs/segevalkit` (`slurm/env.sh` activates it) |
| Model envs | `/nfs-stor/$USER/envs/sek-{nnunet,medformer,totalseg}` (`benchmarks/setup_model_envs.sh`) |
| Data / outputs / logs / checkpoints / third_party | `/l/users/$USER/SegEvalKit/...` (symlinked `data/`, `outputs/`, `logs/`) |
| Raw datasets (read-only symlinks) | `data/raw/{PanTS,TotalSegmentator,MSD_Task03_Liver,MSD_Task10_Colon}` |
| Paper (Overleaf project `6abcaa8acdc298d46a0558d3`, branch `main`) | `paper/` with its own credential store |
| Secrets | `~/.config/research-credentials/segevalkit.env` (never print/commit) |

## Rules
- Fresh project: never reuse earlier predictions/models/logs from other projects (e.g. `~/projects/jhu`).
  Model outputs come only from official code + official checkpoints run here.
- DANA cluster: no compute on login nodes; tests via `srun -p cscc-cpu-p ...`; GPU via `sbatch` (see `slurm/`).
- Metric conventions are documented in `docs/guide/conventions.md` and pinned by
  `tests/test_reference_implementations.py`; change both together.
- Every empirical number in docs/paper comes from files under `outputs/` produced by scripts in `benchmarks/`.

## Commands
```bash
srun -p cscc-cpu-p --cpus-per-task=8 --mem=32G -t 00:25:00 $ENV/bin/python -m pytest -q   # tests
$ENV/bin/mkdocs build --strict                                                         # docs
git -C paper pull --ff-only && git -C paper push origin main                            # Overleaf
```
