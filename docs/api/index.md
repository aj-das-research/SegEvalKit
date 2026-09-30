# API reference

The public API is small and layered. Most users only need the first row.

| Layer | Entry points | Page |
|---|---|---|
| Dataset evaluation | `Evaluator`, `EvaluationResult`, `load_results` | [Evaluator & results](evaluator.md) |
| Single pair | `compute_metrics`, `PairContext`, `EmptyPolicy`, `register_metric` | [Metrics](metrics.md) |
| Files | `load_volume`, `Source`, `parse_labels`, `LabelSpec` | [I/O](io.md) |
| Analysis | `summarize`, `compare`, `rank_methods`, `ranking_stability`, `bland_altman`, `icc`, `presence_detection` | [Statistics](stats.md) |
| Figures | `plotting.*`, `viz.*`, `report.build_report` | [Plotting](plotting.md) · [Visualisation](viz.md) · [Report](report.md) |
| Guidance | `guide.recommend`, `datasets.get_dataset`, `synthetic.*` | [Guide](guide.md) · [Datasets](datasets.md) · [Synthetic](synthetic.md) |

Every page on this tab is generated from the docstrings in `src/segevalkit`.
