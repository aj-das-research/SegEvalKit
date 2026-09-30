# Contributing

Metrics, dataset presets, plots and fixes are welcome.

## Development setup

```console
$ git clone https://github.com/aj-das-research/SegEvalKit.git && cd SegEvalKit
$ pip install -e ".[all,dev,docs]"
$ pytest -q
$ mkdocs serve
```

## What to contribute

| Contribution | Essentials | Guide |
|---|---|---|
| Metric | `fn(ctx: PairContext, **params) -> float` with `@register_metric(...)` in its family module; LaTeX definition in the docstring; reuse `PairContext` caches or `ctx.memo`; explicit empty-mask behaviour; analytic and (if possible) conformance tests; a card on its family page | [Adding a metric](../developer/adding-a-metric.md) |
| Dataset preset | a `DatasetPreset` in `src/segevalkit/datasets.py` with label ids, reference layout, official metrics and parameters, citation; deviations in `notes` | [Adding a dataset preset](../developer/adding-a-dataset.md) |
| Plot | returns a figure, labels axes with `info.label`, uses the theme palette | [Plots, themes & visualisation](../developer/plots-and-themes.md) |

## Style

`ruff check src tests`, Google-style docstrings, and every number shown in the docs produced by a script in
`benchmarks/`.
