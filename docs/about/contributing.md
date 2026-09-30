# Contributing

Contributions of metrics, dataset presets, plots and fixes are welcome.

## Development setup

```console
$ git clone https://github.com/aj-das-research/SegEvalKit.git && cd SegEvalKit
$ pip install -e ".[all,dev,docs]"
$ pytest -q
$ mkdocs serve
```

## Adding a metric

1. Implement it as `fn(ctx: PairContext, **params) -> float` in the module of its family and decorate it with
   `@register_metric(...)`, giving display name, abbreviation, family, better direction, range, unit, inputs,
   a one-sentence summary and the primary reference. Put the LaTeX definition in the docstring.
2. Reuse the cached intermediates of `PairContext` (`counts`, `surface_distances`, `pred_components`,
   `pred_skeleton`...) or cache your own with `ctx.memo(key, fn)`.
3. Decide the empty-mask behaviour explicitly (`ctx.best_or_nan`, `ctx.distance_penalty`).
4. Add tests with analytically known values, and a conformance test if a reference implementation exists.
5. Document it on its family page. The [catalogue](../metrics/catalogue.md) updates itself from the registry.

## Adding a dataset preset

Add a `DatasetPreset` in `src/segevalkit/datasets.py` with the label ids, the reference layout, the official
metrics and parameters, and a citation. Record protocol deviations in `notes`.

## Style

`ruff check src tests`; Google-style docstrings; every number shown in the docs is produced by a script in
`benchmarks/`.
