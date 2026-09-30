# Releasing & maintenance

## Versioning

SegEvalKit follows semantic versioning. The version lives in `src/segevalkit/__init__.py`.

| Change | Version bump |
|---|---|
| a metric's value changes for the same input (convention change, bug fix that alters numbers) | **minor** at least, and a changelog entry under "Changed numbers" |
| new metric, preset, plot or CLI option | minor |
| documentation, performance with identical numbers | patch |
| removal of a public name or results-format change | major |

## The results-format contract

`meta.json` carries `results_format`. Tools reading SegEvalKit results rely on the column names of `per_case.csv`
(`case_id, label, metric, value`), the `_` prefix for descriptive flags, and the columns of `summary.csv` and
`lesions.csv`. Changing them requires a new `results_format` and a loader that still reads the old one.

## Deprecation policy

Public names are deprecated for one minor release (with a `FutureWarning` naming the replacement) before removal.
Metric registry keys are never reused for a different definition.

## Release checklist

1. `pytest -q` passes, including the conformance tests.
2. `mkdocs build --strict` passes; benchmark figures regenerated if results changed.
3. Changelog updated; numbers-changing fixes called out explicitly.
4. Version bumped; tag `vX.Y.Z` pushed; docs deployed.
