# Plots, themes & visualisation

## The design system

Every figure (plots, overlays, the HTML report, this site and the terminal) shares one palette, defined once in
`segevalkit.plotting.theme`.

| Role | Values | Rule |
|---|---|---|
| Categorical (identity of a method or structure) | violet `#6d3fd6`, teal `#139a8a`, orange `#e2712f`, blue `#3f7fd6`, magenta `#cf3f8f`, gold `#b88a00` | fixed order, never cycled; colour follows the entity (`color_for(names)`); more than six series → facet |
| Sequential (magnitude) | purple ramp `#f5f1fe` → `#2e1766` | one hue, light to dark (`sequential_cmap()`) |
| Diverging (signed values, correlations) | orange ← grey → purple | neutral grey midpoint (`diverging_cmap()`) |
| Segmentation errors | TP violet, FN orange, FP teal | the three all-pairs colour-vision-safe slots |

The categorical order was chosen by running a colour-vision-deficiency validator over candidate orderings: every
adjacent pair stays separable under protan, deutan and tritan simulation (ΔE ≥ 11), and the first three slots are
separable in every pair, which is why they encode TP / FN / FP.

## Writing a new plot

Follow the conventions of `segevalkit/plotting/quantitative.py`:

```python
def my_plot(results, metric, label=None, ax=None, figsize=(5, 3.5)):
    info = get_metric(metric)                      # direction, unit, abbreviation
    df = _long(results)                            # one EvaluationResult or {name: result}
    colors = color_for(_methods(df))               # stable method -> colour mapping
    with theme():                                  # rcParams scoped to this figure
        fig, ax = _new_ax(ax, figsize)
        ...
        ax.set_ylabel(info.label)                  # "HD95 [mm] ↓"
        ax.set_title("...")
    return fig                                     # never plt.show(), never save inside
```

Checklist:

* accept one result **or** a mapping of results, and an optional `ax`;
* label axes with `info.label` so units and direction are always shown;
* one y-axis per chart; small multiples instead of dual axes;
* show individual cases where possible (dots, rainclouds, ECDFs), not only bars of means;
* legend for two or more series; no legend for one;
* return the figure; add it to `tests/test_cli_plots.py::test_all_plots_render`.

## Qualitative views

`segevalkit.viz` reorients every volume to RAS with its affine, displays slices in radiological convention with the
physical aspect ratio, and picks slices automatically (`pick_slice`: most error, largest reference, or centroid).
New views should reuse `_prep`, `_slice`, `_draw` and `_legend` so the orientation and colour conventions stay
identical across figures.

## Terminal

`segevalkit._console` provides the rich console, banner, progress bar and tables. Use `console.print` with the
`sek.*` styles (`sek.brand`, `sek.key`, `sek.muted`, `sek.ok`, `sek.warn`) rather than raw colours. Output degrades to
plain text when `NO_COLOR` or `SEGEVALKIT_PLAIN` is set or when output is not a terminal.
