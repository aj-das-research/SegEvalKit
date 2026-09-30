# HTML report

```console
$ segevalkit report eval/ --images imagesTs/ --gallery-metric dice --window abdomen
```

```python
from segevalkit.report import build_report
build_report(res, "eval/report.html", image_source="imagesTs/", gallery_metric="nsd", gallery_k=3)
```

The report is **one self-contained HTML file** (figures embedded, no server) with light and dark themes. It has
five parts:

1. **Setup & provenance.** Prediction and reference paths and layouts, structures, empty-mask policy,
   connectivity, alignment, device, metric parameters, and warnings for missing or failed cases.
2. **Summary per structure.** Mean ± SD, median [IQR], 95 % CI, *n* and NaN count for every metric, with the
   better-direction arrow and unit.
3. **Figures.** Raincloud distributions of the main metrics, metric vs structure size, failure quadrants
   (Dice vs HD95), volume agreement, metric correlation, and lesion detection by size.
4. **Worst cases.** Tri-planar error overlays of the *k* worst cases per structure for the chosen metric.
5. **Metric glossary.** For every metric in the report: definition, equation, range, direction, unit and
   primary reference, so a reader never has to guess what a number means.
