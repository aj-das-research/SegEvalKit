---
hide:
  - navigation
  - toc
---

<div class="sek-hero" markdown>

# SegEvalKit

Holistic, literature-grounded evaluation of **volumetric medical image segmentation**. Point it at a folder of
predictions and a folder of references (CT or MR, NIfTI) and get every metric that matters, with honest
conventions, statistics you can defend, and figures you can publish.

[Get started](getting-started/quickstart.md){ .md-button .md-button--primary }
[Choose your metrics](guide/choosing.md){ .md-button }
[Metric catalogue](metrics/catalogue.md){ .md-button }

<div class="sek-badges">
<span class="sek-badge">50+ metrics · 7 families</span>
<span class="sek-badge">NIfTI / ITK / NumPy</span>
<span class="sek-badge">GPU surface distances</span>
<span class="sek-badge">bootstrap CIs · paired tests · ranking</span>
<span class="sek-badge">conformance-tested vs MONAI · MedPy · DeepMind</span>
</div>
</div>

## Why SegEvalKit?

A Dice score alone does not tell you whether a model finds small tumours, whether it breaks vessels, whether its
boundaries are clinically acceptable, whether its probabilities can be trusted, or whether it really beats the
baseline. Answering those questions takes many metrics, and existing tools disagree on how to compute them:
HD95 has two definitions in use, "ASSD" means different things in different libraries, and empty masks are
handled in at least four ways. SegEvalKit brings the metrics together, makes every convention explicit, and
tests them against the reference implementations.

<div class="grid cards" markdown>

-   :material-shape-outline:{ .lg } **Every family of metric**

    ---

    Overlap, volume, surface distance, topology, lesion-wise detection, calibration and agreement, with one
    shared computation context so twenty metrics cost little more than one.

    [:octicons-arrow-right-24: Metrics](metrics/index.md)

-   :material-compass-outline:{ .lg } **Knows which metric to use**

    ---

    A problem-fingerprint recommender following Metrics Reloaded, a pitfalls catalogue, and a sensitivity
    study on real anatomy that shows what each metric does and does not see.

    [:octicons-arrow-right-24: Choosing metrics](guide/index.md)

-   :material-folder-table-outline:{ .lg } **A standard input and output**

    ---

    Reads the three layouts used by virtually every dataset and model (flat, folder, per-structure), checks
    geometry, and writes one documented results format.

    [:octicons-arrow-right-24: Data & output format](getting-started/data-format.md)

-   :material-chart-box-outline:{ .lg } **Statistics you can defend**

    ---

    Bootstrap confidence intervals, paired Wilcoxon tests with Holm correction, effect sizes, challenge-style
    rankings with bootstrap stability, Bland–Altman and ICC.

    [:octicons-arrow-right-24: Statistics](analysis/statistics.md)

-   :material-image-filter-center-focus-weak:{ .lg } **See the errors**

    ---

    Colour-vision-safe error overlays, tri-planar views, error projections, 3D surface-distance maps, worst-case
    galleries and a self-contained HTML report.

    [:octicons-arrow-right-24: Visualisation](analysis/qualitative.md)

-   :material-lightning-bolt-outline:{ .lg } **Fast on big volumes**

    ---

    Surface distances on crops, exact PyTorch nearest-neighbour search on the GPU, and parallel case workers
    for thousands of CT scans.

    [:octicons-arrow-right-24: GPU & HPC](getting-started/gpu-hpc.md)

</div>

## Thirty-second tour

=== "Python"

    ```python
    import segevalkit as sek

    ev = sek.Evaluator(
        labels={"liver": 1, "tumour": 2},             # ids in the label maps
        metrics=["default", "detection"],              # Dice, NSD, HD95, ASSD, ... + lesion-wise
        params={"nsd": {"tolerance_mm": 2.0}},
    )
    res = ev.evaluate("predictions/", "labelsTr/", n_workers=8, out_dir="eval/")

    res.summary()                                      # mean, median, IQR, 95 % CI per structure & metric
    sek.plotting.metric_distribution(res, "dice")      # raincloud plot per structure
    sek.report.build_report(res, "eval/report.html", image_source="imagesTr/")
    ```

=== "Command line"

    ```console
    $ segevalkit evaluate --pred predictions/ --ref labelsTr/ \
          --labels liver=1,tumour=2 --metrics default,detection \
          --out eval/ --workers 8 --report --images imagesTr/
    $ segevalkit compare eval_modelA/ eval_modelB/ --out comparison/
    $ segevalkit recommend --structure small_lesion --multi-instance
    ```

=== "One pair of arrays"

    ```python
    from segevalkit import compute_metrics

    compute_metrics(pred, ref, ["dice", "hd95", "nsd", "cldice"], spacing=(0.8, 0.8, 2.5))
    # {'dice': 0.91, 'hd95': 3.2, 'nsd': 0.94, 'cldice': 0.88}
    ```

## What is inside

| | |
|---|---|
| **Metrics** | Dice, IoU, VOE, precision, recall, specificity, F-beta, Tversky, MCC, kappa · volumes, RVD, AVD, VS · HD, HD95 / HDq (directed or pooled), ASSD, MASD, NSD, boundary IoU, centroid distance · clDice, Betti-0/1/2 and Euler errors · lesion recall / precision / F1, lesion-wise Dice, panoptic quality, split / merge counts · soft Dice, AUROC, AUPRC, Brier, NLL, ECE · MI, VI, ARI, GCE |
| **Conventions** | explicit empty-mask policies (incl. BraTS 2023 and Metrics Reloaded presets), 6/18/26 connectivity, spacing-aware everything, geometry checks with optional resampling |
| **Statistics** | bootstrap CIs, Wilcoxon / t / permutation tests with Holm / BH, rank-biserial effect sizes, aggregate-then-rank and rank-then-aggregate, Kendall-tau ranking stability, Bland–Altman, ICC(2,1)/(3,1), size stratification, patient-level presence detection |
| **Figures** | raincloud distributions, ECDFs, per-case heatmaps, metric-vs-size, metric correlation, volume agreement, Bland–Altman, metric profiles, comparison forests, ranking blob plots, reliability diagrams, lesion detection by size, failure quadrants, sensitivity curves |
| **Qualitative** | error overlays (TP / FN / FP), contours, tri-planar views, montages, 3D error projections, surface-distance meshes (static and interactive) |
| **Datasets** | presets with official protocols for PanTS, TotalSegmentator, BTCV, AMOS, FLARE, MSD, KiTS23, BraTS 2023, ISLES'22, autoPET, TopCoW |

## Cite

If SegEvalKit helps your research, please cite it, and cite the original papers of the metrics you report. Each metric's page lists them. See [Citation](about/citation.md).
