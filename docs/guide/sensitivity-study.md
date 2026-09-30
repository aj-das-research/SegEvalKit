# Sensitivity study: what does each metric see?

Metric pitfalls are usually illustrated with toy shapes. Here they are measured on **real CT anatomy**: reference
masks from the PanTS test set, disturbed by errors of known type and physical size, scored with every metric.
The question for each (metric, error) pair is simple: *when the error grows, does the metric notice?*

## Design

**Structures.** Seven PanTS structures chosen to span size and shape: liver (large organ), left kidney (compact
organ), pancreas (elongated organ), gallbladder (small organ), aorta (tubular), veins (branching tubular) and
pancreatic lesion (small, irregular). Ten randomly chosen test cases per structure (lesions: ten cases that
contain one), at native resolution and spacing, cropped with an 80 mm margin.

**Errors.** Eight perturbations from `segevalkit.synthetic`, each at 4–5 magnitudes in physical units:

| Perturbation | What it simulates | Magnitudes |
|---|---|---|
| Erosion / dilation | uniform under- / over-segmentation of the boundary | 0, 1, 2, 3, 5 mm |
| Boundary noise | smooth random in-and-out boundary jitter at constant volume | 0, 1, 2, 3, 5 mm |
| Shift | registration-like rigid displacement | 0, 1, 2, 4, 8 mm |
| FP islands | spurious 4 mm blobs within 60 mm of the structure | 0, 1, 2, 4, 8 blobs |
| Missing slab | part of the structure missed | 0, 5, 10, 20, 40 % |
| Internal holes | cavities of 3 mm radius inside the structure | 0, 1, 2, 4, 8 holes |
| Cut | a planar cut that breaks connectivity with almost no volume change | 0, 2, 4, 8 mm |

**Responsiveness.** For each structure-case, the Spearman correlation between error magnitude and metric value;
the matrix reports the median |ρ| over structure-cases (1 = the metric tracks the error perfectly, 0 = it does not
move). Topology metrics (clDice, Betti errors) were computed for the tubular structures, the lesion, and three
organs as controls.

```console
$ python benchmarks/metric_sensitivity.py --ref-root data/raw/PanTS/LabelTe --out outputs/sensitivity --n-cases 10
$ python benchmarks/plot_sensitivity.py --csv outputs/sensitivity/sensitivity.csv --out docs/assets/figures
```

## Results

!!! note "Pending"
    The study is running on the cluster; the figures and their discussion will appear here when it completes.
