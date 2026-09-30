# Topology & centreline metrics

Overlap and distance metrics can be excellent for a vessel tree broken into ten pieces, or for a hollow organ
whose lumen has been filled in. Topological metrics count what the shape **is**: connected pieces, tunnels and
enclosed cavities. They matter whenever connectivity carries clinical meaning: vessels, airways, ducts, the
circle of Willis, cortical surfaces and thin-walled organs.

They are also blind to much of what the other families measure. A Betti error of 0 says nothing about overlap,
and errors in different places can cancel. Report them next to an overlap metric, never instead of one.

## Notation

For a 3D binary object \(X\) the Betti numbers are

| Symbol | Counts | Computed as |
|---|---|---|
| \(\beta_0(X)\) | connected components | 26-connected foreground components |
| \(\beta_1(X)\) | independent tunnels / handles (loops) | \(\beta_0 + \beta_2 - \chi\) |
| \(\beta_2(X)\) | enclosed cavities | 6-connected background components minus the outside one |
| \(\chi(X)\) | Euler characteristic | \(\beta_0 - \beta_1 + \beta_2\), from `skimage.measure.euler_number` (26-connectivity) |

The (26, 6) foreground/background adjacency pair is the standard well-composed choice for 3D digital topology
(Kong & Rosenfeld 1989). The mask is padded with background, so exactly one background component (the outside)
is unbounded. These conventions are fixed: the `connectivity` argument of `PairContext` / `Evaluator`
affects lesion components for [detection metrics](detection.md), not Betti numbers.

A solid ball has \((\beta_0,\beta_1,\beta_2) = (1, 0, 0)\), a ring (torus) \((1, 1, 0)\) and a hollow shell
\((1, 0, 1)\). `segevalkit.metrics.betti_numbers(mask)` returns the triple for any mask.

\(S_X\) denotes the topological skeleton of \(X\): 3D thinning of Lee et al. (1994) as implemented in
scikit-image, with one safeguard: a non-empty component whose skeleton vanishes under thinning (for example a
bar with an even, symmetric cross-section) keeps its most interior voxel.

## At a glance

| Metric | Key | Detects | Misses |
|---|---|---|---|
| [β₀ error](#betti0_error) | `betti0_error` | fragmentation, spurious islands | where the break is; matching of pieces |
| [β₁ error](#betti1_error) | `betti1_error` | opened or falsely closed loops | location; a lost and a gained loop cancel |
| [β₂ error](#betti2_error) | `betti2_error` | filled-in lumens, spurious voids | location |
| [Euler error](#euler_error) | `euler_error` | any net topology change | errors of different kind cancel |
| [clDice](#cldice) | `cldice` | incomplete / leaking centrelines | radius errors; short gaps barely move it |

## Metrics

<div class="sek-metric" markdown>

### Betti-0 error (components) (β₀ err) {#betti0_error}

<div class="sek-meta"><span class="sek-chip">key: <code>betti0_error</code></span><span class="sek-chip">range [0, ∞)</span><span class="sek-chip teal">lower is better</span><span class="sek-chip orange">count</span></div>

\[
\varepsilon_{\beta_0} = \lvert\beta_0(P) - \beta_0(G)\rvert
\]

**In plain words:** the difference in the number of connected pieces. It counts fragmentation (one vessel broken
into two) and spurious islands.

**Use it when:** connectivity matters: vessel trees, airways, tubular organs, or any structure that should be a
single piece. A 2-voxel gap in a 56-voxel vessel leaves Dice at 0.98 and HD95 at 0 mm, but gives
\(\varepsilon_{\beta_0} = 1\) (see the [example](#code)).

**Watch out for:** it compares counts only: one missing piece and one spurious piece elsewhere give 0. It is
sensitive to single-voxel noise; a tiny false-positive speck counts as much as a severed artery.

**Empty masks:** both empty: 0 (`"best"`) or NaN. Exactly one empty: the Betti number of the other mask (the empty
mask has \(\beta = (0,0,0)\)).

**Parameters:** none.

**Reference:** Hu X, Li F, Samaras D, Chen C. Topology-preserving deep image segmentation. *NeurIPS 2019*.
[arXiv:1906.05404](https://arxiv.org/abs/1906.05404). Yang K, Musio F, Ma Y, et al. TopCoW: benchmarking
topology-aware anatomical segmentation of the circle of Willis. [arXiv:2312.17670](https://arxiv.org/abs/2312.17670)
(2023).

</div>

<div class="sek-metric" markdown>

### Betti-1 error (tunnels/loops) (β₁ err) {#betti1_error}

<div class="sek-meta"><span class="sek-chip">key: <code>betti1_error</code></span><span class="sek-chip">range [0, ∞)</span><span class="sek-chip teal">lower is better</span><span class="sek-chip orange">count</span></div>

\[
\varepsilon_{\beta_1} = \lvert\beta_1(P) - \beta_1(G)\rvert
\]

**In plain words:** the difference in the number of loops or handles, for example a vessel ring (the circle of
Willis) that is opened, or two branches falsely fused into a loop.

**Use it when:** loops are anatomically meaningful (circle of Willis, vascular anastomoses) or should be absent
(most airway and vessel trees). Filling the hole of a ring gives \(\varepsilon_{\beta_1} = 1\) with Dice 0.80.

**Watch out for:** it is spatially blind: a lost loop in one place and a spurious loop elsewhere give 0. Small
false bridges between adjacent structures create loops, so it is sensitive to touching predictions.

**Empty masks:** as [β₀ error](#betti0_error).

**Parameters:** none.

**Reference:** Hu et al. 2019, [arXiv:1906.05404](https://arxiv.org/abs/1906.05404); Yang et al. 2023 (TopCoW),
[arXiv:2312.17670](https://arxiv.org/abs/2312.17670).

</div>

<div class="sek-metric" markdown>

### Betti-2 error (cavities) (β₂ err) {#betti2_error}

<div class="sek-meta"><span class="sek-chip">key: <code>betti2_error</code></span><span class="sek-chip">range [0, ∞)</span><span class="sek-chip teal">lower is better</span><span class="sek-chip orange">count</span></div>

\[
\varepsilon_{\beta_2} = \lvert\beta_2(P) - \beta_2(G)\rvert
\]

**In plain words:** the difference in the number of enclosed holes, for example a lumen that has been filled in or
a spurious void inside a solid organ.

**Use it when:** segmenting hollow or thin-walled structures (bladder wall, myocardium, colon wall) whose
cavity must be preserved.

**Watch out for:** a cavity counts only if it is fully enclosed (no 6-connected path to the outside). A lumen
that is open at both ends is a tunnel (\(\beta_1\)), not a cavity. It is spatially blind like the other Betti
errors.

**Empty masks:** as [β₀ error](#betti0_error).

**Parameters:** none.

**Reference:** Hu et al. 2019, [arXiv:1906.05404](https://arxiv.org/abs/1906.05404).

</div>

<div class="sek-metric" markdown>

### Euler characteristic error (χ err) {#euler_error}

<div class="sek-meta"><span class="sek-chip">key: <code>euler_error</code></span><span class="sek-chip">range [0, ∞)</span><span class="sek-chip teal">lower is better</span><span class="sek-chip orange">count</span></div>

\[
\varepsilon_\chi = \lvert\chi(P) - \chi(G)\rvert,\qquad \chi = \beta_0 - \beta_1 + \beta_2
\]

**In plain words:** one number for the overall topology: components minus loops plus cavities.

**Use it when:** you want a cheap topology screen, for example cortical-surface genus.

**Watch out for:** errors of different kind cancel: one extra component plus one extra loop gives
\(\Delta\chi = 0\). Use it as a screen and follow up with the individual Betti errors. It is not the expected
calibration error ([ECE](calibration.md#ece)), which is why the key is `euler_error`.

**Empty masks:** both empty: 0 (`"best"`) or NaN. Exactly one empty: \(\lvert\chi\rvert\) of the other mask.

**Parameters:** none.

**Reference:** Kong TY, Rosenfeld A. Digital topology: introduction and survey. *Computer Vision, Graphics, and
Image Processing* 48(3), 357–393 (1989).
[doi:10.1016/0734-189X(89)90147-3](https://doi.org/10.1016/0734-189X(89)90147-3)

</div>

<div class="sek-metric" markdown>

### Centreline Dice (clDice) {#cldice}

<div class="sek-meta"><span class="sek-chip">key: <code>cldice</code></span><span class="sek-chip">alias: <code>cldsc</code></span><span class="sek-chip">range [0, 1]</span><span class="sek-chip teal">higher is better</span><span class="sek-chip orange">dimensionless</span></div>

\[
\mathrm{clDice} = 2\,\frac{T_{prec}\,T_{sens}}{T_{prec}+T_{sens}},\qquad
T_{prec} = \frac{\lvert S_P\cap G\rvert}{\lvert S_P\rvert},\quad
T_{sens} = \frac{\lvert S_G\cap P\rvert}{\lvert S_G\rvert}
\]

**In plain words:** checks that the reference centreline is covered by the prediction (topology sensitivity) and
that the predicted centreline stays inside the reference (topology precision). It rewards complete, connected
tubes.

**Use it when:** segmenting tubular structures (vessels, airways, nerves, ducts). Metrics Reloaded recommends
clDice as the overlap metric when the target is tubular.

**Watch out for:** it depends on the skeletonisation algorithm and on voxel anisotropy; resample to isotropic
spacing for thin structures, or state the choice. It is insensitive to radius errors. A short gap removes only a
short piece of centreline, so a 2-voxel break in a 56-voxel vessel still scores 0.98; pair it with
[β₀ error](#betti0_error) when breaks matter. Skeletons of blob-like, non-tubular structures are unstable.

**Empty masks:** both empty: 1 (`"best"`) or NaN. Exactly one empty: 0. If a skeleton is empty, or
\(T_{prec} + T_{sens} = 0\), the value is 0.

**Parameters:** none.

**Reference:** Shit S, Paetzold JC, Sekuboyina A, et al. clDice – a novel topology-preserving loss function for
tubular structure segmentation. *CVPR 2021*, 16560–16569.
[arXiv:2003.07311](https://arxiv.org/abs/2003.07311)

</div>

## Code

The `topology` metric set contains `cldice`, `betti0_error`, `betti1_error`, `betti2_error` and `euler_error`.
Betti numbers are computed once per pair and cached.

```python
import numpy as np
from segevalkit.metrics import compute_metrics, betti_numbers

vessel = np.zeros((60, 9, 9), bool)
vessel[2:58, 3:6, 3:6] = True            # a straight 3x3 "vessel"
broken = vessel.copy()
broken[29:31] = False                    # a 2-voxel gap in the middle

print(betti_numbers(vessel), betti_numbers(broken))
scores = compute_metrics(broken, vessel, ["dice", "hd95", "topology"], spacing=(1, 1, 1))
for name, value in scores.items():
    print(f"{name:14s} {value:.4f}")
```

```text
(1, 0, 0) (2, 0, 0)
dice           0.9818
hd95           0.0000
cldice         0.9818
betti0_error   1.0000
betti1_error   0.0000
betti2_error   0.0000
euler_error    1.0000
```

Dice, HD95 and even clDice barely register the break; the β₀ error is the only metric that flags it.

!!! note "Betti matching"
    Betti number errors compare counts, not locations. Betti matching (Stucki et al., ICML 2023,
    [code](https://github.com/nstucki/Betti-matching)) matches features spatially through persistent homology.
    It is not built into SegEvalKit, but can be added as a [custom metric](index.md#custom-metrics).
