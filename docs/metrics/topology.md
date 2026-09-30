# Topology & centreline metrics

A vessel tree broken into ten pieces, or a hollow organ with its lumen filled in, can score excellent overlap and
distance. Topological metrics count what the shape **is**: components, tunnels and cavities. Use them when
connectivity matters (vessels, airways, ducts, circle of Willis, cortex, thin walls), always next to an overlap
metric: they say nothing about overlap, and errors in different places cancel.

## At a glance

| Metric | Key | Detects | Misses |
|---|---|---|---|
| [β₀ error](#betti0_error) | `betti0_error` | fragmentation, spurious islands | where the break is |
| [β₁ error](#betti1_error) | `betti1_error` | opened or falsely closed loops | location; lost + gained loop cancel |
| [β₂ error](#betti2_error) | `betti2_error` | filled lumens, spurious voids | location |
| [Euler error](#euler_error) | `euler_error` | any net topology change | different error kinds cancel |
| [clDice](#cldice) | `cldice` | incomplete / leaking centrelines | radius errors; short gaps |

## Notation

| Symbol | Counts | Computed as |
|---|---|---|
| \(\beta_0\) | connected components | 26-connected foreground components |
| \(\beta_1\) | tunnels / loops | \(\beta_0 + \beta_2 - \chi\) |
| \(\beta_2\) | enclosed cavities | 6-connected background components minus the outside |
| \(\chi\) | Euler characteristic | `skimage.measure.euler_number` (26-connectivity) |
| \(S_X\) | skeleton of \(X\) | Lee et al. (1994) 3D thinning (scikit-image) |

The (26, 6) adjacency is the standard well-composed 3D choice (Kong & Rosenfeld 1989); masks are padded with
background, and the `connectivity` argument affects only [detection](detection.md), not Betti numbers. A ball is
\((1,0,0)\), a torus \((1,1,0)\), a hollow shell \((1,0,1)\); `segevalkit.metrics.betti_numbers(mask)` returns the
triple. A component whose skeleton vanishes under thinning keeps its most interior voxel. An empty mask has
\(\beta = (0,0,0)\). None of these metrics takes parameters.

## Metrics

<div class="sek-metric" markdown>

### Betti-0 error (β₀ err; components) {#betti0_error}

<div class="sek-meta"><span class="sek-chip">betti0_error</span><span class="sek-chip">[0, ∞)</span><span class="sek-chip teal">↓ lower is better</span><span class="sek-chip orange">count</span></div>

\[
\varepsilon_{\beta_0} = \lvert\beta_0(P) - \beta_0(G)\rvert
\]

**In words** The difference in the number of connected pieces: fragmentation and spurious islands.

**Use when** The structure should be one piece (vessels, airways); a 2-voxel gap leaves Dice at 0.98 but gives 1 ([example](#code)).

**Watch out** Compares counts only (a missing and a spurious piece cancel), and a noise speck counts as much as a severed artery.

??? info "Details: empty masks, parameters, reference"
    **Empty masks.** Both empty: 0 (`"best"`) or NaN. One empty: \(\beta_0\) of the other mask.

    **Reference.** Hu X et al. Topology-preserving deep image segmentation. *NeurIPS 2019*. [arXiv:1906.05404](https://arxiv.org/abs/1906.05404). Yang K et al. TopCoW, 2023. [arXiv:2312.17670](https://arxiv.org/abs/2312.17670)

</div>

<div class="sek-metric" markdown>

### Betti-1 error (β₁ err; tunnels/loops) {#betti1_error}

<div class="sek-meta"><span class="sek-chip">betti1_error</span><span class="sek-chip">[0, ∞)</span><span class="sek-chip teal">↓ lower is better</span><span class="sek-chip orange">count</span></div>

\[
\varepsilon_{\beta_1} = \lvert\beta_1(P) - \beta_1(G)\rvert
\]

**In words** The difference in the number of loops, e.g. an opened circle of Willis or two branches falsely fused.

**Use when** Loops are meaningful (circle of Willis, anastomoses) or should be absent (most trees); filling a ring's hole gives 1 at Dice 0.80.

**Watch out** Spatially blind (lost and spurious loops cancel), and small false bridges between touching structures create loops.

??? info "Details: empty masks, parameters, reference"
    **Empty masks.** As [β₀ error](#betti0_error), with \(\beta_1\).

    **Reference.** Hu et al. 2019, [arXiv:1906.05404](https://arxiv.org/abs/1906.05404); Yang et al. 2023 (TopCoW), [arXiv:2312.17670](https://arxiv.org/abs/2312.17670).

</div>

<div class="sek-metric" markdown>

### Betti-2 error (β₂ err; cavities) {#betti2_error}

<div class="sek-meta"><span class="sek-chip">betti2_error</span><span class="sek-chip">[0, ∞)</span><span class="sek-chip teal">↓ lower is better</span><span class="sek-chip orange">count</span></div>

\[
\varepsilon_{\beta_2} = \lvert\beta_2(P) - \beta_2(G)\rvert
\]

**In words** The difference in the number of enclosed holes, e.g. a filled-in lumen or a spurious void.

**Use when** Hollow or thin-walled structures (bladder wall, myocardium, colon wall) must keep their cavity.

**Watch out** Only fully enclosed cavities count; a lumen open at both ends is a tunnel (\(\beta_1\)); spatially blind.

??? info "Details: empty masks, parameters, reference"
    **Empty masks.** As [β₀ error](#betti0_error), with \(\beta_2\).

    **Reference.** Hu et al. 2019, [arXiv:1906.05404](https://arxiv.org/abs/1906.05404).

</div>

<div class="sek-metric" markdown>

### Euler characteristic error (χ err) {#euler_error}

<div class="sek-meta"><span class="sek-chip">euler_error</span><span class="sek-chip">[0, ∞)</span><span class="sek-chip teal">↓ lower is better</span><span class="sek-chip orange">count</span></div>

\[
\varepsilon_\chi = \lvert\chi(P) - \chi(G)\rvert,\qquad \chi = \beta_0 - \beta_1 + \beta_2
\]

**In words** One number for the overall topology: components minus loops plus cavities.

**Use when** You want a cheap topology screen, for example cortical-surface genus.

**Watch out** Error kinds cancel (one extra component plus one extra loop gives 0); follow up with the Betti errors.

??? info "Details: empty masks, parameters, reference"
    **Empty masks.** Both empty: 0 (`"best"`) or NaN. One empty: \(\lvert\chi\rvert\) of the other mask.

    **Naming.** The key is `euler_error` to avoid confusion with the calibration error [ECE](calibration.md#ece).

    **Reference.** Kong TY, Rosenfeld A. Digital topology. *CVGIP* 48(3):357–393, 1989. [doi:10.1016/0734-189X(89)90147-3](https://doi.org/10.1016/0734-189X(89)90147-3)

</div>

<div class="sek-metric" markdown>

### Centreline Dice (clDice) {#cldice}

<div class="sek-meta"><span class="sek-chip">cldice · cldsc</span><span class="sek-chip">[0, 1]</span><span class="sek-chip teal">↑ higher is better</span></div>

\[
\mathrm{clDice} = 2\,\frac{T_{\mathrm{prec}}\,T_{\mathrm{sens}}}{T_{\mathrm{prec}}+T_{\mathrm{sens}}},\qquad
T_{\mathrm{prec}} = \frac{\lvert S_P\cap G\rvert}{\lvert S_P\rvert},\quad
T_{\mathrm{sens}} = \frac{\lvert S_G\cap P\rvert}{\lvert S_G\rvert}
\]

**In words** Rewards complete tubes: the reference centreline must lie in the prediction, and the predicted centreline in the reference.

**Use when** Segmenting tubular structures (vessels, airways, nerves, ducts); Metrics Reloaded's overlap metric for tubular targets.

**Watch out** Short gaps barely move it (a 2-voxel break scores 0.98; add [β₀ error](#betti0_error)); radius-blind; skeleton- and anisotropy-dependent.

??? info "Details: empty masks, parameters, reference"
    **Skeletons.** Resample thin structures to isotropic spacing or state the choice; skeletons of blob-like, non-tubular structures are unstable.

    **Empty masks.** Both empty: 1 (`"best"`) or NaN. One empty: 0. An empty skeleton, or \(T_{\mathrm{prec}}+T_{\mathrm{sens}}=0\), gives 0.

    **Reference.** Shit S et al. clDice. *CVPR 2021*, 16560–16569. [arXiv:2003.07311](https://arxiv.org/abs/2003.07311)

</div>

## Code

The `topology` set is `cldice`, `betti0_error`, `betti1_error`, `betti2_error`, `euler_error`; Betti numbers are
computed once per pair and cached.

```python
import numpy as np
from segevalkit.metrics import compute_metrics, betti_numbers

vessel = np.zeros((60, 9, 9), bool); vessel[2:58, 3:6, 3:6] = True    # straight 3x3 "vessel"
broken = vessel.copy(); broken[29:31] = False                         # 2-voxel gap
print(betti_numbers(vessel), betti_numbers(broken))                   # (1, 0, 0) (2, 0, 0)
scores = compute_metrics(broken, vessel, ["dice", "hd95", "topology"], spacing=(1, 1, 1))
# dice 0.9818  hd95 0.0000  cldice 0.9818
# betti0_error 1  betti1_error 0  betti2_error 0  euler_error 1
```

Only the β₀ error flags the break. Spatial Betti matching (Stucki et al., ICML 2023,
[code](https://github.com/nstucki/Betti-matching)) is not built in but can be added as a
[custom metric](index.md#custom-metrics).
