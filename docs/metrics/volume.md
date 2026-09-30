# Volume metrics

Volume metrics compare **how much** was segmented, not **where**: a perfect volume can have zero overlap. Use them
for volumetry endpoints (organ volume, tumour burden, response, atrophy) and always pair them with an
[overlap](overlap.md) or [distance](distance.md) metric. Dataset-level agreement (Bland–Altman, ICC) is in
[Statistics](../analysis/statistics.md).

## At a glance

| Metric | Key | Unit | Ideal | Spacing | Signed | Note |
|---|---|---|---|:-:|:-:|---|
| [Predicted volume](#pred_volume) | `pred_volume` | mL | – | yes | – | not a score |
| [Reference volume](#ref_volume) | `ref_volume` | mL | – | yes | – | stratification variable |
| [Signed difference](#volume_difference) | `volume_difference` | mL | 0 | yes | yes | shows bias |
| [Absolute difference](#absolute_volume_difference) | `absolute_volume_difference` | mL | 0 | yes | no | ISLES, autoPET |
| [Relative difference](#relative_volume_difference) | `relative_volume_difference` | fraction | 0 | no | yes | explodes for tiny references |
| [Volumetric similarity](#volumetric_similarity) | `volumetric_similarity` | – | 1 | no | no | bounded [0, 1] |

## Notation

\(G, P\): reference and predicted voxel sets, \(\lvert\cdot\rvert\) counts voxels (\(\lvert P\rvert = TP+FP\),
\(\lvert G\rvert = TP+FN\)), \(v_{\text{voxel}} = s_x s_y s_z\) the voxel volume in mm³. Volumes are in mL
(1 mL = 1000 mm³): \(V_X = \lvert X\rvert\, v_{\text{voxel}}/1000\). Without a spacing SegEvalKit assumes 1 mm
isotropic voxels, so "mL" then means thousands of voxels. None of these metrics takes parameters.

## Metrics

<div class="sek-metric" markdown>

### Predicted volume (Vol(P)) {#pred_volume}

<div class="sek-meta"><span class="sek-chip">pred_volume</span><span class="sek-chip">[0, ∞)</span><span class="sek-chip teal">→ descriptive</span><span class="sek-chip orange">mL</span></div>

\[
V_P = \lvert P\rvert\cdot v_{\text{voxel}} / 1000
\]

**In words** The physical volume of the prediction.

**Use when** Feeding Bland–Altman, ICC, or presence detection (`segevalkit.stats.presence_detection` uses it by default).

**Watch out** Not a quality score and has no better direction; plots show it without an arrow.

??? info "Details: empty masks, parameters, reference"
    **Empty masks.** 0 for an empty prediction. **Parameters.** None. **Reference.** Descriptive quantity; none.

</div>

<div class="sek-metric" markdown>

### Reference volume (Vol(G)) {#ref_volume}

<div class="sek-meta"><span class="sek-chip">ref_volume</span><span class="sek-chip">[0, ∞)</span><span class="sek-chip teal">→ descriptive</span><span class="sek-chip orange">mL</span></div>

\[
V_G = \lvert G\rvert\cdot v_{\text{voxel}} / 1000
\]

**In words** The physical volume of the reference.

**Use when** Stratifying another metric by size to expose Dice's size bias: `stats.stratify(result, "dice", by="ref_volume_ml")`.

**Watch out** Nothing specific; the `Evaluator` always records it as the `_ref_volume_ml` flag row, requested or not.

??? info "Details: empty masks, parameters, reference"
    **Empty masks.** 0 for an empty reference. **Parameters.** None. **Reference.** Descriptive quantity; none.

</div>

<div class="sek-metric" markdown>

### Signed volume difference (ΔV) {#volume_difference}

<div class="sek-meta"><span class="sek-chip">volume_difference</span><span class="sek-chip">(−∞, ∞)</span><span class="sek-chip teal">→0 best</span><span class="sek-chip orange">mL</span></div>

\[
\Delta V = V_P - V_G
\]

**In words** How many millilitres the prediction is too large (positive) or too small (negative).

**Use when** Detecting systematic bias; its mean over cases is the Bland–Altman bias.

**Watch out** Signed errors cancel when averaged (±10 mL on half the cases each gives mean 0); it is blind to position.

??? info "Details: empty masks, parameters, reference"
    **Empty masks.** Always defined; an empty prediction gives \(-V_G\). Report the absolute difference or limits of agreement alongside the mean.

    **Reference.** Heimann T et al. *IEEE TMI* 28(8):1251–1265, 2009. [doi:10.1109/TMI.2009.2013851](https://doi.org/10.1109/TMI.2009.2013851)

</div>

<div class="sek-metric" markdown>

### Absolute volume difference (AVD) {#absolute_volume_difference}

<div class="sek-meta"><span class="sek-chip">absolute_volume_difference · avd</span><span class="sek-chip">[0, ∞)</span><span class="sek-chip teal">↓ lower is better</span><span class="sek-chip orange">mL</span></div>

\[
\mathrm{AVD} = \lvert V_P - V_G\rvert
\]

**In words** The size of the volume error in millilitres, regardless of sign.

**Use when** Volume is the endpoint (ISLES'22 lesion volume, autoPET tumour volume), especially for small structures where relative errors explode.

**Watch out** A right-sized mask in the wrong place scores 0; mL values are not comparable across structures of different size.

??? info "Details: empty masks, parameters, reference"
    **Empty masks.** Always defined; with an empty reference it equals the predicted volume.

    **Reference.** Hernandez Petzsche MR et al. ISLES 2022. *Scientific Data* 9:762, 2022. [doi:10.1038/s41597-022-01875-5](https://doi.org/10.1038/s41597-022-01875-5)

</div>

<div class="sek-metric" markdown>

### Relative volume difference (RVD) {#relative_volume_difference}

<div class="sek-meta"><span class="sek-chip">relative_volume_difference · rvd</span><span class="sek-chip">[−1, ∞)</span><span class="sek-chip teal">→0 best</span></div>

\[
\mathrm{RVD} = \frac{\lvert P\rvert - \lvert G\rvert}{\lvert G\rvert}
\]

**In words** Signed volume error relative to the reference: \(-0.2\) is 20 % too small, \(+0.1\) is 10 % too large.

**Use when** You want scale-free volumetric bias comparable across patient sizes; it is in the `default` set.

**Watch out** Explodes for tiny references (3 extra voxels on a 2-voxel lesion is \(+1.5\)); signed values cancel; blind to position.

??? info "Details: empty masks, parameters, reference"
    **Convention.** A fraction, not a percentage; spacing-free because the voxel volume cancels.

    **Empty masks.** Both empty: 0 (`"best"`) or NaN (`"nan"`). Empty reference, non-empty prediction: NaN. Empty prediction, non-empty reference: \(-1\).

    **Reference.** Heimann et al. 2009, [doi:10.1109/TMI.2009.2013851](https://doi.org/10.1109/TMI.2009.2013851).

</div>

<div class="sek-metric" markdown>

### Volumetric similarity (VS) {#volumetric_similarity}

<div class="sek-meta"><span class="sek-chip">volumetric_similarity · vs</span><span class="sek-chip">[0, 1]</span><span class="sek-chip teal">↑ higher is better</span></div>

\[
\mathrm{VS} = 1 - \frac{\big\lvert \lvert P\rvert - \lvert G\rvert \big\rvert}{\lvert P\rvert + \lvert G\rvert}
= 1 - \frac{\lvert FN - FP\rvert}{2\,TP + FP + FN}
\]

**In words** How similar the two volumes are in size, on a bounded 0–1 scale, ignoring location.

**Use when** You want a bounded volumetric score next to Dice in a table (it shares Dice's denominator).

**Watch out** VS = 1 is possible with zero overlap ([boundary vs volume](../guide/pitfalls.md#boundary-volume)); never use it alone.

??? info "Details: empty masks, parameters, reference"
    **Example.** Two equal-sized cubes in different places score VS = 1, RVD = 0, AVD = 0 mL and Dice = 0.

    **Empty masks.** Both empty: 1 (`"best"`) or NaN (`"nan"`). One empty: 0.

    **Reference.** Taha AA, Hanbury A. *BMC Medical Imaging* 15:29, 2015. [doi:10.1186/s12880-015-0068-x](https://doi.org/10.1186/s12880-015-0068-x)

</div>

## Code

The `volume` set is `pred_volume`, `ref_volume`, `absolute_volume_difference`, `relative_volume_difference`,
`volumetric_similarity`. Pass the spacing (the `Evaluator` reads it from the header).

```python
import numpy as np
from segevalkit.metrics import compute_metrics

ref = np.zeros((40, 40, 20), bool); ref[10:30, 10:30, 5:15] = True    # 4000 voxels
pred = np.zeros_like(ref); pred[10:30, 10:30, 5:16] = True            # +1 slice: 4400 voxels
scores = compute_metrics(pred, ref, ["volume", "volume_difference", "dice"],
                         spacing=(0.8, 0.8, 2.5))                     # voxel = 1.6 mm^3
# pred_volume 7.04  ref_volume 6.40  absolute_volume_difference 0.64
# relative_volume_difference 0.10  volumetric_similarity 0.9524
# volume_difference 0.64  dice 0.9524
```

For dataset-level volumetry pass per-case volumes to `segevalkit.stats.bland_altman(pred_ml, ref_ml)` and
`segevalkit.stats.icc(pred_ml, ref_ml, kind="ICC(2,1)")`.
