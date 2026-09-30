# Volume metrics

Volume metrics compare **how much** was segmented, not **where**. A prediction can have a perfect volume and zero
overlap with the reference. They are the right endpoint for volumetry questions (organ volume, tumour burden,
treatment response, atrophy) and the wrong one for anything else, so always pair them with an
[overlap](overlap.md) or [distance](distance.md) metric.

Dataset-level volume agreement (Bland–Altman bias and limits of agreement, intraclass correlation) is computed
from the per-case volumes by `segevalkit.stats.bland_altman` and `segevalkit.stats.icc`; see
[Statistics & ranking](../analysis/statistics.md).

## Notation

\(G\) and \(P\) are the reference and predicted foreground voxel sets, \(\lvert\cdot\rvert\) counts voxels,
\(s = (s_x, s_y, s_z)\) is the voxel spacing in mm and \(v_{\text{voxel}} = s_x s_y s_z\) the voxel volume in mm³.
Physical volumes are in millilitres:

\[
V_P = \lvert P\rvert\, v_{\text{voxel}} / 1000, \qquad V_G = \lvert G\rvert\, v_{\text{voxel}} / 1000 \qquad (1\ \text{mL} = 1000\ \text{mm}^3).
\]

In terms of confusion counts, \(\lvert P\rvert = TP + FP\) and \(\lvert G\rvert = TP + FN\). Without a spacing,
SegEvalKit assumes 1 mm isotropic voxels, so mL values are then really "thousands of voxels".

## At a glance

| Metric | Key | Unit | Ideal | Needs spacing | Signed | Note |
|---|---|---|---|:-:|:-:|---|
| [Predicted volume](#pred_volume) | `pred_volume` | mL | descriptive | yes | – | not a score |
| [Reference volume](#ref_volume) | `ref_volume` | mL | descriptive | yes | – | stratification variable |
| [Signed volume difference](#volume_difference) | `volume_difference` | mL | 0 | yes | yes | shows bias |
| [Absolute volume difference](#absolute_volume_difference) | `absolute_volume_difference` | mL | 0 | yes | no | ISLES, autoPET |
| [Relative volume difference](#relative_volume_difference) | `relative_volume_difference` | fraction | 0 | no | yes | explodes for tiny references |
| [Volumetric similarity](#volumetric_similarity) | `volumetric_similarity` | – | 1 | no | no | bounded in [0, 1] |

## Metrics

<div class="sek-metric" markdown>

### Predicted volume (Vol(P)) {#pred_volume}

<div class="sek-meta"><span class="sek-chip">key: <code>pred_volume</code></span><span class="sek-chip">range [0, ∞)</span><span class="sek-chip teal">descriptive (not a score)</span><span class="sek-chip orange">mL</span><span class="sek-chip">needs spacing</span></div>

\[
V_P = \lvert P\rvert\cdot v_{\text{voxel}} / 1000
\]

**In plain words:** the physical volume of the prediction.

**Use it when:** you need the raw volumes for Bland–Altman analysis, ICC, or presence detection from the
predicted volume (`segevalkit.stats.presence_detection` uses it by default).

**Watch out for:** it has no "better" direction and is not a quality score. Plots show it without a direction
arrow.

**Empty masks:** 0 for an empty prediction.

**Parameters:** none.

**Reference:** descriptive quantity; no primary reference.

</div>

<div class="sek-metric" markdown>

### Reference volume (Vol(G)) {#ref_volume}

<div class="sek-meta"><span class="sek-chip">key: <code>ref_volume</code></span><span class="sek-chip">range [0, ∞)</span><span class="sek-chip teal">descriptive (not a score)</span><span class="sek-chip orange">mL</span><span class="sek-chip">needs spacing</span></div>

\[
V_G = \lvert G\rvert\cdot v_{\text{voxel}} / 1000
\]

**In plain words:** the physical volume of the reference.

**Use it when:** stratifying any other metric by structure size, which is the standard way to expose the size
dependence of Dice (`segevalkit.stats.stratify(result, "dice", by="ref_volume_ml")`).

**Watch out for:** nothing specific; the `Evaluator` records it for every case and label anyway (as the
`_ref_volume_ml` flag row), whether or not you request it.

**Empty masks:** 0 for an empty reference.

**Parameters:** none.

**Reference:** descriptive quantity; no primary reference.

</div>

<div class="sek-metric" markdown>

### Signed volume difference (ΔV) {#volume_difference}

<div class="sek-meta"><span class="sek-chip">key: <code>volume_difference</code></span><span class="sek-chip">range (−∞, ∞)</span><span class="sek-chip teal">0 is best</span><span class="sek-chip orange">mL</span><span class="sek-chip">needs spacing</span></div>

\[
\Delta V = V_P - V_G
\]

**In plain words:** how many millilitres the prediction is too large (positive) or too small (negative).

**Use it when:** you want to detect systematic bias. Its mean over cases is the Bland–Altman bias.

**Watch out for:** averaging signed errors lets over- and under-segmentation cancel: a method that is 10 mL too
large on half the cases and 10 mL too small on the other half has mean \(\Delta V = 0\). Report the absolute
difference or the limits of agreement as well. It is blind to position.

**Empty masks:** always defined; an empty prediction gives \(-V_G\).

**Parameters:** none.

**Reference:** Heimann T, van Ginneken B, Styner MA, et al. Comparison and evaluation of methods for liver
segmentation from CT datasets. *IEEE TMI* 28(8), 1251–1265 (2009).
[doi:10.1109/TMI.2009.2013851](https://doi.org/10.1109/TMI.2009.2013851)

</div>

<div class="sek-metric" markdown>

### Absolute volume difference (AVD) {#absolute_volume_difference}

<div class="sek-meta"><span class="sek-chip">key: <code>absolute_volume_difference</code></span><span class="sek-chip">alias: <code>avd</code></span><span class="sek-chip">range [0, ∞)</span><span class="sek-chip teal">lower is better</span><span class="sek-chip orange">mL</span><span class="sek-chip">needs spacing</span></div>

\[
\mathrm{AVD} = \lvert V_P - V_G\rvert
\]

**In plain words:** the size of the volume error in millilitres, regardless of sign.

**Use it when:** volume is the clinical endpoint (stroke lesion volume in ISLES'22, total tumour volume in
autoPET), and especially for small structures, where relative errors explode.

**Watch out for:** it is blind to position: a mask of the right size in the wrong place scores 0. Absolute mL
values are not comparable between structures of very different size.

**Empty masks:** always defined; if the reference is empty it equals the predicted volume.

**Parameters:** none.

**Reference:** Hernandez Petzsche MR, de la Rosa E, Hanning U, et al. ISLES 2022: A multi-center magnetic
resonance imaging stroke lesion segmentation dataset. *Scientific Data* 9, 762 (2022).
[doi:10.1038/s41597-022-01875-5](https://doi.org/10.1038/s41597-022-01875-5)

</div>

<div class="sek-metric" markdown>

### Relative volume difference (RVD) {#relative_volume_difference}

<div class="sek-meta"><span class="sek-chip">key: <code>relative_volume_difference</code></span><span class="sek-chip">alias: <code>rvd</code></span><span class="sek-chip">range [−1, ∞)</span><span class="sek-chip teal">0 is best</span><span class="sek-chip orange">fraction</span></div>

\[
\mathrm{RVD} = \frac{\lvert P\rvert - \lvert G\rvert}{\lvert G\rvert}
\]

**In plain words:** the signed volume error relative to the reference volume. \(-0.2\) means the prediction is
20 % too small; \(+0.1\) means 10 % too large.

**Use it when:** you want a scale-free measure of volumetric bias that can be compared across patients with
different organ sizes. It is part of the `default` metric set.

**Watch out for:** it explodes for tiny references: 3 extra voxels on a 2-voxel lesion is \(+1.5\). Averaging the
signed value lets errors cancel. It is blind to position. SegEvalKit returns a fraction, not a percentage, and
the value is spacing-free because the voxel volume cancels.

**Empty masks:** both empty: 0 (`"best"`) or NaN (`"nan"`). Empty reference with a non-empty prediction: NaN
(undefined). Empty prediction with a non-empty reference: \(-1\).

**Parameters:** none.

**Reference:** Heimann et al. 2009, [doi:10.1109/TMI.2009.2013851](https://doi.org/10.1109/TMI.2009.2013851).

</div>

<div class="sek-metric" markdown>

### Volumetric similarity (VS) {#volumetric_similarity}

<div class="sek-meta"><span class="sek-chip">key: <code>volumetric_similarity</code></span><span class="sek-chip">alias: <code>vs</code></span><span class="sek-chip">range [0, 1]</span><span class="sek-chip teal">higher is better</span><span class="sek-chip orange">dimensionless</span></div>

\[
\mathrm{VS} = 1 - \frac{\big\lvert \lvert P\rvert - \lvert G\rvert \big\rvert}{\lvert P\rvert + \lvert G\rvert}
= 1 - \frac{\lvert FN - FP\rvert}{2\,TP + FP + FN}
\]

**In plain words:** how similar the two volumes are in size, on a bounded 0–1 scale, ignoring location.

**Use it when:** you want a bounded volumetric score to put next to Dice in a table (it shares Dice's
denominator).

**Watch out for:** VS = 1 is possible with zero overlap. Two equal-sized cubes in different places score
VS = 1, RVD = 0, AVD = 0 mL and Dice = 0 (see [Pitfalls](../guide/pitfalls.md#boundary-volume)). Never use it
alone.

**Empty masks:** both empty: 1 (`"best"`) or NaN (`"nan"`). Exactly one empty: 0.

**Parameters:** none.

**Reference:** Taha AA, Hanbury A. Metrics for evaluating 3D medical image segmentation: analysis, selection,
and tool. *BMC Medical Imaging* 15, 29 (2015).
[doi:10.1186/s12880-015-0068-x](https://doi.org/10.1186/s12880-015-0068-x)

</div>

## Code

The `volume` metric set contains `pred_volume`, `ref_volume`, `absolute_volume_difference`,
`relative_volume_difference` and `volumetric_similarity`. Volumes use the spacing, so pass it (the `Evaluator`
reads it from the image header).

```python
import numpy as np
from segevalkit.metrics import compute_metrics

ref = np.zeros((40, 40, 20), bool)
ref[10:30, 10:30, 5:15] = True           # 4000 voxels
pred = np.zeros_like(ref)
pred[10:30, 10:30, 5:16] = True          # one extra slice: 4400 voxels

scores = compute_metrics(pred, ref, ["volume", "volume_difference", "dice"],
                         spacing=(0.8, 0.8, 2.5))   # voxel = 1.6 mm^3
for name, value in scores.items():
    print(f"{name:28s} {value:.4f}")
```

```text
pred_volume                  7.0400
ref_volume                   6.4000
absolute_volume_difference   0.6400
relative_volume_difference   0.1000
volumetric_similarity        0.9524
volume_difference            0.6400
dice                         0.9524
```

For dataset-level volumetry, collect per-case volumes from an `EvaluationResult` and pass them to
`segevalkit.stats.bland_altman(pred_ml, ref_ml)` and `segevalkit.stats.icc(pred_ml, ref_ml, kind="ICC(2,1)")`.
