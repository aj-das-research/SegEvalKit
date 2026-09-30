# Evaluation levels

A model is judged on a whole test set, not on one file, and a test set answers questions at several levels.
One `Evaluator.evaluate()` call over a folder produces all of them.

| Level | Question | Where |
|---|---|---|
| **Instance** | Was this lesion found, and how well? | `res.lesions` · `lesions.csv` |
| **Sample** | How good is this patient's segmentation? | `res.wide()` · `per_case.csv` |
| **Dataset, macro** | What is the typical per-case score? | `res.summary()` · `summary.csv` |
| **Dataset, micro** | What is the score with all voxels or lesions pooled? | `res.pooled()` · `pooled.csv` |
| **Cohort** | Does performance differ by site, scanner, phase, sex, age, size? | `cohort_summary`, `cohort_tests` |
| **Patient** | Does this patient have the structure (tumour) at all? | `stats.presence_detection` |

```python
import segevalkit as sek
res = sek.Evaluator(labels=..., metrics=["default", "detection"]).evaluate("preds/", "labels/", out_dir="eval/")
```

## Instance and sample

Every case gets every metric (`per_case.csv`), and with detection metrics every reference lesion and every
unmatched predicted component gets a row (`lesions.csv`: volume, detected, Dice, IoU). These are the units to
inspect when something goes wrong, e.g. `res.worst_cases("dice", "pancreas")`.

## Dataset: macro and micro

=== "Macro (per-case mean)"

    ```python
    res.summary()          # mean, median, IQR, 95 % bootstrap CI over cases
    ```

    Every patient weighs the same. This is what most challenges rank and what clinicians usually mean by
    "how well does it work on a patient".

=== "Micro (pooled)"

    ```python
    res.pooled()           # pooled Dice / IoU / precision / recall, pooled lesion sensitivity, precision, F1,
                           # false-positive lesions per scan; 95 % CIs from a case bootstrap
    ```

    Every voxel (or lesion) weighs the same:
    \( \mathrm{DSC}_{\text{agg}} = 2\sum_i TP_i / (2\sum_i TP_i + \sum_i FP_i + \sum_i FN_i) \). Large
    structures and lesion-rich patients dominate. HECKTOR 2022 ranks by aggregated Dice; pooled lesion
    sensitivity is the natural "fraction of tumours found".

The two can disagree sharply: a model that fails on small, tumour-free-looking cases loses much macro Dice but
little pooled Dice. Report the level that matches your question and say which one it is.

## Cohorts

Attach any per-case metadata (a CSV or Excel file with a case-id column) and compare subgroups:

=== "Python"

    ```python
    from segevalkit.cohort import cohort_summary, cohort_tests
    from segevalkit import plotting as P

    meta = "PanTS/metadata.xlsx"                     # one row per case
    by = ["ct phase", "sex", "manufacturer", "site nationality"]
    summ = cohort_summary(res, meta, by, metrics=["dice", "nsd", "hd95"], labels=["pancreas"],
                          bins={"age": [0, 40, 60, 80, 120]})
    tests = cohort_tests(res, meta, by, metrics=["dice", "nsd"], labels=["pancreas"])
    P.cohort_plot(summ, "dice", "pancreas", factor="ct phase")
    ```

=== "Command line"

    ```console
    $ segevalkit cohort eval/ --metadata PanTS/metadata.xlsx --id-column "PanTS ID" \
          --by "ct phase,sex,manufacturer" --metrics dice,nsd,hd95 --labels pancreas
    ```

`cohort_summary` gives, per factor and subgroup, *n*, mean, median, IQR and a bootstrap CI; subgroups smaller than
`min_n` (default 5) are flagged. `cohort_tests` asks whether subgroups differ: Mann–Whitney U for two groups,
Kruskal–Wallis for more, with an effect size and Holm correction across every factor × structure × metric.
Categorical values are normalised ("M " and "M" are one group); numeric columns can be binned.

!!! note "Subgroups are observational"
    A difference between sites or scanners may reflect case mix (tumour size, phase) rather than the scanner.
    Stratify by the variable you suspect, and read cohort differences as hypotheses, not causes.

## Patient level

When references can be empty (tumour-free patients), segmentation metrics describe only the patients that have
the structure. The patient-level question, "is there a tumour?", is answered by
[presence detection](statistics.md#presence-detection): sensitivity and specificity at a volume threshold and the
ROC AUC across thresholds, the patient-wise protocol of the PanTS benchmark.
