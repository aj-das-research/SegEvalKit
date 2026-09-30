# Datasets & benchmarks

A **preset** encodes a public benchmark's label map, evaluation regions, on-disk layout, official metrics and
parameters (including per-structure NSD tolerances) and empty-mask convention. One flag reproduces the protocol:

```console
$ segevalkit datasets                     # list the presets
$ segevalkit datasets kits23              # details of one
$ segevalkit evaluate --dataset msd_liver --pred preds/ \
    --ref Task03_Liver/labelsTr --out eval/
```

<div class="grid cards" markdown>

-   **[Dataset presets](presets.md)**

    Every preset with labels, layout, metrics, parameters, licence and citation (generated from the registry).

-   **[Full datasets survey](../research/datasets-survey.md)**

    Forty-odd CT / MR / PET datasets and challenges with verified label maps, layouts and protocols.

</div>

## Official protocols at a glance

Protocols differ in ways that change the numbers. Key differences, verified against official evaluation code where
public:

| Benchmark | Official metrics | Boundary tolerance | Empty reference & empty prediction | Notable detail |
|---|---|---|---|---|
| MSD | DSC, NSD | per task: 1 mm (hippocampus) to 7 mm (liver) | reported as 0 if undefined | Wilcoxon significance ranking |
| FLARE22 | DSC, NSD | per organ: 2 mm (vessels, adrenals) to 7 mm (duodenum) | 1 / 1 | NSD forced to 0 if DSC < 0.2 |
| KiTS21/23 | Dice, surface Dice on nested regions | 1.03 / 1.13 / 1.15 mm (from inter-observer variability) | 1 / 1 | surfel-weighted surface Dice |
| BraTS 2023 | lesion-wise Dice and HD95 on WT / TC / ET | – | Dice 1, HD95 0 | reference dilated before matching; each FP lesion scores HD95 374 mm; lesions ≤ 50 mm³ dropped |
| ISLES'22 | Dice, volume difference, lesion count difference, lesion F1 | – | Dice 1, F1 1 | any 1-voxel overlap is a hit |
| autoPET | Dice, FP volume, FN volume | – | tumour-free studies scored by FP volume only | 18-connected components |
| TopCoW 2024 | Dice, clDice, Betti-0 error, HD95 | – | classes absent from both skipped | HD95 capped at 90 mm |
| LiTS | Dice, VOE, RVD, ASSD, MSD | – | ASSD = 0 when a mask is empty (**not** penalised) | per-case lesion Dice |
| WORD | DSC, HD95 | – | DSC 0, HD95 50 even when **both** are empty | |
| PanTS | lesion DSC, patient- and tumour-wise sensitivity, specificity, AUC | not published | not published | no official evaluation script |

!!! warning "Label maps are not interchangeable"
    ACDC is 1 RV, 2 MYO, 3 LV; M&Ms is 1 LV, 2 MYO, 3 RV. MSD Task01 uses 1 edema, 2 non-enhancing, 3 enhancing;
    BraTS 2023 uses 1 NCR, 2 ED, 3 ET. PanTS per-structure masks overlap (pancreas contains its sub-parts, duct and
    lesion), so its `combined_labels.nii.gz` is lossy. The presets encode the correct maps.

## Data available in this project

The real-data tests (`tests/test_real_data.py`) check layouts, geometry handling and presets on these public
datasets (read-only):

| Dataset | Split | Cases |
|---|---|---|
| PanTS | public test (`ImageTe`, `LabelTe`) | 901 |
| TotalSegmentator v2 | local subset | 183 |
| MSD Task03 Liver, Task10 Colon | labelled training data | 131, 126 |
