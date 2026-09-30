# Volumetric Medical Image Segmentation: Datasets, Benchmarks and Official Evaluation Protocols

Survey for **SegEvalKit** (an open-source evaluation library). Compiled 2026-09-30.

## How to read this document

- **Verification levels.** Every fact carries one of three tags:
  - **[V-code]**: checked directly against official files. That means official evaluation code, `dataset.json`, a Zenodo record, a HuggingFace card, or label files downloaded and inspected by us.
  - **[V-doc]**: checked against an official web page, README or paper.
  - **[UNVERIFIED]**: taken from memory or secondary sources. Check it before encoding it in SegEvalKit defaults.
- The **"official metric" rows** describe what the organisers' own code does. In particular they record how it handles **empty masks**, **tolerances**, and **aggregation**. Those three things are where re-implementations usually disagree with leaderboards.
- The table of empty-mask conventions across benchmarks is in [Section 4](#4-cross-benchmark-empty-mask--tolerance-conventions).

---

## 1. Deep-dive: the four priority datasets

### 1.1 PanTS (Pancreatic Tumor Segmentation, JHU; NeurIPS 2025 D&B)

| Field | Value |
|---|---|
| Modality | CT (3D, multi-phase; 145 centres) [V-doc] |
| Scale | 36,390 CT volumes in total. The public release is **PanTS-tr n=9,000** (`PanTS_00000001` to `PanTS_00009000`) and **PanTS-te n=901** (`PanTS_00009001` to `PanTS_00009901`) [V-code: download scripts]. The remaining volumes are proprietary out-of-distribution test sets (UCSF 13,458; Poland 5,259; Peking Univ. 3,066) plus RSNA-ATD (4,706, re-annotated). These are evaluated by the JHU team when you email them a checkpoint [V-doc]. |
| Targets | Pancreatic lesion, pancreas plus head/body/tail, pancreatic duct, common bile duct, and 24 surrounding structures. 28 label names in total [V-code] |
| Spacing | In-plane median 0.81 mm (IQR 0.73–0.92) in train and 0.75 mm (0.70–0.83) in test. Slice thickness median 2.50 mm (IQR 0.87–3.00) in train and 1.25 mm (1.25–2.50) in test. These are the paper's Table 1 values [V-doc]. A case we inspected had 5.0 mm slices [V-code]. |
| Licence | The GitHub `LICENSE` says **CC BY-NC-ND 4.0** ("© The Johns Hopkins University"). The HuggingFace card `BodyMaps/PanTSMini` declares **cc-by-nc-sa-4.0**. The two conflict [V-code, both]. Treat the data as non-commercial and check with the authors before redistributing derived labels. |
| Download | `git clone https://github.com/MrGiovanni/PanTS && bash download_PanTS_data.sh && bash download_PanTS_label.sh`. Images come from HF `BodyMaps/PanTSMini`: 9 tarballs `PanTSMini_ImageTr_XXXXXXXX_YYYYYYYY.tar.gz` of about 35 GB each, plus `PanTSMini_ImageTe_00009001_00009901.tar.gz` (about 28 GB) and `metadata.xlsx`. Labels are one 15.6 GB tarball at `https://www.cs.jhu.edu/~zongwei/dataset/PanTSMini_Label.tar.gz`. The total needs about 300 GB [V-code]. |
| Official benchmark metrics | The README leaderboard on the in-distribution test set reports **P-Sen** (patient-wise sensitivity: a hit if any tumour is predicted in a tumour patient, location ignored), **T-Sen** (tumour-wise sensitivity: each tumour must be correctly localised), **Specificity**, **AUC**, and **DSC** (pancreatic lesion) [V-doc]. The paper also defines **NSD** with a tolerance τ, but we could **not find a stated value of τ** [V-doc]. **No official evaluation script is published in the repo** (we checked the repo tree). The detection threshold (minimum lesion volume or overlap for a "hit") and the handling of DSC on tumour-free cases are **not documented** [UNVERIFIED; ask the authors]. |
| Baselines (README) | nnU-Net (DSC 50.9%, AUC 0.902), MedFormer (52.9%, 0.924), R-Super (53.4%, 0.903; trained with extra report supervision). Checkpoints are on HF (`edomerli/nnUNet-PanTS-regions`, `AbdomenAtlas/MedFormerPanTS`, `AbdomenAtlas/R-SuperPanTSMerlin`) [V-doc] |

**File layout** [V-code: `data/README.md`, plus 3 cases from the label tarball streamed and inspected with nibabel]:

```
PanTS/
├── metadata.xlsx
├── ImageTr/PanTS_00000001/ct.nii.gz          (ImageTe/ for 00009001–00009901)
├── ReportTr/PanTS_00000001/report.pdf        (ReportTe/)
└── LabelTr/PanTS_00000001/
    ├── combined_labels.nii.gz                ← present in the tarball but NOT listed in data/README.md
    └── segmentations/
        ├── pancreatic_lesion.nii.gz
        ├── pancreas.nii.gz, pancreas_head.nii.gz, pancreas_body.nii.gz, pancreas_tail.nii.gz
        └── ... (28 binary files, one per class; files are always present, even when empty)
```

**Class map** (`class_map_abdomenatlas_pants`, from `data/README.md`) [V-code]:

| id | name | id | name | id | name | id | name |
|---|---|---|---|---|---|---|---|
| 1 | adrenal_gland_left | 8 | duodenum | 15 | lung_left | 22 | postcava |
| 2 | adrenal_gland_right | 9 | femur_left | 16 | lung_right | 23 | prostate |
| 3 | aorta | 10 | femur_right | 17 | pancreas | 24 | spleen |
| 4 | bladder | 11 | gall_bladder | 18 | pancreas_body | 25 | stomach |
| 5 | celiac_artery | 12 | kidney_left | 19 | pancreas_head | 26 | superior_mesenteric_artery |
| 6 | colon | 13 | kidney_right | 20 | pancreas_tail | 27 | veins |
| 7 | common_bile_duct | 14 | liver | 21 | pancreatic_duct | 28 | pancreatic_lesion |

**Pitfalls found by inspecting real label files** [V-code]:

1. **The per-structure masks overlap.** `pancreas` is the union of `pancreas_head`, `pancreas_body` and `pancreas_tail` (and contains the duct and lesion). `veins` overlaps pancreas and liver voxels. As a result `combined_labels.nii.gz` (int16, ids 1–28 above) is **lossy**: in case `PanTS_00001349` only 906 of the 16,106 pancreas voxels keep id 17, and the rest were overwritten by ids 18–21 and 27. **Evaluate from `segmentations/*.nii.gz`, not from `combined_labels`.** The only exception is when you deliberately want the "last label wins" semantics.
2. **The pancreas sub-part masks are not exact integers once loaded.** `pancreas_head`, `pancreas_body` and `pancreas_tail` are stored as int8 but load via nibabel as `{0.0, 1.0000000591}`. The other masks load as `{0, 1}`. **Binarise with `> 0.5` (or `!= 0`), never with `== 1`.**
3. **Orientation and axis order vary per case.** We saw `('I','P','L')` with zooms `(5.0, 0.943, 0.943)`, `('L','P','S')` and `('R','A','S')`. The slice axis is not always the last array axis. Take spacing per axis from the header or affine. Never assume z is the last axis, and never assume RAS.
4. **Empty files are common.** Masks with zero voxels are shipped for absent structures. For example, `pancreatic_lesion`, `spleen` and `kidney_left` are empty in `PanTS_00001349`. Most PanTS cases are tumour-free, so the empty-GT policy dominates the lesion DSC.
5. **Paths differ between the label tarball and the final layout.** The tarball holds flat `PanTS_XXXXXXXX/` folders. The download script moves them into `LabelTr/` and `LabelTe/` by ID range.

### 1.2 TotalSegmentator (CT and MRI; University Hospital Basel)

| Field | CT dataset | MRI dataset |
|---|---|---|
| Latest version | **v3.0.0** (2026-09-10): **1,939 CT**, 117 structures. v3 adds 291 paediatric CTs, fixes label errors (bones, femur) and **fixes vertebra label mix-ups**. Zenodo record **6802613**, `Totalsegmentator_dataset_v300.zip` (37.4 GB) [V-code: Zenodo API] | **v3.0.0** (2026-09-10): **1,296 MR**, 50 structures. Zenodo record **11367004**, `TotalsegmentatorMRI_dataset_v300.zip` (15.2 GB) [V-code] |
| Previous versions | v2.0.1 (Zenodo 10047292, 2023-10-27): 1,228 CT, 117 classes, `meta.csv` split = **train 1082 / val 57 / test 89** [V-code: counted from `meta.csv`]. v1 had 104 classes (`total_v1` in the class map) [V-code] | v2.0.0 (Zenodo 14710732): 616 MR. v1: 298 MR [V-code] |
| Licence | Zenodo metadata: **CC BY 4.0** [V-code]. The README says the datasets and most weights are "openly available for any usage (Apache-2.0)" and that some subtasks need a licence [V-doc]. | Zenodo v3: CC BY 4.0. **Zenodo v2.0.0 metadata says `cc-by-nc-sa-2.0`** [V-code]. Check which version you use. |
| Layout | `sXXXX/ct.nii.gz` + `sXXXX/segmentations/<class>.nii.gz` (**one binary file per class**, 117 files) + top-level `meta.csv` (**semicolon-separated**, UTF-8 BOM; columns `image_id;age;gender;institute;study_type;split;manufacturer;scanner_model;kvp;pathology;pathology_location`) [V-code: v2.0.1 zip central directory listed remotely] | Same scheme (`sXXXX/mri.nii.gz` + `segmentations/`) [UNVERIFIED: MR image filename] |
| Official metrics | The papers report Dice and NSD per class (Wasserthal et al., *Radiology: AI* 2023). **There is no challenge and no official evaluation script** [V-doc]. The NSD tolerance used in the paper is not stated here [UNVERIFIED: 3 mm?]. | Same [UNVERIFIED] |

**Model output conventions** (TotalSegmentator tool) [V-doc/V-code]:

- The default is one binary NIfTI per class in the output folder (`<class>.nii.gz`). `--ml` writes a single multi-label NIfTI whose ids follow `class_map[task]` in `totalsegmentator/map_to_binary.py`.
- Models run at 1.5 mm. `--fast` runs at 3 mm. Outputs are resampled back to the input geometry.
- `map_to_binary.py` defines about 55 task maps (`total`, `total_mr`, `lung_vessels`, `body`, `vertebrae_mr`, `liver_segments`, `teeth`, `brain_aneurysm`, …).

**`total` (CT) class map, 117 classes** [V-code, parsed from `map_to_binary.py`]:

```
1 spleen  2 kidney_right  3 kidney_left  4 gallbladder  5 liver  6 stomach  7 pancreas
8 adrenal_gland_right  9 adrenal_gland_left  10 lung_upper_lobe_left  11 lung_lower_lobe_left
12 lung_upper_lobe_right  13 lung_middle_lobe_right  14 lung_lower_lobe_right  15 esophagus
16 trachea  17 thyroid_gland  18 small_bowel  19 duodenum  20 colon  21 urinary_bladder
22 prostate  23 kidney_cyst_left  24 kidney_cyst_right  25 sacrum  26 vertebrae_S1
27 vertebrae_L5 28 L4 29 L3 30 L2 31 L1 32 T12 33 T11 34 T10 35 T9 36 T8 37 T7 38 T6 39 T5
40 T4 41 T3 42 T2 43 T1 44 C7 45 C6 46 C5 47 C4 48 C3 49 C2 50 vertebrae_C1   (all "vertebrae_<X>")
51 heart  52 aorta  53 pulmonary_vein  54 brachiocephalic_trunk  55 subclavian_artery_right
56 subclavian_artery_left  57 common_carotid_artery_right  58 common_carotid_artery_left
59 brachiocephalic_vein_left  60 brachiocephalic_vein_right  61 atrial_appendage_left
62 superior_vena_cava  63 inferior_vena_cava  64 portal_vein_and_splenic_vein
65 iliac_artery_left  66 iliac_artery_right  67 iliac_vena_left  68 iliac_vena_right
69 humerus_left  70 humerus_right  71 scapula_left  72 scapula_right  73 clavicula_left
74 clavicula_right  75 femur_left  76 femur_right  77 hip_left  78 hip_right  79 spinal_cord
80 gluteus_maximus_left  81 gluteus_maximus_right  82 gluteus_medius_left  83 gluteus_medius_right
84 gluteus_minimus_left  85 gluteus_minimus_right  86 autochthon_left  87 autochthon_right
88 iliopsoas_left  89 iliopsoas_right  90 brain  91 skull
92–103 rib_left_1 … rib_left_12   104–115 rib_right_1 … rib_right_12
116 sternum  117 costal_cartilages
```

**`total_mr` class map, 50 classes** [V-code]:

```
1 spleen 2 kidney_right 3 kidney_left 4 gallbladder 5 liver 6 stomach 7 pancreas
8 adrenal_gland_right 9 adrenal_gland_left 10 lung_left 11 lung_right 12 esophagus
13 small_bowel 14 duodenum 15 colon 16 urinary_bladder 17 prostate 18 sacrum 19 vertebrae
20 intervertebral_discs 21 spinal_cord 22 heart 23 aorta 24 inferior_vena_cava
25 portal_vein_and_splenic_vein 26 iliac_artery_left 27 iliac_artery_right 28 iliac_vena_left
29 iliac_vena_right 30 humerus_left 31 humerus_right 32 scapula_left 33 scapula_right
34 clavicula_left 35 clavicula_right 36 femur_left 37 femur_right 38 hip_left 39 hip_right
40 gluteus_maximus_left 41 gluteus_maximus_right 42 gluteus_medius_left 43 gluteus_medius_right
44 gluteus_minimus_left 45 gluteus_minimus_right 46 autochthon_left 47 autochthon_right
48 iliopsoas_left 49 iliopsoas_right 50 brain
```

**Pitfalls:**

- Class **names** are stable across CT v2 and v3, but **ids differ from v1** (104 classes).
- The MR ids **differ from the CT ids** for the same organ, for example `lung_left` (MR 10) versus the CT lobes (10–14). A generic "TotalSegmentator" evaluator must therefore key on task plus version, or on names.
- Structures cut off at the edge of the field of view are often left empty in the ground truth, so empty-GT handling matters for partial-FOV scans [UNVERIFIED as to the annotation policy].

### 1.3 MSD Task03_Liver (Medical Segmentation Decathlon)

| Field | Value [V-code: `dataset.json` extracted from the official MSD tar at `msd-for-monai.s3-us-west-2.amazonaws.com/Task03_Liver.tar`] |
|---|---|
| `name` / `description` | `"Liver"` / `"Liver, and cancer segmentation"` |
| Modality | `{"0": "CT"}`, `tensorImageSize "3D"` |
| **Labels** | **`{"0": "background", "1": "liver", "2": "cancer"}`**. This is a single multi-label NIfTI in which the tumour (2) is *inside* the liver: liver voxels = label 1 ∪ label 2. |
| Cases | `numTraining 131`, `numTest 70` (test labels are withheld) |
| Files | `Task03_Liver/imagesTr/liver_{N}.nii.gz`, `labelsTr/liver_{N}.nii.gz`, `imagesTs/liver_{N}.nii.gz`. Training N is 0–130 and test N is 132–201 (the first test entry in `dataset.json` is `liver_132`) [training ids 0–130 are UNVERIFIED; the test start is from `dataset.json`] |
| Licence | `"licence": "CC-BY-SA 4.0"`, release `"1.0 04/05/2018"` |
| Source | Derived from **LiTS 2017** (same 131 train / 70 test volumes) [V-doc] |
| Spacing | See Section 1.5 (computed from the headers) |
| Official metrics | MSD: **DSC and NSD** per label (liver, cancer), ranked by significance ranking [V-doc: Antonelli et al., Nat Commun 2022]. See Section 3 for the per-task NSD tolerances. |

### 1.4 MSD Task10_Colon

| Field | Value [V-code: `dataset.json` from `Task10_Colon.tar`] |
|---|---|
| `name` / `description` | `"Colon"`, release `"1.0 06/08/2018"` |
| Modality | `{"0": "CT"}` |
| **Labels** | **`{"0": "background", "1": "colon cancer primaries"}`** (binary) |
| Cases | `numTraining 126`, `numTest 64` |
| Files | `Task10_Colon/imagesTr/colon_{NNN}.nii.gz`, `labelsTr/colon_{NNN}.nii.gz`, `imagesTs/colon_{NNN}.nii.gz`. The ids are **non-contiguous, zero-padded 3-digit numbers** (for example `colon_219` in training and `colon_198` in test) |
| Licence | CC-BY-SA 4.0 |
| Official metrics | DSC and NSD (MSD) |

**Layout gotchas common to all MSD tars** [V-code, from streaming the tar headers]:

- The tars contain **macOS AppleDouble files**: `._Task10_Colon`, `Task10_Colon/._imagesTr`, `Task03_Liver/imagesTr/._liver_14.nii.gz`, and so on. **Any `glob('*.nii.gz')` picks up `._*.nii.gz` junk files of 187–331 bytes.** Filter out names that start with `._`.
- `dataset.json` entries use `./imagesTr/...` relative paths. Several tasks spell the release key `"relase"` (a typo).
- Tasks 01 and 05 are 4D, multi-channel in one file (`tensorImageSize "4D"`).

### 1.5 Voxel-geometry statistics for Task03 and Task10

These were computed by streaming the official tars and reading every NIfTI header (the label files were read in full).

**Task10_Colon** [V-code, all 316 volumes]:

| Split | n | In-plane spacing (mm), min / median / max | Slice spacing (mm), min / median / max | Shape |
|---|---|---|---|---|
| imagesTr / labelsTr | 126 | 0.535 / 0.781 / 0.977 | 1.25 / 5.0 / 7.5 | 512×512×(37–729) |
| imagesTs | 64 | 0.578 / 0.768 / 0.977 | 2.2 / 5.0 / 6.0 | 512×512×(36–263) |

- Case ids: train 1–219 (126 ids), test 3–209 (64 ids). There are gaps in both.
- **Labels are stored as float32** (the images are float32 too). Cast them to an integer type before comparing labels.
- **All 126 training labels contain `{0, 1}`, so no training case has an empty tumour mask.**

**Task03_Liver:** Not computed. Streaming the 27 GB tar hit the 1-hour background limit, so this scan did not finish. The LiTS paper (Bilic et al.) reports in-plane spacing of about 0.56–1.0 mm and slice thickness of about 0.45–6.0 mm [UNVERIFIED]. Some LiTS/Task03 training cases are known to have **no tumour (label 2 absent)**, so the empty-GT policy affects the cancer DSC [UNVERIFIED count; re-run `msdstats.py` locally to confirm].

---

---

## 2. Dataset catalogue

Abbreviations: **ML** = one multi-label integer NIfTI per case. **PS** = per-structure binary files. **Inst** = instance or per-annotator files. Tags follow the conventions in the introduction.

### 2.1 Abdominal and whole-body CT/MR (multi-organ)

| Dataset | Modality | Targets / label ids | Cases | Label format | Spacing | Licence | Download | Tag |
|---|---|---|---|---|---|---|---|---|
| **AbdomenAtlas 1.0 (Mini)** | CT | 1 aorta, 2 gall_bladder, 3 kidney_left, 4 kidney_right, 5 liver, 6 pancreas, 7 postcava, 8 spleen, 9 stomach | 5,195 public ("Mini"); the paper describes 8,448 | Case folder `BDMAP_XXXXXXXX/` with `ct.nii.gz`, `combined_labels.nii.gz` (ML) and `segmentations/*.nii.gz` (PS) | Multi-source | HF card: CC BY-NC-SA 4.0 (gated, no redistribution) | HF `AbdomenAtlas/AbdomenAtlas1.0Mini`, `BodyMaps/AbdomenAtlas1.0Mini`; Dropbox; Baidu | V-doc |
| **AbdomenAtlas 1.1 (Mini)** | CT | The 1.0 ids 1–9, plus 10 adrenal_gland_left, 11 adrenal_gland_right, 12 bladder, 13 celiac_trunk, 14 colon, 15 duodenum, 16 esophagus, 17 femur_left, 18 femur_right, 19 hepatic_vessel, 20 intestine, 21 lung_left, 22 lung_right, 23 portal_vein_and_splenic_vein, 24 prostate, 25 rectum | 9,262 (`BDMAP_00000001` to `BDMAP_00009262`) | As 1.0 | Multi-source (17 public datasets) | CC BY-NC-SA 4.0 | HF `BodyMaps/AbdomenAtlas1.1Mini` (gated; the mask-only repo is `…1.1MiniMask`) | V-doc |
| **AbdomenAtlas 3.0** (RadGPT, ICCV 2025) | CT | Per-voxel liver, kidney and pancreatic tumour masks plus structured and narrative reports, on the same 9,262 CTs (3,955 with tumours) | 9,262 | PS (AbdomenAtlas style) [UNVERIFIED: exact file names] | — | [UNVERIFIED; likely CC BY-NC-SA] | Via BodyMaps/AbdomenAtlas HF [UNVERIFIED: repo id] | V-doc (counts) |
| **Merlin** (Stanford) | CT (abdomen/pelvis, emergency department 2012–2018) | **No voxel masks are distributed.** You get reports (Findings), EHR-derived labels, zero-shot disease labels and 5-year disease tasks. The Merlin segmentation work uses nnU-Net (`ashwinkumargb/Merlin-nnUNet`) [UNVERIFIED: which organ labels it uses] | 25,494 scans / 18,317 patients. The split is in `reports_final.xlsx` | Images as `.nii` in `merlin_data/` | — | Stanford AIMI DUA | stanfordaimi.azurewebsites.net (azcopy) | V-doc |
| **TotalSegmentator** | CT / MR | See §1.2 | 1,939 CT / 1,296 MR (v3) | PS | Varied | CC BY 4.0 | Zenodo 6802613 / 11367004 | V-code |
| **AMOS 2022** | CT + MRI | 1 spleen, 2 right kidney, 3 left kidney, 4 gall bladder, 5 esophagus, 6 liver, 7 stomach, 8 aorta (spelled `"arota"` in `dataset.json`), 9 postcava/IVC, 10 pancreas, 11 right adrenal, 12 left adrenal, 13 duodenum, 14 bladder, 15 prostate/uterus | Tr 200 CT + 40 MR; Va 100 CT + 20 MR; Ts 240 (labels withheld). **Case id < 500 means CT, ≥ 500 means MR** | ML (`amos_XXXX.nii.gz`) | CT 0.57–0.78 mm in-plane, 5 mm slices; MR 0.69–1.19 mm in-plane, 3 mm slices (sampled) | Zenodo: CC BY 4.0; `dataset.json`: CC BY-SA 4.0 (conflict) | Zenodo 7262581 (`amos22.zip`, 24 GB) | V-code |
| **BTCV** (Synapse multi-atlas, abdomen) | CT (portal venous) | 1 spleen, 2 right kidney, 3 left kidney, 4 gallbladder, 5 esophagus, 6 liver, 7 stomach, 8 aorta, 9 IVC, 10 portal and splenic vein, 11 pancreas, 12 right adrenal, 13 left adrenal. "Some patients may not have (2) right kidney or (4) gallbladder" | 30 train / 20 test | ML: `img/img0001.nii.gz`, `label/label0001.nii.gz` | 0.54–0.98 mm in-plane, 2.5–5 mm slices | Synapse terms [UNVERIFIED] | Synapse syn3193805 (login) | V-doc |
| **WORD** | CT | 16 organs, in the paper's order: liver, spleen, kidney_L, kidney_R, stomach, gallbladder, esophagus, pancreas, duodenum, colon, intestine, adrenal, rectum, bladder, femoral head L, femoral head R. **Ids 1–16 in this order are UNVERIFIED** | 100 / 20 / 30 | ML | 0.976 mm in-plane, 2.5–3 mm slices | GPL-3.0 (academic) | Google Drive / Baidu (password by email) | V-doc |
| **AbdomenCT-1K** | CT | 1 liver, 2 kidney, 3 spleen, 4 pancreas | 1,112 in total; fully-supervised subtask 361 train; shared test 100 | ML | Multi-centre | Data licence [UNVERIFIED]; repo Apache-2.0 | Google Form (github.com/JunMa11/AbdomenCT-1K) | V-doc |
| **FLARE21** | CT | 1 liver, 2 kidney, 3 spleen, 4 pancreas | 361 train / 50 val / 100 test [UNVERIFIED splits] | ML | — | [UNVERIFIED] | flare.grand-challenge.org | V-code (labels from eval code) |
| **FLARE22** | CT | 1 liver, 2 right kidney, 3 spleen, 4 pancreas, 5 aorta, 6 IVC, 7 right adrenal, 8 left adrenal, 9 gallbladder, 10 esophagus, 11 stomach, 12 duodenum, 13 left kidney | 50 labelled + 2,000 unlabelled train; 200 test [UNVERIFIED] | ML | — | [UNVERIFIED] | flare22.grand-challenge.org | V-code (ids from eval code) |
| **FLARE23** | CT | FLARE22 ids 1–13 plus **14 lesion** (pan-cancer) | 4,000 partially labelled [UNVERIFIED] | ML | — | [UNVERIFIED] | codabench / grand-challenge | V-code (ids) |
| **FLARE24** | CT / MR | Task 1: pan-cancer lesion (binary). Tasks 2–3: organs on CT (Task 2) and MRI (Task 3), using the FLARE22 ids [UNVERIFIED] | [UNVERIFIED] | ML | — | [UNVERIFIED] | FLARE24 site | V-code (metric) |
| **CHAOS** | CT (liver) + MR T1-DUAL in/out phase and T2-SPIR (liver, kidneys, spleen) | MR PNG intensities: liver 63 (range 55–70), right kidney 126 (110–135), left kidney 189 (175–200), spleen 252 (240–255). The CT mask is binary liver | CT 20/20; MR 20/20 | **DICOM + PNG slices (not NIfTI)** | — | CC BY-NC-SA 4.0 | Zenodo 3431873 | V-doc |
| **CT-ORG** | CT | 1 liver, 2 bladder, 3 lungs, 4 kidneys, 5 bone, 6 brain | 119 / 21 | ML (`labels-XX.nii.gz`, **float32**) | — | CC BY 3.0 | TCIA | V-doc |
| **SAROS** | CT (900 scans, 28 TCIA collections) | `body-regions.nii.gz`: 1 subcutaneous tissue, 2 muscle, 3 abdominal cavity, 4 thoracic cavity, 5 bones, 6 parotid glands, 7 pericardium, 8 breast implant, 9 mediastinum, 10 brain, 11 spinal cord, 12 thyroid glands, 13 submandibular glands. `body-parts.nii.gz`: 1 torso, 2 head, 3 right leg, 4 left leg, 5 right arm, 6 left arm | 5 folds × 150 plus test 150 | ML, **sparse: every 5th axial slice annotated, all other slices = 255 (ignore)** | — | Segmentations [UNVERIFIED]; paper CC BY 4.0 | TCIA | V-doc |

### 2.2 Kidney, liver and other tumour CT

| Dataset | Labels | Cases | Format | Licence | Download | Tag |
|---|---|---|---|---|---|---|
| **KiTS19** | 0 background, 1 kidney, 2 tumour | 210 / 90 | `case_XXXXX/imaging.nii.gz` + `segmentation.nii.gz` (ML) | Data CC BY-NC-SA 4.0 | `kits19` starter code; HF `neheller/KiTS-Challenge-Imaging` | V-doc |
| **KiTS21** | 1 kidney, 2 tumour, 3 cyst. Aggregation order (1, 3, 2): tumour overwrites cyst, which overwrites kidney | 300 / 100 [test UNVERIFIED] | ML aggregated (`aggregated_{OR,AND,MAJ}_seg.nii.gz`) + Inst `segmentations/{kidney,tumor,cyst}_instance-N_annotation-M.nii.gz` | [UNVERIFIED]; HF imaging CC BY-NC-SA | `neheller/kits21` | V-code |
| **KiTS23** | 1 kidney, 2 tumour, 3 cyst (same as KiTS21) | 489 train / 110 test [counts UNVERIFIED] | `dataset/case_XXXXX/imaging.nii.gz`, `segmentation.nii.gz` (+ instances) | Data CC BY-NC-SA; code MIT | `kits23_download_data` (HF) | V-code (labels, metric) |
| **KiPA22** | 1 renal vein, 2 kidney, 3 renal artery, 4 tumour [UNVERIFIED officially] | 70 / 30 open / 30 closed | NIfTI | [UNVERIFIED] | kipa22.grand-challenge.org | UNVERIFIED |
| **LiTS 2017** | 0 background, 1 liver, 2 tumour | 131 / 70 | ML (`volume-N.nii`, `segmentation-N.nii`) | [UNVERIFIED] | CodaLab / MSD Task03 | V-code (metric) |
| **ULS23** | Binary lesion inside a 256×256×128 VOI | Test 725 lesions / 268 patients. Training draws on DeepLesion3D, KiTS, LiTS, LIDC, MSD and others | NIfTI VOIs (padded with min − 1) | CC BY-NC-SA 4.0 | Zenodo 10035161; github DIAGNijmegen/ULS23 | V-doc |

### 2.3 Medical Segmentation Decathlon: all 10 tasks

All values below are from each task's `dataset.json`, extracted from the official AWS tars [V-code]. The licence is CC-BY-SA 4.0 for every task. Layout is `TaskXX_Name/{imagesTr,labelsTr,imagesTs}/<prefix>_<id>.nii.gz`, with ML labels. The NSD tolerances come from Antonelli et al. 2022 [V-doc].

| Task | Modality (channels) | Labels | Train / Test | File prefix | NSD tol (mm) |
|---|---|---|---|---|---|
| 01 BrainTumour | 4D MRI: 0 FLAIR, 1 T1w, 2 t1gd, 3 T2w | 1 edema, 2 non-enhancing tumor, 3 enhancing tumour (**not** the BraTS numbering) | 484 / 266 | `BRATS_NNN` | 5 |
| 02 Heart | MRI | 1 left atrium | 20 / 10 | `la_NNN` | 4 |
| 03 Liver | CT | 1 liver, 2 cancer | 131 / 70 | `liver_N` | 7 |
| 04 Hippocampus | MRI | 1 Anterior, 2 Posterior | 260 / 130 (the paper says 263 / 131) | `hippocampus_NNN` | 1 |
| 05 Prostate | 4D MRI: 0 T2, 1 ADC | 1 PZ, 2 TZ | 32 / 16 | `prostate_NN` | 4 |
| 06 Lung | CT | 1 cancer | 63 / 32 (the paper says 64 / 32) | `lung_NNN` | 2 |
| 07 Pancreas | CT | 1 pancreas, 2 cancer | 281 / 139 (the paper says 282) | `pancreas_NNN` | 5 |
| 08 HepaticVessel | CT | 1 Vessel, 2 Tumour | 303 / 140 | `hepaticvessel_NNN` | 3 |
| 09 Spleen | CT | 1 spleen | 41 / 20 | `spleen_N` | 3 |
| 10 Colon | CT | 1 colon cancer primaries | 126 / 64 | `colon_NNN` | 4 |

Note that the `numTraining` values in `dataset.json` differ from the paper's table for Hippocampus, Lung and Pancreas. Trust the files.

### 2.4 Neuro

| Dataset | Modality | Labels | Cases | Format / naming | Licence | Download | Tag |
|---|---|---|---|---|---|---|---|
| **BraTS 2021** | mpMRI (T1, T1Gd, T2, FLAIR), SRI24 atlas space, 1 mm³, 240×240×155 | 1 NCR, 2 ED, **4 ET**. Regions: ET = {4}, TC = {1, 4}, WT = {1, 2, 4} | 1251 train / 219 val / 570 test [test UNVERIFIED] | `BraTS2021_XXXXX_{flair,t1,t1ce,t2,seg}.nii.gz` [UNVERIFIED] | Synapse terms | Synapse | V-doc |
| **BraTS 2023** (GLI/MEN/MET/PED/SSA) | mpMRI | 1 NCR, 2 ED (SNFH), **3 ET**. Regions: WT = {1, 2, 3}, TC = {1, 3}, ET = {3} | GLI 1251 / 219 | `BraTS-GLI-XXXXX-XXX-{t1c,t1n,t2w,t2f,seg}.nii.gz` [UNVERIFIED exact] | [UNVERIFIED] | Synapse | V-code |
| **BraTS 2024** (post-treatment glioma, plus MEN-RT, MET, PED, SSA, …) | mpMRI | 1 NETC, 2 SNFH, 3 ET, **4 RC** (resection cavity) [UNVERIFIED against the data README]. TC = ET + NETC; WT = ET + SNFH + NETC | About 2,200 patients, 70/10/20 split | NIfTI | CC BY 4.0 [UNVERIFIED] | Synapse syn53708249 | V-doc (partial) |
| **BraTS 2025** | Clustered challenges (GLI, MEN, MEN-RT, METS, Africa, PED, GOAT) | — | — | — | — | Zenodo 15094823 (design document) | V-doc (existence) |
| **ISLES 2022** | MRI: DWI (b = 1000), ADC, FLAIR | Binary infarct | 250 train / 150 test (400 total) | BIDS: `rawdata/sub-strokecaseXXXX/ses-0001/*_{dwi,adc,flair}.nii.gz` and `derivatives/.../*_msk.nii.gz` | Data CC BY-SA 4.0 | grand-challenge; Zenodo | V-doc |
| **TopCoW 2023 / 2024** | CTA + MRA (paired), Circle of Willis | 1 BA, 2 R-PCA, 3 L-PCA, 4 R-ICA, 5 R-MCA, 6 L-ICA, 7 L-MCA, 8 R-Pcom, 9 L-Pcom, 10 Acom, 11 R-ACA, 12 L-ACA, **15 3rd-A2** (13 and 14 are unused) | 2023: 90 train / 5 val / 35 test pairs. 2024: 125 train pairs / 70+ test | `.nii.gz` / `.mha`, LPS, braincase-cropped; `cow_seg_labelsTr`, `roi_loc_labelsTr` | "Open use, cite source, commercial use needs permission" | Zenodo 15692630 | V-code (labels) |

### 2.5 PET/CT and head and neck

| Dataset | Modality | Labels | Cases | Format | Licence | Tag |
|---|---|---|---|---|---|---|
| **autoPET I (2022)** | FDG PET/CT (whole body) | Binary lesion (`SEG.nii.gz`). Many studies are lesion-negative | 1,014 studies / 900 patients. Test 150 (later described as 200) | `SUV.nii.gz`, `CTres.nii.gz`, `SEG.nii.gz` (TCIA FDG-PET-CT-Lesions) | TCIA CC BY-NC [UNVERIFIED exact version] | V-doc |
| **autoPET II (2023)** | FDG PET/CT | Binary | Same training data; test 200 | NIfTI / mha | same | V-doc |
| **autoPET III (2024)** | FDG (UKT, 1,014) + PSMA (LMU, 597) PET/CT; tracer unknown at test time | Binary | Test 200 = 4 strata × 50 (PSMA-LMU, FDG-UKT, PSMA-UKT, FDG-LMU) | NIfTI / mha | same | V-doc |
| **autoPET IV (2025)** | Interactive: PET/CT clicks (Task 1); longitudinal CT (Task 2) | Binary | — | — | — | V-doc |
| **autoPET V (2026)** | Scribble-based interactive | — | — | — | — | V-doc (existence) |
| **HECKTOR 2021** | FDG PET/CT, H&N | GTVp (binary) | 224 / 101 [UNVERIFIED] | NIfTI | [UNVERIFIED] | V-code (metric) |
| **HECKTOR 2022** | PET/CT | 1 GTVp, 2 GTVn | 524 / 359 [UNVERIFIED] | NIfTI | [UNVERIFIED] | V-doc |
| **HECKTOR 2025** | PET/CT | 1 GTVp, 2 GTVn | About 700 / about 450 | `__CT.nii.gz`, `__PT.nii.gz` | [UNVERIFIED] | V-doc |

### 2.6 Cardiac, thoracic and vascular

| Dataset | Modality | Labels | Cases | Format | Licence | Tag |
|---|---|---|---|---|---|---|
| **ACDC** | Cine SA MRI (ED/ES) | **1 RV, 2 MYO, 3 LV** | 100 / 50 | `patientXXX_frameYY.nii.gz` + `_gt.nii.gz` | CC BY-NC-SA 4.0 [UNVERIFIED primary] | V-code (labels) |
| **M&Ms-1** | Cine SA MRI, 4 vendors | **1 LV, 2 MYO, 3 RV (the reverse of ACDC!)** | 375 subjects: 150 labelled + 25 unlabelled train / 34 val / 136 test [UNVERIFIED] | `<code>_sa.nii.gz`, `_sa_gt.nii.gz` | [UNVERIFIED] | V-code (labels via nnU-Net conversion) |
| **M&Ms-2** | SA + LA cine MRI | 1 LV, 2 MYO, 3 RV | 160 / 40 / 160 | `{id}_SA_ED(_gt).nii.gz` [UNVERIFIED] | [UNVERIFIED] | V-doc |
| **LA 2018 (Utah)** | 3D LGE-MRI | LA cavity, **0/255** | 154 volumes (100 / 54 [UNVERIFIED]) | **NRRD** (`lgemri.nrrd`, `laendo.nrrd`) | [UNVERIFIED] | V-doc |
| **MSD Heart (Task02)** | MRI | 1 left atrium | 20 / 10 | NIfTI | CC BY-SA 4.0 | V-code |
| **MM-WHS 2017** | CT + MRI | **Intensity codes:** 205 MYO, 420 LA, 500 LV, 550 RA, 600 RV, 820 ascending aorta, 850 pulmonary artery | 20+20 train / 40+40 test | NIfTI (`ct_train_1001_label.nii.gz` [UNVERIFIED]) | [UNVERIFIED] | UNVERIFIED (secondary) |
| **SegTHOR** | CT | 1 esophagus, 2 heart, 3 trachea, 4 aorta | 40 / 20 | `Patient_XX/GT.nii.gz` | [UNVERIFIED] | V-code (via nnU-Net conversion) |
| **Parse2022** | CT | Pulmonary artery (binary [UNVERIFIED]) | 100 / 30 / 70 | NIfTI [UNVERIFIED] | [UNVERIFIED] | V-doc |
| **LNQ2023** | Contrast CT | Mediastinal lymph nodes. **Training masks are partial** (a subset of nodes); val/test are fully annotated (nodes ≥ 1 cm) | 383 / 20 / 100 | **NRRD** | [UNVERIFIED] | V-doc |
| **LUNA16** | CT | Nodule centroids and diameters (**detection, no masks**) | 888 scans, 1,186 nodules (≥ 3 mm, ≥ 3 of 4 readers), 10 subsets | **MetaImage `.mhd/.raw`** + CSV | CC BY (via LIDC) [UNVERIFIED] | V-doc |
| **LIDC-IDRI** | CT | Up to 4 readers' nodule contours | 1,010 patients / 1,308 studies | **DICOM + XML** (use `pylidc`) | CC BY 3.0 | V-doc |
| **Duke Breast Cancer MRI** | DCE-MRI | 3D tumour **bounding boxes** for all cases. Breast/FGT (and vessel) segmentations for about 100–127 cases. **No official tumour masks** | 922 patients | **DICOM** | CC BY-NC 4.0 | V-doc |
| **PI-CAI** | bpMRI (T2W, ADC, HBV) | csPCa lesions. Expert labels: 0 = ISUP ≤ 1, 2–5 = ISUP grade. AI-derived labels are binary | 1,500 public (1,476 patients; 425 csPCa) | Images **`.mha`** `{pid}_{sid}_{t2w,adc,hbv}.mha`; labels NIfTI (picai_labels) | CC BY-NC 4.0 | V-code |

### 2.7 Dental and spine

| Dataset | Modality | Labels | Cases | Format | Licence | Tag |
|---|---|---|---|---|---|---|
| **ToothFairy (2023)** | CBCT | Binary inferior alveolar canal (IAC) | 443 (153 dense) | `.npy` → `.mha` | [UNVERIFIED] | V-code (eval) |
| **ToothFairy2 (2024)** | CBCT | 42 classes: 1 lower jawbone, 2 upper jawbone, 3 left IAC, 4 right IAC, 5 left and 6 right maxillary sinus, 7 pharynx, 8 bridge, 9 crown, 10 implant, teeth by FDI code 11–18, 21–28, 31–38, 41–48 | 480 / 50 | `.mha` | CC BY-NC-SA [UNVERIFIED] | V-code |
| **ToothFairy3 (2025)** | CBCT | 77 raw labels (adds 103/104 incisive canals, 105 lingual canal, pulp 111–148). **Pulp labels are merged to 150 for evaluation** | 532 | NIfTI | CC BY-NC-SA | V-code |
| **SPIDER** | Lumbar MR (T1, T2, T2-SPACE) | Vertebrae 1..N and IVDs 201..N, numbered **bottom-up** (not anatomical); 100 = spinal canal | 447 series / 218 patients (179 / 39 + hidden 39) | **`.mha`** int16 | CC BY 4.0 | V-code (masks sampled) |
| **VerSe 2019 / 2020** | CT | 1–7 C1–C7, 8–19 T1–T12, 20–25 L1–L6, 28 T13 | '19: 80/40/40; '20: 113/103/103 | BIDS NIfTI `*_seg-vert_msk.nii.gz` + centroid JSON | CC BY-SA 4.0 | V-code |

---

## 3. Official evaluation protocols (metrics, tolerances, empty masks)

### 3.1 Quick reference table

| Benchmark | Official metrics | Tolerance / threshold | Empty GT and empty pred | Empty GT, non-empty pred (FP) | Non-empty GT, empty pred (FN) | Aggregation / ranking | Tag |
|---|---|---|---|---|---|---|---|
| **MSD** | DSC, NSD | NSD τ per task: Brain 5, Heart 4, Hippo 1, Liver 7, Lung 2, Prostate 4, Pancreas 5, Colon 4, HepVessel 3, Spleen 3 mm | Paper: "set to 0 if undefined" [exact both-empty behaviour UNVERIFIED] | 0 | 0 | Wilcoxon significance ranking per region, averaged over tasks | V-doc |
| **FLARE21** | DSC, NSD | **1 mm for all organs** | DSC = NSD = 1 | 0 | *Not guarded*: falls into the general branch → NaN/inf risk (a bug) | Mean per organ | V-code |
| **FLARE22** | DSC, NSD (+ runtime, GPU) | **Per-organ τ**: Liver 5, RK 3, Spleen 3, Pancreas 5, Aorta 2, IVC 2, RAG 2, LAG 2, Gallbladder 2, Esophagus 3, Stomach 5, Duodenum 7, LK 3 mm | 1 / 1 | 0 / 0 | 0 / 0 | Mean. **NSD is forced to 0 if DSC < 0.2.** Aorta, IVC and esophagus are evaluated only on GT-labelled z-slices `[z_min:z_max)` (the last slice is excluded, an off-by-one) | V-code |
| **FLARE23** | DSC, NSD | FLARE22 τ + **Lesion 2 mm** | 1 / 1 | 0 | 0 | Organ_DSC/NSD = mean over the 13 organs (lesion reported separately) | V-code |
| **FLARE24 T1** | Lesion DSC, NSD | 2 mm | 1 | 0 | 0 | **If the prediction has any value > 1, the score is 0** (binary expected) | V-code |
| **FLARE24 T2/T3** | Organ DSC, NSD | FLARE22 τ | 1 | 0 | 0 | same as FLARE22 | V-code |
| **AMOS22** | DSC, NSD | [UNVERIFIED τ] | [UNVERIFIED] | | | Rank-then-aggregate | V-doc |
| **KiTS19** | Dice (kidney+tumour, tumour) | — | [UNVERIFIED] | | | Mean composite Dice | V-doc |
| **KiTS21 / 23** | Dice, **Surface Dice** on HECs: kidney+masses (1, 2, 3), masses (2, 3), tumour (2) | **SD τ: 1.0331, 1.1329, 1.1498 mm** respectively (from inter-observer variability) | Dice = SD = 1 | 0 | 0 | Mean over cases and HECs, then rank-then-aggregate. KiTS21 test uses **sampled multi-annotator references** (5 groups). KiTS23 compares against `segmentation.nii.gz` | V-code |
| **LiTS** | Dice per case (lesion ranking), Dice global, VOE, RVD, ASSD, MSD | — | medpy `dc` → 0 | — | ASSD = MSD = 0 when either mask is empty (**not** penalised) | Per-case lesion Dice | V-code |
| **BraTS 2021** | Dice, HD95 on ET/TC/WT (+ sensitivity, specificity) | — | Dice 1, HD95 0 [convention; official code not public] | Dice 0, HD95 373.13 [UNVERIFIED] | Dice 0, HD95 373.13 | Per-case rank, then permutation tests | V-doc |
| **BraTS 2023+** | **Lesion-wise** Dice and HD95 on WT/TC/ET | cc3d 26-connectivity; GT dilation (3D, 2-conn struct) × *d* iterations; lesions with GT volume ≤ *v* mm³ dropped. *(d, v)*: GLI (3, 50), SSA (3, 50), PED (3, 50), MEN (1, 50), MET (1, 2) | 0/0 → NaN → **Dice 1, HD95 0** | Each FP component adds 0 to the Dice numerator and **374** to the HD95 numerator (and +1 to the denominator) | Missed lesion: Dice 0, HD95 inf → **374** | Σ(lesion scores) / (n_GT_lesions + n_FP). FPs are **not** volume-filtered | V-code |
| **ISLES22** | Dice, absolute volume difference (ml), absolute lesion-count difference, lesion-wise F1 | cc3d 26-connectivity; any 1-voxel overlap counts as a hit | Dice = 1, F1 = 1 (`empty_value`) | Dice 0; F1 = 0 | Dice 0; F1 = 0 | Rank per metric → mean rank | V-code |
| **autoPET I–III** | Dice, **FP volume** (ml), **FN volume** (ml) | cc3d **18-connectivity**; FP = pred components with zero GT overlap; FN = GT components with zero pred overlap | Dice = 0/0 = NaN; **negatives use FPV only** | FPV counted; Dice not used | Dice 0; FNV = GT volume | Rank per metric; final = 0.5·rank_Dice + 0.25·rank_FPV + 0.25·rank_FNV (III: per tracer×centre stratum) | V-code |
| **autoPET IV** | Task 1: DSC@last, AUC-DSC, FPV@last, FNV@last, AUC-FPV, AUC-FNV over 11 click steps (weights 0.25/0.25/0.125×4). Task 2: DSC, FPV, FNV (0.5/0.25/0.25) | — | as above | | | | V-doc |
| **HECKTOR 2021** | DSC (mean), HD95 (median) | Crop to bbox | — | — | Dice 0, HD95 0 when pred empty (as coded) | Mean DSC / median HD95 | V-code (per agent) |
| **HECKTOR 2022** | **Aggregated DSC** per class: 2ΣTP / Σ(\|G\|+\|P\|) over all test cases | Spacing must match CT within 1e-6 | Contributes nothing (0/0 avoided by aggregation) | Adds \|P\| to the denominator | Adds \|G\| | Mean of GTVp and GTVn DSCagg | V-doc |
| **HECKTOR 2025** | GTVp: mean DSC. GTVn: aggregated DSC + aggregated F1 (lesion match IoU > 0.3) | | | | | Borda count | V-doc |
| **TopCoW 2024** | Class-average Dice, clDice (merged binary), Betti-0 error, HD95, Group-2 detection F1 (IoU ≥ 0.25), graph classification, topology match | HD95 = max(d95(A,B), d95(B,A)); **HD95 upper bound 90 mm** | Class skipped (only the union of labels present in GT ∪ pred is scored) | Dice 0, HD95 90, b0 error = b0(pred) | Dice 0, HD95 90 | Equal-weight ranks | V-code |
| **TopCoW 2023** | Dice, clDice, Betti-0 (not Betti-1) | | | | | | V-doc |
| **ToothFairy2/3** | Dice, HD95 (mean over classes) | **HD95 in voxels** (medpy with no spacing) | Dice 1, HD95 0 | HD95 = volume diagonal ‖shape‖ in voxels | same | Mean | V-code |
| **ULS23** | 0.8·SegPerf (Dice) + 0.05·LAE + 0.05·SAE + 0.1·consistency | SMAPE for long/short axis | — | — | Missing prediction → max error 1 | | V-doc |
| **PI-CAI** | (AUROC + AP)/2 | Hit if IoU ≥ 0.10 with a GT lesion (Hungarian one-to-one matching) | Benign case: every prediction is an FP | | | Ranking score | V-code |
| **LUNA16** | CPM: mean sensitivity at 1/8…8 FP/scan | Hit if within nodule radius | — | | | | V-doc |
| **WORD** | DSC, HD95 (medpy) | — | **DSC 0, HD95 50** (whenever either mask is empty, including both!) | 0 / 50 | 0 / 50 | Classes looped over `range(1, gt.max()+1)` | V-code |
| **CHAOS** | DICE, RAVD, ASSD, MSSD, each mapped to 0–100 (thresholds 0.8, 5%, 15 mm, 60 mm) | — | No guard (crash/NaN) | | | | V-code |
| **ACDC** | Dice, HD (mm), clinical indices | — | medpy → 0 | | | | V-doc |
| **SegTHOR** | Dice, **Average Hausdorff** (max of the directed mean distances) | — | [UNVERIFIED] | | | | V-doc |
| **VerSe** | Dice, **full HD**, id.rate (≤ 20 mm), d_mean | — | Dice 1 | Missing vertebrae excluded from HD | | | V-code |
| **SPIDER** | Dice, ASD per structure, per class group | Structures matched to predictions by largest overlap | | | | | V-doc |
| **LNQ2023** | Dice, ASSD | — | | | | Per-patient rank averaging | V-doc |
| **KiPA22** | Dice, HD, AVD | — | [UNVERIFIED] | | | | V-doc |
| **PanTS** | DSC (lesion), (NSD), patient- and tumour-wise sensitivity, specificity, AUC | **τ and the detection rule are not published** | [UNVERIFIED] | | | | V-doc |
| **TotalSegmentator** | Dice, NSD (paper only) | [UNVERIFIED] | — | | | | V-doc |

### 3.2 Notes on the metric implementations (surface metrics)

- **DeepMind `surface-distance`** is used by FLARE, KiTS, BraTS-2023 and HECKTOR.
  - `compute_surface_distances(gt, pred, spacing)` needs the spacing in **array-axis order**. With SimpleITK arrays (z, y, x) pass `GetSpacing()[::-1]`; KiTS does exactly this. With nibabel arrays pass `header.get_zooms()`; FLARE does this.
  - An empty mask gives `inf` distances. NSD is `nan` when both masks are empty and `0` when only one is.
  - HD95 is surface-area weighted.
- **MONAI `DiceMetric`** defaults to `ignore_empty=True`: an empty GT returns NaN **even when the prediction contains false positives**, and `nanmean` then drops the case. This silently hides FPs.
- **MONAI `HausdorffDistanceMetric`**: one mask empty gives inf; both empty give NaN. Without `spacing=` the result is in voxels.
- **MONAI `SurfaceDiceMetric`**: both empty give NaN; one empty gives 0. With `use_subvoxels=False` it counts edge voxels, which differs from DeepMind's surfel areas.
- **Metrics Reloaded**: undefined values come back as NaN. The recommended aggregation is best-value when both masks are empty and worst-value when one is. The worst distance is the image diagonal in mm (`calculate_worse_dist`).
- **nnU-Net `evaluate_predictions`**: if tp+fp+fn == 0, Dice = NaN, and means use `np.nanmean`.

---

## 4. Cross-benchmark empty-mask and tolerance conventions

| Convention | Used by |
|---|---|
| Both empty → **1** (perfect) | FLARE21–24, KiTS21/23, ISLES22 (Dice, F1), BraTS 2021 (convention), BraTS 2023 (via NaN→1), ToothFairy2/3, VerSe |
| Both empty → **NaN, excluded** | nnU-Net, MONAI DiceMetric (default), DeepMind NSD, autoPET Dice (negatives use FPV only), TopCoW (class skipped) |
| Both empty → **0** | WORD (DSC 0, HD95 50), LiTS medpy `dc`, MSD "undefined → 0" [UNVERIFIED] |
| One empty → distance **penalty constant** | BraTS 2023: 374 mm; BraTS 2021: 373.13 mm [UNVERIFIED]; TopCoW: 90 mm; WORD: 50; ToothFairy: volume diagonal (voxels); Metrics Reloaded: image diagonal (mm) |
| One empty → distance **0** (lenient) | LiTS ASSD/MSD; HECKTOR 2021 HD95 when pred empty [per agent reading] |
| Aggregated (dataset-level) Dice, so no per-case empties | HECKTOR 2022/2025 GTVn |
| Per-structure NSD tolerance | MSD (per task), FLARE22+ (per organ), KiTS (per HEC), SAROS (3 mm) |
| Single NSD tolerance | FLARE21 (1 mm), FLARE23/24 lesion (2 mm) |
| Connected components | BraTS 2023 and ISLES: 26-connectivity; autoPET: **18**; TopCoW Betti-0: 26 |
| Distances in voxels, not mm | ToothFairy2/3 HD95 (medpy with no spacing) |

**Implication for SegEvalKit.** The empty-mask policy has to be a first-class, per-benchmark configurable parameter. It has two parts:

- `both_empty ∈ {1, nan, 0}`
- `one_empty_distance ∈ {inf, constant, diagonal_mm, diagonal_vox, 0}`

It should be combined with an NSD tolerance map, connectivity, and "NSD := 0 if DSC < x" gating (FLARE). Ship presets that reproduce each official script exactly, including the known quirks: FLARE's z-slab cropping, BraTS FP counting, and WORD's both-empty = 0.

---

## 5. Baseline model families and their output conventions

| Model | Repo | Output file | Label ids | Geometry | Probabilities | Notes | Tag |
|---|---|---|---|---|---|---|---|
| **nnU-Net v2** | MIC-DKFZ/nnUNet | `{CASE}.nii.gz` (input `{CASE}_{XXXX}.nii.gz`), ML | From `dataset.json` `"labels": {"background": 0, "liver": 1, …}` (name → int; must be consecutive). Region-based: `"whole_tumor": [1, 2, 3]` + `regions_class_order`; output is still an integer map, so rebuild the regions | Restored to the original spacing, origin and direction | `--save_probabilities` → `{CASE}.npz` (key `probabilities`, shape (C, z, y, x) for SimpleITK) + `.pkl` | `ignore` label = highest int, never predicted | V-code |
| **nnU-Net v1** | branch nnunetv1 | same | `dataset.json` int → name, plus `modality` | Restored | `--save_npz` → key `softmax` (float16) | — | V-code |
| **nnU-Net ResEnc M/L/XL** | same (`-p nnUNetResEncUNet{M,L,XL}Plans`) | as v2 | as v2 | as v2 | as v2 | "nnU-Net Revisited" (arXiv 2404.09556) | V-doc |
| **MedNeXt** | MIC-DKFZ/MedNeXt | nnU-Net v1 conventions (custom trainers) | dataset.json | Restored | npz (v1) | — | V-doc |
| **STU-Net** | uni-medical/STU-Net | nnU-Net style | Its own `label_orders.json`, which **differs from TotalSegmentator ids** | Restored | — | — | V-doc |
| **TotalSegmentator** | wasserth/TotalSegmentator | PS `segmentations/<class>.nii.gz`, or ML with `--ml` (names in the NIfTI extension header) | `map_to_binary.class_map[task]` | Restored (except `--save_lowres`) | `--save_probabilities` (some tasks) | `--roi_subset` produces a subset of files | V-code |
| **MONAI SwinUNETR / UNETR / SegResNet** (bundles) | Project-MONAI/model-zoo | `out/<name>/<name>_trans.nii.gz` (the `SaveImage` default postfix is `trans`; only the BraTS bundle uses `seg`); dtype float32 by default | Bundle `metadata.json` `channel_def` | `Invertd` → original geometry | Optional | BraTS bundle: sigmoid TC/WT/ET → labels {1, 2, 4} | V-code |
| **VISTA3D** | Project-MONAI/VISTA | ML `_trans.nii.gz` | Global ids 0–132 (`channel_def`); "everything" mode skips some ids. **Label 255 = not processed** | Original | — | Point prompts (x, y, z); class prompts | V-code |
| **SuPreM** | MrGiovanni/SuPreM | PS `<case>/segmentations/<organ>.nii.gz` (uint8) + `combined_labels.nii.gz` | AbdomenAtlas 1.1 classes [UNVERIFIED id order] | Original affine | — | Input `<case>/ct.nii.gz` | V-code |
| **CLIP-Driven Universal Model** | ljwztc/CLIP-Driven-Universal-Model | PS `<name>_<Organ>.nii.gz` (float32) | 32-class template | Original | — | — | V-code |
| **MedSAM (v1)** | bowang-lab/MedSAM | 2D PNG per slice | Binary per box prompt | Slice-wise; 1024 internal | — | Needs stacking to 3D | V-code |
| **MedSAM2** | bowang-lab/MedSAM2 | `<img>_k<slice>_mask.nii.gz` (CopyInformation) or npz `segs` | Instance ids | Original | — | Prompted | V-code |
| **SAM-Med3D** | uni-medical/SAM-Med3D | ML uint8 with **GT label ids** | Prompts sampled from GT | Resampled back | — | **Only classes present in GT are predicted**, so false positives on empty classes are impossible. Beware when comparing against automatic models | V-code |
| **MedFormer** | yhygao/CBIM-Medical-Image-Segmentation | No documented NIfTI writer | Dataset preprocessing | [UNVERIFIED] | — | PanTS baseline checkpoint on HF | V-doc |
| **UniverSeg** | JJGO/UniverSeg | Tensor logits (B, 1, 128, 128) | Binary, in-context | 2D, 128² resize | — | No writer | V-code |

---

## 6. Recommendations for SegEvalKit

1. **Ingestion adapters.** Support three shapes of input:
   - (a) an ML NIfTI plus an id→name map (MSD/nnU-Net `dataset.json`, v1 and v2 schemas);
   - (b) a PS folder `segmentations/<name>.nii.gz` (TotalSegmentator, AbdomenAtlas, PanTS, SuPreM);
   - (c) per-instance or per-annotator files (KiTS21/23).

   Also read `.mha`, `.nrrd` and `.mhd` (PI-CAI, SPIDER, ToothFairy, LNQ, LA2018, LUNA16) through SimpleITK.
2. **Name-keyed matching.** Map everything to structure names. Ids collide across datasets: ACDC and M&Ms swap LV/RV; TotalSegmentator CT and MR ids differ; BraTS ET is 4 in 2021 and 3 from 2023; MSD Task01 uses 1 = edema.
3. **Overlapping structures.** Never collapse PS masks into ML for evaluation (PanTS pancreas ⊃ head/body/tail; veins overlapping organs). Evaluate each binary mask independently.
4. **Binarisation.** Use `> 0.5`, not `== 1` (PanTS pancreas sub-parts load as 1.0000000591). Cast float label maps (MSD Colon, CT-ORG) with rounding.
5. **Geometry checks.** Assert shape and affine equality within a tolerance, or resample the prediction onto the GT grid with nearest neighbour (as HECKTOR does). Take spacing in array-axis order.
6. **Ignore labels.** Honour 255 (SAROS unannotated slices, VISTA3D "not processed") and nnU-Net `ignore`.
7. **File hygiene.** Skip `._*` AppleDouble files (they are in the MSD tars).
8. **Empty-mask presets.** See Section 4. Report the number of empty-GT cases per structure alongside every mean.
9. **Region metrics.** Build BraTS WT/TC/ET, KiTS HECs and MSD "liver = 1 ∪ 2" regions from integer maps before scoring.

---

## 7. References

### Datasets and official repositories
- PanTS: https://github.com/MrGiovanni/PanTS (class map in `data/README.md`); HF https://huggingface.co/datasets/BodyMaps/PanTSMini; paper arXiv 2507.01291, https://www.cs.jhu.edu/~zongwei/publication/li2025pants.pdf
- AbdomenAtlas: https://github.com/MrGiovanni/AbdomenAtlas ; https://huggingface.co/datasets/BodyMaps/AbdomenAtlas1.0Mini ; RadGPT / AbdomenAtlas 3.0: https://arxiv.org/abs/2501.04678
- Merlin: https://github.com/StanfordMIMI/Merlin (`documentation/download.md`); https://stanfordaimi.azurewebsites.net/datasets/60b9c7ff-877b-48ce-96c3-0194c8205c40
- TotalSegmentator: https://github.com/wasserth/TotalSegmentator ; class maps https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/map_to_binary.py ; CT https://zenodo.org/records/6802613 (v3), https://zenodo.org/records/10047292 (v2.0.1); MRI https://zenodo.org/records/11367004 (v3), https://zenodo.org/records/14710732 (v2)
- MSD: http://medicaldecathlon.com ; tars https://msd-for-monai.s3-us-west-2.amazonaws.com/TaskXX_Name.tar ; Antonelli et al., Nat Commun 2022, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9287542/
- AMOS: https://zenodo.org/records/7262581 ; https://amos22.grand-challenge.org
- BTCV: https://www.synapse.org/#!Synapse:syn3193805
- WORD: https://github.com/HiLab-git/WORD
- AbdomenCT-1K: https://github.com/JunMa11/AbdomenCT-1K
- FLARE (all years, eval code): https://github.com/JunMa11/FLARE
- CHAOS: https://zenodo.org/records/3431873 ; https://github.com/emrekavur/CHAOS-evaluation
- CT-ORG: https://www.cancerimagingarchive.net/collection/ct-org/
- SAROS: https://github.com/UMEssen/saros-dataset ; Koitka et al., Sci Data 2024 (PMC11087485)
- KiTS19/21/23: https://github.com/neheller/kits19 , https://github.com/neheller/kits21 , https://github.com/neheller/kits23
- LiTS: https://github.com/PatrickChrist/LITS-CHALLENGE ; Bilic et al., MedIA 2023
- ULS23: https://uls23.grand-challenge.org ; https://github.com/DIAGNijmegen/ULS23 ; arXiv 2406.05231
- KiPA22: https://kipa22.grand-challenge.org
- BraTS 2021: arXiv 2107.02314. BraTS 2023 lesion-wise metrics: https://github.com/rachitsaluja/BraTS-2023-Metrics. BraTS 2024: arXiv 2405.18368, 2409.08143; Synapse syn53708249. BraTS 2025: https://zenodo.org/records/15094823
- ISLES 2022: https://github.com/ezequieldlrosa/isles22 ; arXiv 2206.06694
- autoPET: https://github.com/lab-midas/autoPET (`val_script.py`); https://autopet.grand-challenge.org ; https://autopet-iii.grand-challenge.org/evaluation-and-ranking/ ; https://autopet-iv.grand-challenge.org/eval/ ; https://github.com/lab-midas/autoPETCTIV
- HECKTOR: https://github.com/voreille/hecktor ; https://hecktor25.grand-challenge.org/tasks-and-evaluation/
- TopCoW: https://github.com/CoWBenchmark/TopCoW_Eval_Metrics ; https://topcow23.grand-challenge.org ; Zenodo 15692630
- ACDC: https://www.creatis.insa-lyon.fr/Challenge/acdc/ ; M&Ms: doi 10.1109/TMI.2021.3090082 ; LA 2018: arXiv 2004.12314 ; MM-WHS: arXiv 1902.07880 ; SegTHOR: arXiv 1912.05950
- Parse2022: https://parse2022.grand-challenge.org ; LNQ2023: https://lnq2023.grand-challenge.org , arXiv 2408.10069
- ToothFairy 1–3: https://github.com/AImageLab-zip/ToothFairy
- SPIDER: https://zenodo.org/records/10159290 ; Sci Data (PMC10908819)
- VerSe: https://github.com/anjany/verse ; arXiv 2001.09193
- PI-CAI: https://github.com/DIAGNijmegen/picai_eval ; https://github.com/DIAGNijmegen/picai_labels ; Zenodo 6624726
- Duke Breast Cancer MRI: https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/
- LIDC-IDRI: https://www.cancerimagingarchive.net/collection/lidc-idri/ ; LUNA16: https://luna16.grand-challenge.org , arXiv 1612.08012

### Methods and metric libraries
- nnU-Net: https://github.com/MIC-DKFZ/nnUNet (ResEnc: `documentation/resenc_presets.md`, arXiv 2404.09556)
- MedNeXt: https://github.com/MIC-DKFZ/MedNeXt ; STU-Net: https://github.com/uni-medical/STU-Net
- MONAI: https://github.com/Project-MONAI/MONAI ; model zoo https://github.com/Project-MONAI/model-zoo ; VISTA3D https://github.com/Project-MONAI/VISTA
- SuPreM: https://github.com/MrGiovanni/SuPreM ; CLIP-Driven Universal Model: https://github.com/ljwztc/CLIP-Driven-Universal-Model
- MedSAM: https://github.com/bowang-lab/MedSAM ; MedSAM2: https://github.com/bowang-lab/MedSAM2 ; SAM-Med3D: https://github.com/uni-medical/SAM-Med3D
- MedFormer: https://github.com/yhygao/CBIM-Medical-Image-Segmentation ; UniverSeg: https://github.com/JJGO/UniverSeg
- DeepMind surface-distance: https://github.com/google-deepmind/surface-distance ; Metrics Reloaded: https://github.com/Project-MONAI/MetricsReloaded (Maier-Hein et al., Nat Methods 2024)
