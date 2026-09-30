# What you need

SegEvalKit compares **predicted masks** with **reference masks**. It does not run segmentation models; you bring
their outputs. Everything below fits on a laptop. **No data yet?** Open the [live viewer demo](../assets/viewer/demo_phantom.html){ target="_blank" }
or run `segevalkit view --demo` (a synthetic phantom with three synthetic models, see [Interactive viewer](../analysis/viewer.md)).

<div class="grid cards" markdown>

-   :material-file-image-outline:{ .lg } **1. Scans** *(optional)*

    CT or MR volumes as NIfTI (`.nii.gz`), or MHA / NRRD with the `sitk` extra. Only needed for pictures
    (overlays, galleries); the metrics use the masks alone.

-   :material-draw:{ .lg } **2. Reference masks**

    The ground truth, one file per case: a **label map** (0 background, 1 liver, 2 tumour, …) **or** one
    binary file per structure (`case/segmentations/liver.nii.gz`).

-   :material-robot-outline:{ .lg } **3. Predictions**

    Your model's masks, in either layout and with its own label ids; you tell SegEvalKit which id is which
    structure.

-   :material-laptop:{ .lg } **4. A computer with Python**

    Python 3.10–3.12, Linux, macOS or Windows. A GPU is optional.

</div>

## Rules the files must follow

| Rule | Why | Checked by |
|---|---|---|
| Prediction and reference of a case lie on the **same voxel grid** (shape, spacing, orientation) | Metrics compare voxel by voxel | `alignment="strict"` (default) stops on a mismatch; `"resample"` fixes it |
| The case id is the file name without extension, or the folder name | Pairs prediction with reference | Missing predictions are scored as empty, never skipped |
| Spacing in the header is in millimetres | Distances (HD95, NSD) are in mm | `segevalkit audit --ref labels/ --images images/` flags headers that disagree with their scan |
| Masks are integer or boolean | Label ids select structures | Probabilities, if any, go in a separate `prob=` folder (calibration metrics) |

Layouts and examples: [Data & output format](data-format.md).

## Install

```console
$ python -m pip install "segevalkit[all] @ git+https://github.com/aj-das-research/SegEvalKit.git@v0.1.0"
```

In a fresh Python 3.10–3.12 environment; see [Installation](installation.md) for conda, venv and troubleshooting.

## Hardware

| Task | Needs | Notes |
|---|---|---|
| **Evaluating** with SegEvalKit | CPU and RAM: about two copies of the largest mask per worker | A GPU only speeds up surface distances (`device="cuda"`); results are identical |
| GPU acceleration | NVIDIA GPU with CUDA PyTorch (`device="cuda"`), or Apple silicon (`device="mps"`) | No AMD / ROCm testing; use the CPU there |
| **Running a model** to get predictions | Model-specific, separate from SegEvalKit | See the table below |

Published inference requirements of the models used in these docs:

| Model | GPU memory for inference | Source |
|---|---|---|
| TotalSegmentator v2 | 6.1–11.4 GB (RAM 7.6–11.8 GB) for 512×512×280 to ×824 CTs on an RTX 3090; `--fast` 5.2–7.5 GB | TotalSegmentator README, *Resource requirements* |
| nnU-Net v2 | at least 4 GB of free VRAM; CPU and Apple MPS also work but are slower | nnU-Net *installation instructions* |
| MedFormer (R-Super) | not published for inference (the authors report > 30 GB for training) | R-Super *Merlin demo* |

We ran all three on NVIDIA A100 40 GB GPUs. Large CTs also need host RAM: nnU-Net's export workers ran out of
58 GB with three workers per GPU, so give 32–64 GB of RAM or fewer workers.

!!! note "Mac and non-NVIDIA GPUs"
    SegEvalKit's metrics run on any CPU, including Apple silicon and machines with AMD GPUs. On Apple silicon,
    `device="mps"` uses the GPU for surface distances. To *run* models, TotalSegmentator supports
    `--device mps`, and nnU-Net notes that MPS lacks 3D convolutions, so it may fall back to the CPU.

## Checklist

1. Put reference masks in one folder, one file (or one sub-folder) per case.
2. Put predictions in another folder with the same case ids.
3. Write down which label id is which structure, for both sides.
4. Install SegEvalKit and run `segevalkit audit --ref labels/ --images images/` if you have scans.
5. Evaluate: `segevalkit evaluate --pred preds/ --ref labels/ --labels liver=1,tumour=2 --out eval/`, then
   follow the [Quickstart](quickstart.md).
