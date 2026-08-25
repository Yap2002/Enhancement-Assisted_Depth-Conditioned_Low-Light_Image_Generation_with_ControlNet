# Enhancement-Assisted Depth-Conditioned Low-Light Image Generation with ControlNet

This repository contains the implementation and evaluation artefacts for an MSc dissertation on enhancement-assisted pseudo-supervision for low-light image generation.

The study compares two matched ControlNet routes:

- **Baseline (M2):** pseudo-depth maps and captions are extracted from the original ExDark images.
- **Enhancement-assisted (M1):** Zero-DCE is applied before pseudo-depth and caption extraction.

The original low-light image remains the training target in both routes. The comparison therefore changes the source of the pseudo-conditions, rather than the target image domain.

## Evidence boundary

ExDark does not provide measured depth maps or verified image captions. The project evaluates generated-image fidelity, distributional realism, and adherence to route-specific pseudo-conditions. It does not claim direct depth or caption accuracy.

The reported comparison uses one completed training run per route. The final training scripts do not record a training seed, so training-run variance cannot be estimated. Inference uses seed 42 and resets the generator for each sample.

## Repository contents

```text
src/                    ControlNet training, checkpoint testing, and batch generation
evaluation/             LPIPS, SSIM, edge, FID, NIQE, CLIP, and scene analysis
scripts/                Slurm scripts retained from the University of Leeds HPC runs
splits/                 Exact train, validation, test, and complete filename lists
results/                Per-image metric files and aggregate summaries
figures/                Training-loss plots
dissertation/figures/   The 16 rendered figures used in the final dissertation
DATA_LICENSE.md         ExDark provenance and redistribution guidance
requirements.txt        Direct Python dependencies; not an exact package lock
```

The original Kaggle preprocessing notebooks are retained on the [`prototype/demo-v1`](https://github.com/Yap2002/Enhancement-Assisted_Depth-Conditioned_Low-Light_Image_Generation_with_ControlNet/tree/prototype/demo-v1) branch:

1. Zero-DCE image enhancement;
2. Depth Anything depth estimation;
3. BLIP-2 caption generation;
4. the original ControlNet training prototype.

## Dataset and split

The project uses the public ExDark dataset. Obtain the source images from the [official repository](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset) or the [Universiti Malaya Research Data Repository](https://doi.org/10.22452/RD/JUSQEK).

The source collection contains 7,363 image records. After triplet validation, 7,115 usable samples were divided into:

| Split | Images |
| --- | ---: |
| Training | 5,692 |
| Validation | 711 |
| Test | 712 |

Use the committed files in `splits/` to reconstruct the reported partition. Do not create a new random split if the aim is to reproduce the dissertation comparison.

The expected prepared datasets are:

```text
ablation_dataset/                 # M2 baseline
  train|validation|test/
    images/
    depths/
    captions.json

hpc_ready_dataset/                # M1 enhancement-assisted
  train|validation|test/
    images/
    depths/
    captions.json
```

The original ExDark images, derived datasets, captions, and depth maps are not stored in ordinary Git history. See `DATA_LICENSE.md` before redistributing any source or derived data.

The two prepared image-and-depth datasets are available in the [ExDark Derived Datasets v1.0 GitHub Release](https://github.com/Yap2002/Enhancement-Assisted_Depth-Conditioned_Low-Light_Image_Generation_with_ControlNet/releases/tag/dataset-v1.0):

- `exdark_m1_enhanced_images_and_depth_v1.0.tar.gz` contains the M1 enhancement-assisted images and estimated depth maps;
- `exdark_m2_baseline_images_and_depth_v1.0.tar.gz` contains the M2 baseline images and estimated depth maps;
- `SHA256SUMS.txt` provides integrity checks for both archives.

Captions are not included in these two archives. Use the committed split files and the documented BLIP-2 settings to reconstruct them where required.

## Models and recorded settings

| Component | Recorded model or setting |
| --- | --- |
| Stable Diffusion backbone | `runwayml/stable-diffusion-v1-5` |
| Initial ControlNet | `lllyasviel/sd-controlnet-depth` |
| Zero-DCE checkpoint | `Epoch99.pth` |
| Depth estimator | `LiheYoung/depth-anything-small-hf` |
| Caption model | `Salesforce/blip2-opt-2.7b` |
| Caption prompt | `A picture of` |
| Caption decoding | `max_new_tokens=30` |
| Resolution | 512 x 512 |
| Learning rate | `1e-5` |
| Per-device batch size | 1 |
| Gradient accumulation | 4 |
| Effective batch size | 4 |
| Training steps | 10,000 |
| Checkpoint interval | 2,000 steps |
| Precision | FP16 |
| Memory option | Gradient checkpointing |
| Inference seed | 42, reset for each sample |
| Inference steps | 30 |
| Guidance scale | 5.0 |
| ControlNet conditioning scale | 1.0 |

Exact model commit hashes and the complete package environment were not retained. Model identifiers are therefore provided at repository level rather than as fully pinned revisions.

## Environment

The experiments used Python 3.10, PyTorch, Diffusers, Accelerate, CUDA 12.4.1, and one GPU on a Slurm-managed cluster.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` lists the direct dependencies used by the repository. It is not a lock file and does not reproduce the exact historical package versions. For a new reproduction, create and retain a lock file after testing the environment.

## Training

The training implementation is `src/train_controlnet.py`. The reported route configurations are preserved in:

- `scripts/submit.sh` for M1;
- `scripts/submit_ablation.sh` for M2.

The Slurm scripts are historical run records. They contain the original University of Leeds account paths and assume that the Python scripts and prepared datasets are available from the submission working directory. Before reuse, update:

- the environment activation command;
- project and dataset paths;
- checkpoint output paths;
- GPU partition and module names;
- Python script paths.

For example, a portable invocation should call `src/train_controlnet.py` and use absolute or project-root-relative dataset paths. Do not expect the retained Slurm files to run unchanged on another machine.

## Generation and evaluation

`src/batch_generate.py` generates images for the two routes. Its recorded defaults are seed 42, 30 inference steps, guidance scale 5.0, and ControlNet conditioning scale 1.0.

The evaluation directory contains scripts for:

- LPIPS and SSIM;
- edge consistency;
- FID;
- NIQE;
- CLIP similarity;
- shuffled-pair lower-bound checks;
- lighting and environment group analysis.

Several evaluation scripts retain `~/nobackup` or `/users/fkwt0359/` paths from the final HPC runs. Update their path constants before executing them elsewhere.

The planned test split contains 712 images. The principal generated-image comparisons contain 672 matched outputs because 40 caption keys used uppercase `.JPEG`, while the generation lookup accepted only `.jpg`, `.jpeg`, and `.png`. Edge and NIQE have additional metric-specific skipped samples. The relevant CSV and skipped-file records are included in `results/`.

### Result-version note

Some committed per-image CSV files are retained evaluation snapshots rather than the final audited table used in the dissertation. In particular, `metrics_baseline.csv` and `metrics_enhanced.csv` contain 674 rows, while the final dissertation reports 672 matched outputs. The retained edge CSV files likewise contain two more rows than the final reported coverage. The exact two-row reconciliation record was not retained, so these files have not been edited retrospectively. The table below reproduces the final dissertation values; use the CSV files to inspect the retained measurements, not to infer that their row counts are identical to the final audit.

## Main reported results

| Metric | Baseline M2 | Enhanced M1 | Preferred direction |
| --- | ---: | ---: | --- |
| LPIPS | 0.7168 | **0.7061** | Lower |
| SSIM | **0.1796** | 0.1767 | Higher |
| Edge consistency | 0.1875 | **0.2051** | Higher |
| FID | 83.0217 | **76.1046** | Lower |
| NIQE | 5.5188 | **4.8475** | Lower |
| CLIP score | 29.1822 | **29.8411** | Higher |

These are single-run results with metric-specific sample coverage. They support better generation performance and stronger condition adherence under the reported configuration, but they do not establish condition accuracy or repeated-run statistical significance.

## Files not included

The following files are intentionally absent because of size, licensing, availability, or incomplete historical retention:

- original ExDark images;
- prepared M1 and M2 dataset directories in ordinary Git history; image-and-depth archives are available in the `dataset-v1.0` Release;
- trained ControlNet checkpoints;
- generated test-image directories;
- complete raw Slurm logs;
- an exact historical package lock;
- an explicit recorded training seed;
- exact cached model commit revisions.

The repository is sufficient to inspect the implementation, experimental settings, splits, evaluation outputs, and dissertation figures. It is not a self-contained one-command reproduction archive without the external data and model artefacts listed above.

## Dataset citation

If you use ExDark, cite:

> Y. P. Loh and C. S. Chan, "Getting to Know Low-light Images with The Exclusively Dark Dataset," *Computer Vision and Image Understanding*, 2019. https://doi.org/10.1016/j.cviu.2018.10.010
