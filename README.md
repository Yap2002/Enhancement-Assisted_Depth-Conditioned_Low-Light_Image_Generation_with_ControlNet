# Enhancement-Assisted Depth-Conditioned Low-Light Image Generation with ControlNet

This repository contains the final implementation and evaluation artefacts for an MSc dissertation investigating whether low-light enhancement before depth-conditioned ControlNet training improves image generation quality.

The original notebook prototype is preserved on the `prototype/demo-v1` branch. The final implementation is maintained separately so that the experimental pipeline and results remain easy to inspect.

## Experimental design

Two matched ControlNet configurations are compared:

- **Enhanced (M1):** Zero-DCE-enhanced images with corresponding depth maps.
- **Baseline (M2):** original low-light images with corresponding depth maps.

Both configurations use the same training schedule and evaluation split. The committed split contains 5,692 training images, 711 validation images, and 712 test images.

## Repository structure

```text
src/          Training, inference, and batch generation
evaluation/   Full-reference, no-reference, semantic, edge, and FID evaluation
scripts/      Slurm job scripts used on the University of Leeds HPC service
splits/       Exact train, validation, test, and complete filename lists
results/      Per-image metrics and aggregate evaluation summaries
figures/      Training-loss figures used during analysis
DATA_LICENSE.md  Data provenance, redistribution, and citation information
```

## Environment

The experiments used Python 3.10, PyTorch, Diffusers, Accelerate, and CUDA on a Slurm-managed GPU cluster. Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

The shell scripts preserve the HPC configuration used for the reported experiments. Users on another system must update the virtual-environment, dataset, model-output, and project-root paths before running them.

## Training

The main implementation is `src/train_controlnet.py`. The two training configurations are represented by `scripts/submit.sh` and `scripts/submit_ablation.sh`.

The reported run used 10,000 optimisation steps, gradient accumulation, mixed-precision training, gradient checkpointing, and checkpoints every 2,000 steps. Consult the Slurm scripts for the exact command-line arguments used on the cluster.

## Generation and evaluation

`src/batch_generate.py` generates the matched M1 and M2 test outputs. Evaluation scripts report LPIPS, SSIM, CLIP similarity, NIQE, FID, edge consistency, random-pair lower bounds, and scene-level comparisons.

The principal human-readable result is `results/summary_baseline_enhanced.txt`. CSV files contain the corresponding per-image measurements.

## Data availability

The experiments use the public ExDark dataset. Download the original images from the [official ExDark repository](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset) or the [Universiti Malaya Research Data Repository](https://researchdata.um.edu.my/dataset.xhtml?persistentId=doi:10.22452/RD/JUSQEK). The original images are not duplicated in this Git repository.

The exact experimental partitions are recorded in `splits/train.txt`, `splits/validation.txt`, and `splits/test.txt`. `splits/all_images.txt` is the ExDark metadata index: it contains one header row followed by 7,363 image records, rather than 7,364 images. The resulting experiment uses 5,692 training, 711 validation, and 712 test images.

The two prepared variants are:

- `ablation_dataset`: original low-light ExDark images and their estimated depth maps (M2/baseline).
- `hpc_ready_dataset`: Zero-DCE-enhanced images and their estimated depth maps (M1/enhanced).

Large derived-image archives and trained weights are kept outside ordinary Git history. A public archive should include a checksum manifest and preserve the filenames in `splits/`; its URL can be added here after upload. See `DATA_LICENSE.md` for provenance, attribution, and redistribution notes.

## Reproducibility notes

- M1 corresponds to the enhancement-assisted configuration.
- M2 corresponds to the original-input baseline.
- The `baseline` and `enhanced` result filenames are descriptive aliases retained for clarity.
- Skipped-image lists are included so that metric sample sizes can be audited.
- Large checkpoints, caches, generated-image directories, and raw Slurm logs are intentionally excluded.
- Reconstructing the reported partitions requires using the committed manifests rather than creating a new random split.

## Dataset citation

If you use ExDark, cite the dataset paper:

> Y. P. Loh and C. S. Chan, "Getting to Know Low-light Images with The Exclusively Dark Dataset," *Computer Vision and Image Understanding*, 2019. https://doi.org/10.1016/j.cviu.2019.04.010
