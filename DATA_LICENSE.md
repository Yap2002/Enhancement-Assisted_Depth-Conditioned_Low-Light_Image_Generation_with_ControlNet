# Data licence, provenance, and redistribution

## Source dataset

This project uses the Exclusively Dark (ExDark) dataset created by Yuen Peng Loh and Chee Seng Chan. ExDark contains 7,363 low-light images across 12 object classes and ten illumination conditions.

Authoritative sources:

- Official project repository: https://github.com/cs-chan/Exclusively-Dark-Image-Dataset
- Universiti Malaya Research Data Repository: https://doi.org/10.22452/RD/JUSQEK
- Dataset paper: https://doi.org/10.1016/j.cviu.2018.10.010

The official project repository provides a BSD 3-Clause licence file. Anyone redistributing data should retain the upstream copyright and licence notice and should review the authoritative source terms at the time of redistribution. The original ExDark images are not committed to this repository; users should obtain them from an official source.

## Project-generated data

The reported experiments use two aligned derivatives of ExDark:

| Project name | Image input | Conditioning data | Experiment label |
| --- | --- | --- | --- |
| `ablation_dataset` | Original low-light image | Estimated depth map | M2 / baseline |
| `hpc_ready_dataset` | Zero-DCE-enhanced image | Estimated depth map | M1 / enhanced |

The enhanced images and depth maps are transformations generated for this dissertation. Their filenames preserve alignment with the original ExDark records. Publishing these derivatives does not replace the requirement to acknowledge ExDark or comply with its source terms.

## Published derived datasets

The prepared image-and-depth archives are published in the [ExDark Derived Datasets v1.0 GitHub Release](https://github.com/Yap2002/Enhancement-Assisted_Depth-Conditioned_Low-Light_Image_Generation_with_ControlNet/releases/tag/dataset-v1.0):

- `exdark_m1_enhanced_images_and_depth_v1.0.tar.gz`;
- `exdark_m2_baseline_images_and_depth_v1.0.tar.gz`;
- `SHA256SUMS.txt`.

These files are Release assets and are not stored in the repository's ordinary Git history. Verify a downloaded archive against `SHA256SUMS.txt` before use. Captions are not included in these two archives.

## Files included in this repository

- `splits/all_images.txt`: upstream-style metadata index with one header row and 7,363 image records.
- `splits/train.txt`: 5,692 experiment filenames.
- `splits/validation.txt`: 711 experiment filenames.
- `splits/test.txt`: 712 experiment filenames.
- `results/`: numerical experiment outputs; no source images.
- `figures/`: training-loss plots; no source-image archive.
- `dissertation/figures/`: the rendered figures used in the dissertation. Some composite figures contain selected ExDark examples, but this directory is not a complete dataset archive.

## Recommended archive contents

Large derived datasets should be published in a versioned research-data service such as Zenodo or Hugging Face Datasets, not stored in ordinary Git history. Each published archive should contain:

1. the derived images and matching depth maps;
2. the four files from `splits/`;
3. a copy of this provenance notice and the applicable upstream licence;
4. a machine-readable SHA-256 checksum manifest;
5. the archive version, creation date, preprocessing description, and software revision.

Do not describe the ExDark photographs as project-owned data. Describe them as source data and label Zero-DCE images and estimated depth maps as project-generated derivatives.

## Suggested acknowledgement

> This work uses the ExDark dataset by Loh and Chan. Enhancement-assisted images and estimated depth maps were derived from ExDark for the experiments reported in this repository.

This document records project provenance and is not legal advice. The authoritative upstream licence and repository terms take precedence.
