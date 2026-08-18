"""
Compute FID between generated images (M1, M2) and real low-light images (ExDark test set).
FID lower = better (generated distribution closer to real distribution).

Usage:
    python compute_fid.py
"""

import os
import sys
import argparse
from pathlib import Path
import shutil
import tempfile

import torch
from pytorch_fid.fid_score import calculate_fid_given_paths


# ============== Configuration ==============
M1_DIR = "/users/fkwt0359/nobackup/eval_results/M1_generated"
M2_DIR = "/users/fkwt0359/nobackup/eval_results/M2_generated"
REAL_DIR = "/users/fkwt0359/nobackup/hpc_ready_dataset/test/images"

BATCH_SIZE = 32
DIMS = 2048               # Inception feature dim (2048 is standard)
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ===========================================


def count_images(folder):
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return sum(1 for p in Path(folder).iterdir() if p.suffix.lower() in exts)


def compute_fid(gen_dir, real_dir, label):
    print(f"\n[{label}]")
    print(f"  Generated : {gen_dir}  ({count_images(gen_dir)} images)")
    print(f"  Real      : {real_dir}  ({count_images(real_dir)} images)")
    print(f"  Device    : {DEVICE}")

    fid = calculate_fid_given_paths(
        [gen_dir, real_dir],
        batch_size=BATCH_SIZE,
        device=DEVICE,
        dims=DIMS,
        num_workers=NUM_WORKERS,
    )
    print(f"  FID = {fid:.4f}")
    return fid


def main():
    print("=" * 60)
    print("FID Evaluation: M1 vs M2 (lower is better)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, computation will be slow.")

    fid_m1 = compute_fid(M1_DIR, REAL_DIR, "M1 (enhancement-assisted)")
    fid_m2 = compute_fid(M2_DIR, REAL_DIR, "M2 (raw low-light baseline)")

    diff = fid_m2 - fid_m1
    better = "M1" if fid_m1 < fid_m2 else "M2"

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"FID (M1): {fid_m1:.4f}")
    print(f"FID (M2): {fid_m2:.4f}")
    print(f"Diff (M2 - M1): {diff:+.4f}")
    print(f"Better model: {better}")
    print("=" * 60)

    # save to file
    with open("fid_results.txt", "w") as f:
        f.write("FID Evaluation Results (lower is better)\n")
        f.write("=" * 60 + "\n")
        f.write(f"M1 (enhancement-assisted): {fid_m1:.4f}\n")
        f.write(f"M2 (raw low-light):        {fid_m2:.4f}\n")
        f.write(f"Diff (M2 - M1):            {diff:+.4f}\n")
        f.write(f"Better model:              {better}\n")
    print("Saved to fid_results.txt")


if __name__ == "__main__":
    main()