"""Compare generated images with the original low-light targets.

Examples:
    python evaluate.py \
        --generated_dir ./eval_results/M1_generated \
        --gt_dir ./hpc_ready_dataset/test/images \
        --output_csv ./eval_results/M1_metrics.csv

    python evaluate.py \
        --generated_dir ./eval_results/M2_generated \
        --gt_dir ./ablation_dataset/test/images \
        --output_csv ./eval_results/M2_metrics.csv

Requires lpips and scikit-image.
"""

import os
import argparse
import csv
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim

try:
    import lpips
except ImportError:
    print("lpips is required; install it with: pip install lpips")
    exit(1)


def load_image_as_tensor(path, size=512):
    """Load an RGB image as an LPIPS tensor in the range [-1, 1]."""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),           # [0, 1]
        transforms.Normalize([0.5], [0.5])  # [-1, 1]
    ])
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)  # (1, 3, H, W)


def load_image_as_numpy(path, size=512):
    """Load an RGB image as a NumPy array for SSIM."""
    img = Image.open(path).convert("RGB").resize((size, size))
    return np.array(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_dir", type=str, required=True,
                        help="Directory containing generated images")
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="Directory containing original low-light targets (L0)")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Output path for per-image metrics")
    parser.add_argument("--size", type=int, default=512,
                        help="Square image size used for comparison")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    print("Loading the LPIPS model...")
    lpips_fn = lpips.LPIPS(net='alex').cuda()

    gen_files = sorted([f for f in os.listdir(args.generated_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Found {len(gen_files)} generated images")

    lpips_scores = []
    ssim_scores = []
    per_image_results = []
    matched = 0
    skipped = 0

    for i, gen_fname in enumerate(gen_files):
        name_no_ext = os.path.splitext(gen_fname)[0]

        gt_path = None
        for ext in [".jpg", ".png", ".jpeg"]:
            candidate = os.path.join(args.gt_dir, name_no_ext + ext)
            if os.path.exists(candidate):
                gt_path = candidate
                break

        if gt_path is None:
            skipped += 1
            continue

        gen_path = os.path.join(args.generated_dir, gen_fname)

        gen_tensor = load_image_as_tensor(gen_path, args.size).cuda()
        gt_tensor = load_image_as_tensor(gt_path, args.size).cuda()
        with torch.no_grad():
            lpips_val = lpips_fn(gen_tensor, gt_tensor).item()

        gen_np = load_image_as_numpy(gen_path, args.size)
        gt_np = load_image_as_numpy(gt_path, args.size)
        ssim_val = ssim(gen_np, gt_np, channel_axis=2, data_range=255)

        lpips_scores.append(lpips_val)
        ssim_scores.append(ssim_val)
        per_image_results.append({
            "filename": name_no_ext,
            "lpips": lpips_val,
            "ssim": ssim_val
        })
        matched += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(gen_files):
            print(f"Progress: {i + 1}/{len(gen_files)}")

    mean_lpips = np.mean(lpips_scores) if lpips_scores else 0
    mean_ssim = np.mean(ssim_scores) if ssim_scores else 0
    std_lpips = np.std(lpips_scores) if lpips_scores else 0
    std_ssim = np.std(ssim_scores) if ssim_scores else 0

    print(f"\nEvaluation complete: matched {matched}, skipped {skipped}")
    print(f"LPIPS (lower is better): {mean_lpips:.4f} ± {std_lpips:.4f}")
    print(f"SSIM (higher is better): {mean_ssim:.4f} ± {std_ssim:.4f}")

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "lpips", "ssim"])
        writer.writeheader()
        writer.writerows(per_image_results)
        writer.writerow({"filename": "MEAN", "lpips": mean_lpips, "ssim": mean_ssim})
        writer.writerow({"filename": "STD", "lpips": std_lpips, "ssim": std_ssim})

    print(f"Per-image results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
