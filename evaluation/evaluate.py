"""
定量评估脚本 - LPIPS & SSIM
===============================
比较生成图像与原始低光图 (L0) 的相似度。

用法:
    python evaluate.py \
        --generated_dir ./eval_results/M1_generated \
        --gt_dir ./hpc_ready_dataset/test/images \
        --output_csv ./eval_results/M1_metrics.csv

    python evaluate.py \
        --generated_dir ./eval_results/M2_generated \
        --gt_dir ./ablation_dataset/test/images \
        --output_csv ./eval_results/M2_metrics.csv

需要安装: pip install lpips scikit-image
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
    print("请先安装 lpips: pip install lpips")
    exit(1)


def load_image_as_tensor(path, size=512):
    """加载图片并转为 [-1, 1] 范围的 tensor (LPIPS 需要)"""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),           # [0, 1]
        transforms.Normalize([0.5], [0.5])  # [-1, 1]
    ])
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)  # (1, 3, H, W)


def load_image_as_numpy(path, size=512):
    """加载图片为 numpy array (SSIM 需要)"""
    img = Image.open(path).convert("RGB").resize((size, size))
    return np.array(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_dir", type=str, required=True,
                        help="生成图像目录")
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="Ground truth (L0 原图) 目录")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="评估结果 CSV 输出路径")
    parser.add_argument("--size", type=int, default=512,
                        help="统一 resize 到这个尺寸再比较")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    print("加载 LPIPS 模型...")
    lpips_fn = lpips.LPIPS(net='alex').cuda()

    gen_files = sorted([f for f in os.listdir(args.generated_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"共 {len(gen_files)} 张生成图待评估")

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
            print(f"进度: {i + 1}/{len(gen_files)}")

    mean_lpips = np.mean(lpips_scores) if lpips_scores else 0
    mean_ssim = np.mean(ssim_scores) if ssim_scores else 0
    std_lpips = np.std(lpips_scores) if lpips_scores else 0
    std_ssim = np.std(ssim_scores) if ssim_scores else 0

    print(f"\n{'='*50}")
    print(f"评估完成: 匹配 {matched} 张, 跳过 {skipped} 张")
    print(f"{'='*50}")
    print(f"LPIPS (越低越好): {mean_lpips:.4f} ± {std_lpips:.4f}")
    print(f"SSIM  (越高越好): {mean_ssim:.4f} ± {std_ssim:.4f}")
    print(f"{'='*50}")

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "lpips", "ssim"])
        writer.writeheader()
        writer.writerows(per_image_results)
        writer.writerow({"filename": "MEAN", "lpips": mean_lpips, "ssim": mean_ssim})
        writer.writerow({"filename": "STD", "lpips": std_lpips, "ssim": std_ssim})

    print(f"逐张结果已保存到: {args.output_csv}")


if __name__ == "__main__":
    main()