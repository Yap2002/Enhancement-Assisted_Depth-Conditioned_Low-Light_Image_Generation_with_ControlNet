#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
随机配对下限 (shuffled lower bound) —— LPIPS / SSIM / Edge-Consistency
================================================================
目的: 给逐图配对指标造 anchor。逻辑不同于 FID:
  这些指标衡量"生成图 X vs 它对应的目标 X"。random-pairing 把生成图和
  【打乱后不对应】的目标去算, 得到"两张无关低光图之间"的天然基线。
  若 matched 明显优于 shuffled, 证明生成图是在重建【正确目标】, 而非
  碰巧长得像随便一张低光图。

关键: 随机配对必须和主结果同尺度才可比。
  - Edge: import 你 edge_consistency.py 里的函数, 保证同尺度。
  - LPIPS/SSIM: 本脚本同时算 matched 和 shuffled, 并打印 matched 与
    你已报告值(0.7159/0.1795 等)对比。matched 对得上 => shuffled 可信。

打乱方式: 固定 roll(+SHIFT), 保证无自配对 (derangement), 可复现。
================================================================
前置: pip install lpips scikit-image
      本脚本需放在 ~/nobackup 下 (与 edge_consistency.py 同目录) 才能 import。
      若 import edge_consistency 触发它跑全套评估, 说明该文件的主逻辑没有
      放在 `if __name__ == "__main__":` 下 —— 加上即可。
================================================================
"""

import os
import sys
import json
import random

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim_fn

# ----------------------------------------------------------------------
BASE = os.path.expanduser("~/nobackup")
sys.path.insert(0, BASE)

BASELINE_GEN = os.path.join(BASE, "eval_results/M2_generated")   # Baseline
ENHANCED_GEN = os.path.join(BASE, "eval_results/M1_generated")   # Enhanced
L0_DIR       = os.path.join(BASE, "ablation_dataset/test/images")

BASELINE_DEPTH = os.path.join(BASE, "ablation_dataset/test/depths")   # baseline depth maps
ENHANCED_DEPTH = os.path.join(BASE, "hpc_ready_dataset/test/depths")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIZE = 512
SHIFT = 1

REPORTED = {
    "baseline": {"lpips": 0.7159, "ssim": 0.1795, "edge": 0.1873},
    "enhanced": {"lpips": 0.7052, "ssim": 0.1766, "edge": 0.2049},
}

try:
    from edge_consistency import edge_consistency as edge_fn
    EDGE_OK = True
except Exception as e:
    print(f"[WARN] 无法 import edge_consistency.edge_consistency: {e}")
    print("       Edge 随机配对将跳过 (LPIPS/SSIM 不受影响)。")
    edge_fn = None
    EDGE_OK = False


# ----------------------------------------------------------------------
def list_pngs(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(".png"))


def find_gt(stem):
    for ext in (".jpg", ".jpeg", ".png"):
        p = os.path.join(L0_DIR, stem + ext)
        if os.path.exists(p):
            return p
    return None


_to_tensor = transforms.Compose([
    transforms.Resize((SIZE, SIZE)),
    transforms.ToTensor(),
])


def load_lpips_tensor(path):
    """LPIPS 需 [-1,1]"""
    t = _to_tensor(Image.open(path).convert("RGB"))
    return (t * 2 - 1).unsqueeze(0).to(DEVICE)


def load_np01(path):
    img = Image.open(path).convert("RGB").resize((SIZE, SIZE))
    return np.asarray(img).astype(np.float32) / 255.0


# ----------------------------------------------------------------------
def run_pipeline(name, gen_dir, depth_dir, lpips_model):
    gen_files = list_pngs(gen_dir)
    stems = [os.path.splitext(f)[0] for f in gen_files]
    n = len(stems)
    shuffled_idx = [(i + SHIFT) % n for i in range(n)]

    res = {k: {"matched": [], "shuffled": []} for k in ("lpips", "ssim", "edge")}

    for i, fname in enumerate(gen_files):
        stem = stems[i]
        gen_path = os.path.join(gen_dir, fname)

        gt_m = find_gt(stem)
        gt_s = find_gt(stems[shuffled_idx[i]])
        dp_m = os.path.join(depth_dir, stem + ".png")
        dp_s = os.path.join(depth_dir, stems[shuffled_idx[i]] + ".png")

        # ---- LPIPS ----
        if gt_m and gt_s:
            g = load_lpips_tensor(gen_path)
            with torch.no_grad():
                res["lpips"]["matched"].append(
                    float(lpips_model(g, load_lpips_tensor(gt_m)).item()))
                res["lpips"]["shuffled"].append(
                    float(lpips_model(g, load_lpips_tensor(gt_s)).item()))
            # ---- SSIM ----
            gnp = load_np01(gen_path)
            res["ssim"]["matched"].append(
                float(ssim_fn(gnp, load_np01(gt_m), channel_axis=2, data_range=1.0)))
            res["ssim"]["shuffled"].append(
                float(ssim_fn(gnp, load_np01(gt_s), channel_axis=2, data_range=1.0)))

        # ---- Edge ----
        if EDGE_OK and os.path.exists(dp_m) and os.path.exists(dp_s):
            try:
                res["edge"]["matched"].append(float(edge_fn(dp_m, gen_path, SIZE)))
                res["edge"]["shuffled"].append(float(edge_fn(dp_s, gen_path, SIZE)))
            except Exception as e:
                print(f"    [skip edge] {stem}: {e}")

    return res


def summarize(name, res):
    lines = [f"\n----- {name} -----"]
    for k in ("lpips", "ssim", "edge"):
        m = res[k]["matched"]
        s = res[k]["shuffled"]
        if not m:
            lines.append(f"  {k.upper():6}: (无数据)")
            continue
        mm, sm = float(np.mean(m)), float(np.mean(s))
        rep = REPORTED[name].get(k)
        chk = ""
        if rep is not None:
            diff = abs(mm - rep)
            flag = "OK" if diff < (0.03 if k != "edge" else 0.01) else "⚠尺度不一致!"
            chk = f"  [matched≈已报告{rep} ? diff={diff:.4f} {flag}]"
        gap = mm - sm
        lines.append(f"  {k.upper():6}: matched={mm:.4f}  shuffled={sm:.4f}  gap={gap:+.4f}{chk}")
    return "\n".join(lines)


def main():
    print(f"设备: {DEVICE}")
    import lpips
    lpips_model = lpips.LPIPS(net="alex").to(DEVICE)

    out = []
    out.append("=" * 70)
    out.append("随机配对下限 (matched vs shuffled)")
    out.append("方向: LPIPS 越低越好 -> matched 应 < shuffled")
    out.append("      SSIM/Edge 越高越好 -> matched 应 > shuffled")
    out.append("=" * 70)

    for name, gen_dir, depth_dir in [
        ("baseline", BASELINE_GEN, BASELINE_DEPTH),
        ("enhanced", ENHANCED_GEN, ENHANCED_DEPTH),
    ]:
        res = run_pipeline(name, gen_dir, depth_dir, lpips_model)
        block = summarize(name, res)
        print(block)
        out.append(block)

    out.append("\n" + "=" * 70)
    out.append("读法:")
    out.append("  * gap 越大, 说明生成图越是在重建【正确目标】而非随便一张低光图。")
    out.append("  * 若某指标 matched≈shuffled (gap≈0), 说明该指标在低光域判别力弱")
    out.append("    —— 对 SSIM 若如此, 正好解释你已知的 SSIM 不显著。")
    out.append("  * [尺度校验] matched 若≈已报告值, 证明本脚本与 evaluate.py 同尺度,")
    out.append("    shuffled 数可直接与主表并列; 若 ⚠, 需把 LPIPS net 调成与 evaluate.py 一致。")
    out.append("=" * 70)

    summary = "\n".join(out)
    with open(os.path.join(BASE, "eval_results", "anchor_lowerbounds.txt"),
              "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(f"\n已写入 eval_results/anchor_lowerbounds.txt")


if __name__ == "__main__":
    main()
