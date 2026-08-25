import os
import sys
import json
import random

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim_fn

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
    print(f"[WARN] Could not import edge_consistency.edge_consistency: {e}")
    print("       Shuffled edge evaluation will be skipped; LPIPS and SSIM are unaffected.")
    edge_fn = None
    EDGE_OK = False


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
    """Load an image as an LPIPS tensor in the range [-1, 1]."""
    t = _to_tensor(Image.open(path).convert("RGB"))
    return (t * 2 - 1).unsqueeze(0).to(DEVICE)


def load_np01(path):
    img = Image.open(path).convert("RGB").resize((SIZE, SIZE))
    return np.asarray(img).astype(np.float32) / 255.0


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

        # LPIPS and SSIM use the same matched and shuffled target pairs.
        if gt_m and gt_s:
            g = load_lpips_tensor(gen_path)
            with torch.no_grad():
                res["lpips"]["matched"].append(
                    float(lpips_model(g, load_lpips_tensor(gt_m)).item()))
                res["lpips"]["shuffled"].append(
                    float(lpips_model(g, load_lpips_tensor(gt_s)).item()))
            gnp = load_np01(gen_path)
            res["ssim"]["matched"].append(
                float(ssim_fn(gnp, load_np01(gt_m), channel_axis=2, data_range=1.0)))
            res["ssim"]["shuffled"].append(
                float(ssim_fn(gnp, load_np01(gt_s), channel_axis=2, data_range=1.0)))

        # Edge consistency uses matched and shuffled depth maps.
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
            lines.append(f"  {k.upper():6}: no data")
            continue
        mm, sm = float(np.mean(m)), float(np.mean(s))
        rep = REPORTED[name].get(k)
        chk = ""
        if rep is not None:
            diff = abs(mm - rep)
            flag = "OK" if diff < (0.03 if k != "edge" else 0.01) else "scale mismatch"
            chk = f"  [reported={rep}, difference={diff:.4f}, {flag}]"
        gap = mm - sm
        lines.append(f"  {k.upper():6}: matched={mm:.4f}  shuffled={sm:.4f}  gap={gap:+.4f}{chk}")
    return "\n".join(lines)


def main():
    print(f"Device: {DEVICE}")
    import lpips
    lpips_model = lpips.LPIPS(net="alex").to(DEVICE)

    out = []
    out.append("Shuffled-pair references (matched versus shuffled)")
    out.append("LPIPS: matched should be lower than shuffled.")
    out.append("SSIM and edge consistency: matched should be higher than shuffled.")

    for name, gen_dir, depth_dir in [
        ("baseline", BASELINE_GEN, BASELINE_DEPTH),
        ("enhanced", ENHANCED_GEN, ENHANCED_DEPTH),
    ]:
        res = run_pipeline(name, gen_dir, depth_dir, lpips_model)
        block = summarize(name, res)
        print(block)
        out.append(block)

    out.append("\nInterpretation:")
    out.append("  * A larger gap indicates stronger sensitivity to the correct target pairing.")
    out.append("  * A near-zero gap indicates weak discrimination within the low-light domain.")
    out.append("  * Reported-value checks confirm that metric configurations use comparable scales.")

    summary = "\n".join(out)
    with open(os.path.join(BASE, "eval_results", "anchor_lowerbounds.txt"),
              "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print("\nResults saved to eval_results/anchor_lowerbounds.txt")


if __name__ == "__main__":
    main()
