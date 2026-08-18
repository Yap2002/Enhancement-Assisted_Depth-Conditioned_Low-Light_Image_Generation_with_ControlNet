#!/usr/bin/env python3
"""Run the complete baseline-versus-enhanced evaluation.

Existing LPIPS, SSIM, and edge-consistency CSV files are reused. FID, NIQE,
and CLIP metrics are computed here, including real-image reference values.
Baseline denotes M2 (original input); enhanced denotes M1 (Zero-DCE input).
"""

import os
import json
import csv
import sys
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

BASE = os.path.expanduser("~/nobackup")

BASELINE_GEN = os.path.join(BASE, "eval_results/M2_generated")   # original input = Baseline
ENHANCED_GEN = os.path.join(BASE, "eval_results/M1_generated")   # Zero-DCE  = Enhanced

L0_DIR = os.path.join(BASE, "ablation_dataset/test/images")        # real L0 test images (n=712)
TRAIN_L0_DIR = os.path.join(BASE, "ablation_dataset/train/images")

BASELINE_CAP = os.path.join(BASE, "ablation_dataset/test/captions.json")   # T_B
ENHANCED_CAP = os.path.join(BASE, "hpc_ready_dataset/test/captions.json")  # T_E

OLD_EDGE_ENHANCED    = os.path.join(BASE, "eval_results/edge_consistency_M1.csv")
OLD_EDGE_BASELINE    = os.path.join(BASE, "eval_results/edge_consistency_M2.csv")
OLD_METRICS_ENHANCED = os.path.join(BASE, "eval_results/M1_metrics.csv")
OLD_METRICS_BASELINE = os.path.join(BASE, "eval_results/M2_metrics.csv")

OUT_DIR = os.path.join(BASE, "eval_results")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_EXTS = (".jpg", ".jpeg", ".png")

EDGE_ENHANCED_REF = 0.2051
EDGE_BASELINE_REF = 0.1875

FID_SIZE = 512
FID_SEED = 42


def list_pngs(gen_dir):
    if not os.path.isdir(gen_dir):
        sys.exit(f"[FATAL] Directory not found: {gen_dir}")
    return sorted(f for f in os.listdir(gen_dir) if f.lower().endswith(".png"))


def list_images(folder):
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if f.lower().endswith(IMG_EXTS)]


def find_gt(stem):
    for ext in IMG_EXTS:
        p = os.path.join(L0_DIR, stem + ext)
        if os.path.exists(p):
            return p
    return None


def load_captions(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def csv_mean(path, col):
    if not os.path.exists(path):
        print(f"[WARN] Existing result file not found: {path}")
        return None, 0
    df = pd.read_csv(path)
    if col not in df.columns:
        print(f"[WARN] Column '{col}' not found in {path}; available: {list(df.columns)}")
        return None, len(df)
    return float(df[col].mean()), len(df)


def sanity_check_edge():
    print("\nChecking edge-consistency run identities")
    enh_mean, enh_n = csv_mean(OLD_EDGE_ENHANCED, "edge_consistency")
    bas_mean, bas_n = csv_mean(OLD_EDGE_BASELINE, "edge_consistency")
    if enh_mean is None or bas_mean is None:
        print("[WARN] Existing edge CSV files are unavailable; identity check skipped")
        return
    print(f"  enhanced (M1) edge = {enh_mean:.4f} (reference {EDGE_ENHANCED_REF}, n={enh_n})")
    print(f"  baseline (M2) edge = {bas_mean:.4f} (reference {EDGE_BASELINE_REF}, n={bas_n})")
    if not (enh_mean > bas_mean and 0.19 <= enh_mean <= 0.22 and 0.17 <= bas_mean <= 0.20):
        print("\n[FATAL] Edge references do not match. Check the generated-image directory assignments.")
        sys.exit(1)
    print("  [OK] Run identities verified.")


class _FidDataset(torch.utils.data.Dataset):
    def __init__(self, files, size=FID_SIZE):
        self.files = files
        self.tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.tf(Image.open(self.files[i]).convert("RGB"))


def _fid_stats(files, model, batch_size=32):
    dl = torch.utils.data.DataLoader(_FidDataset(files), batch_size=batch_size,
                                     shuffle=False, num_workers=2)
    model.eval()
    feats = []
    with torch.no_grad():
        for batch in dl:
            batch = batch.to(DEVICE)
            pred = model(batch)[0]
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = torch.nn.functional.adaptive_avg_pool2d(pred, (1, 1))
            feats.append(pred.squeeze(3).squeeze(2).cpu().numpy())
    feats = np.concatenate(feats, axis=0)
    return np.mean(feats, axis=0), np.cov(feats, rowvar=False)


def build_inception():
    from pytorch_fid.inception import InceptionV3
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    return InceptionV3([block_idx]).to(DEVICE)


def fid_between(files_a, files_b, model):
    from pytorch_fid.fid_score import calculate_frechet_distance
    m1, s1 = _fid_stats(files_a, model)
    m2, s2 = _fid_stats(files_b, model)
    return calculate_frechet_distance(m1, s1, m2, s2)


def build_niqe():
    import pyiqa
    return pyiqa.create_metric("niqe", device=DEVICE)


def compute_niqe(files, niqe_metric, out_csv, skip_txt, label):
    """Return the mean NIQE score, valid count, and skipped filenames."""
    to_tensor = transforms.ToTensor()
    rows, scores, skipped = [], [], []
    for path in files:
        fname = os.path.basename(path)
        try:
            img = Image.open(path).convert("RGB")
            t = to_tensor(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                s = float(niqe_metric(t).item())
            if not np.isfinite(s):
                raise ValueError("non-finite NIQE")
            rows.append((fname, s))
            scores.append(s)
        except Exception as e:
            print(f"    [skip NIQE:{label}] {fname}: {e}")
            skipped.append(fname)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "niqe"])
        w.writerows(rows)
    if skipped:
        with open(skip_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(skipped) + "\n")
    mean = float(np.mean(scores)) if scores else float("nan")
    print(f"    NIQE[{label}] skipped {len(skipped)} images; mean uses {len(scores)} valid images.")
    return mean, len(scores), skipped


def build_clip():
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval().to(DEVICE)
    return model, preprocess, tokenizer


def _clip_cos(bundle, img_path, caption):
    model, preprocess, tokenizer = bundle
    image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    text = tokenizer([caption]).to(DEVICE)
    with torch.no_grad():
        img_f = model.encode_image(image)
        txt_f = model.encode_text(text)
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
        cos = float((img_f * txt_f).sum().item())
    return cos * 100.0


def compute_clip(gen_dir, caption_json, bundle, out_csv, anchor=False):
    caps = load_captions(caption_json)
    rows, scores, missing = [], [], 0
    for fname in list_pngs(gen_dir):
        stem = os.path.splitext(fname)[0]
        key = stem + ".jpg"
        if key not in caps:
            alt = [k for k in (stem + e for e in IMG_EXTS) if k in caps]
            if not alt:
                missing += 1
                continue
            key = alt[0]
        img_path = find_gt(stem) if anchor else os.path.join(gen_dir, fname)
        if img_path is None:
            missing += 1
            continue
        s = _clip_cos(bundle, img_path, caps[key])
        rows.append((fname, s))
        scores.append(s)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "clip_score"])
        w.writerows(rows)
    return (float(np.mean(scores)) if scores else float("nan")), len(scores), missing


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")

    sanity_check_edge()

    lpips_enh, _ = csv_mean(OLD_METRICS_ENHANCED, "lpips")
    ssim_enh, _  = csv_mean(OLD_METRICS_ENHANCED, "ssim")
    lpips_bas, _ = csv_mean(OLD_METRICS_BASELINE, "lpips")
    ssim_bas, _  = csv_mean(OLD_METRICS_BASELINE, "ssim")
    edge_enh, _  = csv_mean(OLD_EDGE_ENHANCED, "edge_consistency")
    edge_bas, _  = csv_mean(OLD_EDGE_BASELINE, "edge_consistency")

    print("\n[1/3] Computing FID...")
    incep = build_inception()
    gen_bas_files = [os.path.join(BASELINE_GEN, f) for f in list_pngs(BASELINE_GEN)]
    gen_enh_files = [os.path.join(ENHANCED_GEN, f) for f in list_pngs(ENHANCED_GEN)]
    l0_files = list_images(L0_DIR)

    fid_bas = fid_between(gen_bas_files, l0_files, incep)
    fid_enh = fid_between(gen_enh_files, l0_files, incep)

    train_files = list_images(TRAIN_L0_DIR)
    fid_anchor = fid_between(train_files, l0_files, incep)
    print(f"  FID baseline={fid_bas:.4f} enhanced={fid_enh:.4f} reference(train vs test L0)={fid_anchor:.4f}")

    print("\n[2/3] Computing NIQE...")
    niqe_metric = build_niqe()
    niqe_bas, n_nb, _ = compute_niqe(
        gen_bas_files, niqe_metric,
        os.path.join(OUT_DIR, "niqe_baseline.csv"),
        os.path.join(OUT_DIR, "niqe_skipped_baseline.txt"), "baseline")
    niqe_enh, n_ne, _ = compute_niqe(
        gen_enh_files, niqe_metric,
        os.path.join(OUT_DIR, "niqe_enhanced.csv"),
        os.path.join(OUT_DIR, "niqe_skipped_enhanced.txt"), "enhanced")
    niqe_anchor, n_na, _ = compute_niqe(
        l0_files, niqe_metric,
        os.path.join(OUT_DIR, "niqe_real_L0.csv"),
        os.path.join(OUT_DIR, "niqe_skipped_real_L0.txt"), "realL0")
    print(f"  NIQE baseline={niqe_bas:.4f}(n={n_nb}) enhanced={niqe_enh:.4f}(n={n_ne}) reference(L0)={niqe_anchor:.4f}(n={n_na})")

    print("\n[3/3] Computing CLIP score...")
    bundle = build_clip()
    clip_bas, n_cb, m_cb = compute_clip(
        BASELINE_GEN, BASELINE_CAP, bundle,
        os.path.join(OUT_DIR, "clip_baseline.csv"), anchor=False)
    clip_anchor, n_ca, m_ca = compute_clip(
        BASELINE_GEN, BASELINE_CAP, bundle,
        os.path.join(OUT_DIR, "clip_baseline_anchor_L0.csv"), anchor=True)
    clip_enh, n_ce, m_ce = compute_clip(
        ENHANCED_GEN, ENHANCED_CAP, bundle,
        os.path.join(OUT_DIR, "clip_enhanced.csv"), anchor=False)
    print(f"  CLIP baseline(L_B vs T_B)={clip_bas:.4f}(n={n_cb}, missing={m_cb})")
    print(f"  CLIP reference(L0 vs T_B)={clip_anchor:.4f}(n={n_ca}, missing={m_ca})")
    print(f"  CLIP enhanced(L_E vs T_E)={clip_enh:.4f}(n={n_ce}, missing={m_ce})")

    def fmt(x):
        return f"{x:.4f}" if isinstance(x, float) and not np.isnan(x) else "  -  "

    L = []
    L.append("Evaluation summary (Baseline vs Enhanced with real-image references)")
    L.append(f"{'Metric':<20}{'Baseline':>11}{'Enhanced':>11}{'Reference':>13}   Direction")
    L.append(f"{'LPIPS(vs L0)':<20}{fmt(lpips_bas):>11}{fmt(lpips_enh):>11}{'  -  ':>13}   lower [reused]")
    L.append(f"{'SSIM(vs L0)':<20}{fmt(ssim_bas):>11}{fmt(ssim_enh):>11}{'  -  ':>13}   higher [reused]")
    L.append(f"{'Edge consistency':<20}{fmt(edge_bas):>11}{fmt(edge_enh):>11}{'  -  ':>13}   higher [reused]")
    L.append(f"{'FID(vs L0)':<20}{fmt(fid_bas):>11}{fmt(fid_enh):>11}{fmt(fid_anchor):>13}   lower [computed]")
    L.append(f"{'NIQE':<20}{fmt(niqe_bas):>11}{fmt(niqe_enh):>11}{fmt(niqe_anchor):>13}   lower [computed]")
    L.append(f"{'CLIP(L vs T)':<20}{fmt(clip_bas):>11}{fmt(clip_enh):>11}{fmt(clip_anchor):>13}   higher [computed]")
    L.append("Reference definitions:")
    L.append("  FID: train L0 (5692) versus test L0 (712), measuring within-domain distance.")
    L.append("  NIQE: the score of real L0 images.")
    L.append("  CLIP: real L0 images versus baseline captions T_B.")
    L.append("")
    L.append("Interpretation notes:")
    L.append("  * LPIPS, SSIM, and edge consistency are loaded from existing result files.")
    L.append("  * Baseline and enhanced CLIP scores use different caption sets and are not directly causal comparisons.")
    L.append("  * Files skipped during NIQE evaluation are recorded in niqe_skipped_*.txt.")

    summary = "\n".join(L)
    print("\n" + summary)
    with open(os.path.join(OUT_DIR, "summary_baseline_enhanced.txt"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(f"\nSummary saved to {OUT_DIR}/summary_baseline_enhanced.txt")


if __name__ == "__main__":
    main()
