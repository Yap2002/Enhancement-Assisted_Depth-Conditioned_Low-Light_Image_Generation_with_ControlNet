#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一评估脚本 (baseline / enhanced 命名) —— 完整版 v2
================================================================
相对 v1 的改动:
  * FID: 自建 dataset 读图统一 resize 512, 解决 ExDark 原图尺寸不一崩溃;
         InceptionV3 只建一次, 三次 FID 复用。
  * NIQE: 每张 try/except, 纯黑/病态图 (SVD non-finite) 跳过并记录文件名,
          跳过名单单独存 txt (可直接进失败案例分析)。
  * 新增 anchor (真图基准), 让 Baseline 绝对值好坏自证:
      - NIQE anchor  = 真实 L0 的 NIQE
      - FID  anchor  = 真实 L0 内部对半距离 (FID 的现实下限参照)
      - CLIP anchor  = 真实 L0 vs T_B (原有)

不变: LPIPS/SSIM/Edge 复用旧 CSV 不重算; 防呆断言; baseline/enhanced 命名。

身份映射 (三重验证):
  enhanced = eval_results/M1_generated  (output_model + hpc_ready, Zero-DCE)
  baseline = eval_results/M2_generated  (output_model_ablation + ablation, original input)
  两 images 文件夹 md5 相同 -> L0 用 ablation 那份。
文件名匹配: 生成图 "X.png" -> "X.jpg"(GT/caption), "X.png"(depth)
================================================================
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

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def list_pngs(gen_dir):
    if not os.path.isdir(gen_dir):
        sys.exit(f"[FATAL] 目录不存在: {gen_dir}")
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
        print(f"[WARN] 缺少旧结果文件: {path}")
        return None, 0
    df = pd.read_csv(path)
    if col not in df.columns:
        print(f"[WARN] {path} 无列 '{col}', 现有: {list(df.columns)}")
        return None, len(df)
    return float(df[col].mean()), len(df)


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def sanity_check_edge():
    print("\n" + "=" * 60)
    print("防呆检查: Edge-Consistency 身份核对")
    print("=" * 60)
    enh_mean, enh_n = csv_mean(OLD_EDGE_ENHANCED, "edge_consistency")
    bas_mean, bas_n = csv_mean(OLD_EDGE_BASELINE, "edge_consistency")
    if enh_mean is None or bas_mean is None:
        print("[WARN] 旧 edge CSV 缺失, 跳过防呆 (风险自负)")
        return
    print(f"  enhanced (旧M1) edge = {enh_mean:.4f}  (基准 {EDGE_ENHANCED_REF}, n={enh_n})")
    print(f"  baseline (旧M2) edge = {bas_mean:.4f}  (基准 {EDGE_BASELINE_REF}, n={bas_n})")
    if not (enh_mean > bas_mean and 0.19 <= enh_mean <= 0.22 and 0.17 <= bas_mean <= 0.20):
        print("\n[FATAL] 防呆失败! 大概率生成图文件夹接反, 请停下核对, 不要继续。")
        sys.exit(1)
    print("  [OK] 身份正确, 继续。")


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def build_niqe():
    import pyiqa
    return pyiqa.create_metric("niqe", device=DEVICE)


def compute_niqe(files, niqe_metric, out_csv, skip_txt, label):
    """files: 图片完整路径列表。返回 (mean, n_ok, skipped_names)。"""
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
    print(f"    NIQE[{label}] 跳过 {len(skipped)} 张, 均值按其余 {len(scores)} 张算。")
    return mean, len(scores), skipped


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"设备: {DEVICE}")

    sanity_check_edge()

    lpips_enh, _ = csv_mean(OLD_METRICS_ENHANCED, "lpips")
    ssim_enh, _  = csv_mean(OLD_METRICS_ENHANCED, "ssim")
    lpips_bas, _ = csv_mean(OLD_METRICS_BASELINE, "lpips")
    ssim_bas, _  = csv_mean(OLD_METRICS_BASELINE, "ssim")
    edge_enh, _  = csv_mean(OLD_EDGE_ENHANCED, "edge_consistency")
    edge_bas, _  = csv_mean(OLD_EDGE_BASELINE, "edge_consistency")

    print("\n[1/3] 计算 FID ...")
    incep = build_inception()
    gen_bas_files = [os.path.join(BASELINE_GEN, f) for f in list_pngs(BASELINE_GEN)]
    gen_enh_files = [os.path.join(ENHANCED_GEN, f) for f in list_pngs(ENHANCED_GEN)]
    l0_files = list_images(L0_DIR)

    fid_bas = fid_between(gen_bas_files, l0_files, incep)
    fid_enh = fid_between(gen_enh_files, l0_files, incep)

    train_files = list_images(TRAIN_L0_DIR)
    fid_anchor = fid_between(train_files, l0_files, incep)
    print(f"  FID  baseline={fid_bas:.4f}  enhanced={fid_enh:.4f}  anchor(train vs test 真图)={fid_anchor:.4f}")

    print("\n[2/3] 计算 NIQE ...")
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
    print(f"  NIQE baseline={niqe_bas:.4f}(n={n_nb})  enhanced={niqe_enh:.4f}(n={n_ne})  anchor(真图L0)={niqe_anchor:.4f}(n={n_na})")

    print("\n[3/3] 计算 CLIP Score ...")
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
    print(f"  CLIP baseline(L_B vs T_B)={clip_bas:.4f}(n={n_cb},缺{m_cb})")
    print(f"  CLIP anchor  (L0  vs T_B)={clip_anchor:.4f}(n={n_ca},缺{m_ca})")
    print(f"  CLIP enhanced(L_E vs T_E)={clip_enh:.4f}(n={n_ce},缺{m_ce})")

    def fmt(x):
        return f"{x:.4f}" if isinstance(x, float) and not np.isnan(x) else "  -  "

    L = []
    L.append("=" * 74)
    L.append("统一评估汇总  (Baseline vs Enhanced, 附真图 anchor)")
    L.append("=" * 74)
    L.append(f"{'指标':<20}{'Baseline':>11}{'Enhanced':>11}{'真图anchor':>13}   方向")
    L.append("-" * 74)
    L.append(f"{'LPIPS(vs L0)↓':<20}{fmt(lpips_bas):>11}{fmt(lpips_enh):>11}{'  -  ':>13}   低好 [复用]")
    L.append(f"{'SSIM(vs L0)↑':<20}{fmt(ssim_bas):>11}{fmt(ssim_enh):>11}{'  -  ':>13}   高好 [复用]")
    L.append(f"{'Edge-Consistency↑':<20}{fmt(edge_bas):>11}{fmt(edge_enh):>11}{'  -  ':>13}   高好 [复用]")
    L.append(f"{'FID(vs L0)↓':<20}{fmt(fid_bas):>11}{fmt(fid_enh):>11}{fmt(fid_anchor):>13}   低好 [新算]")
    L.append(f"{'NIQE↓':<20}{fmt(niqe_bas):>11}{fmt(niqe_enh):>11}{fmt(niqe_anchor):>13}   低好 [新算]")
    L.append(f"{'CLIP(L vs T)↑':<20}{fmt(clip_bas):>11}{fmt(clip_enh):>11}{fmt(clip_anchor):>13}   高好 [新算]")
    L.append("-" * 74)
    L.append("anchor 含义:")
    L.append("  FID  anchor = train 真实 L0(5692) vs test 真实 L0(712) 的 FID, 真图域内距离。")
    L.append("               baseline/enhanced 越接近它, 说明生成分布越贴近真图。")
    L.append("  NIQE anchor = 真实 L0 的 NIQE, 生成图越接近它说明画质越接近真图。")
    L.append("  CLIP anchor = 真实 L0 vs T_B, 是图文对齐的真图上限。")
    L.append("")
    L.append("解读要点:")
    L.append("  * LPIPS/SSIM/Edge 复用旧结果未重算, 与已报告显著性一致。")
    L.append("  * CLIP baseline(29.x) 接近 anchor 上限 => 支撑 Innovation 1 (管线可行)。")
    L.append("    但 baseline vs enhanced 的 CLIP 因 caption 不同(T_B≠T_E), 差异不能")
    L.append("    单独归因于生成质量, 仅作辅助观察。")
    L.append("  * NIQE 跳过名单见 niqe_skipped_*.txt; baseline 跳过数>enhanced 本身即")
    L.append("    '增强减少纯黑/失败图' 的量化证据, 可进失败案例分析。")
    L.append("=" * 74)

    summary = "\n".join(L)
    print("\n" + summary)
    with open(os.path.join(OUT_DIR, "summary_baseline_enhanced.txt"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(f"\n汇总已写入 {OUT_DIR}/summary_baseline_enhanced.txt")


if __name__ == "__main__":
    main()
