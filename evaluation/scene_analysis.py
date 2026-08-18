"""Compare M1 and M2 metrics by lighting condition and location."""

import os
import csv
import numpy as np
from collections import defaultdict

CLASSLIST_PATH  = "./imageclasslist.txt"
M1_METRICS_PATH = "./eval_results/M1_metrics.csv"
M2_METRICS_PATH = "./eval_results/M2_metrics.csv"
OUTPUT_PATH     = "./eval_results/scene_analysis_results.txt"

CLASS_NAMES = {
    1:'Bicycle', 2:'Boat',  3:'Bottle', 4:'Bus',   5:'Car',
    6:'Cat',     7:'Chair', 8:'Cup',    9:'Dog',   10:'Motorbike',
    11:'People', 12:'Table'
}

LIGHTING_NAMES = {
    1:'Low',    2:'Ambient', 3:'Object', 4:'Single', 5:'Weak',
    6:'Strong', 7:'Screen',  8:'Window', 9:'Shadow', 10:'Twilight'
}


def load_classlist(path):
    lighting_map = {}
    indoor_map   = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4 or not parts[2].isdigit():
                continue
            raw  = os.path.splitext(parts[0])[0]
            cls  = CLASS_NAMES.get(int(parts[1]), '')
            key  = f"{cls}_{raw}"
            lighting_map[key] = int(parts[2])
            indoor_map[key]   = int(parts[3])
    return lighting_map, indoor_map


def load_metrics(path):
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row['filename'] not in ('MEAN', 'STD'):
                data[row['filename']] = {
                    'lpips': float(row['lpips']),
                    'ssim':  float(row['ssim'])
                }
    return data


def main():
    lighting_map, indoor_map = load_classlist(CLASSLIST_PATH)
    m1 = load_metrics(M1_METRICS_PATH)
    m2 = load_metrics(M2_METRICS_PATH)

    by_lt  = defaultdict(lambda: {'m1_l':[], 'm2_l':[], 'm1_s':[], 'm2_s':[]})
    by_ind = defaultdict(lambda: {'m1_l':[], 'm2_l':[], 'm1_s':[], 'm2_s':[]})

    matched = 0
    for fname in m1:
        if fname not in m2 or fname not in lighting_map:
            continue
        lt  = lighting_map[fname]
        ind = indoor_map[fname]
        by_lt[lt]['m1_l'].append(m1[fname]['lpips'])
        by_lt[lt]['m2_l'].append(m2[fname]['lpips'])
        by_lt[lt]['m1_s'].append(m1[fname]['ssim'])
        by_lt[lt]['m2_s'].append(m2[fname]['ssim'])
        by_ind[ind]['m1_l'].append(m1[fname]['lpips'])
        by_ind[ind]['m2_l'].append(m2[fname]['lpips'])
        by_ind[ind]['m1_s'].append(m1[fname]['ssim'])
        by_ind[ind]['m2_s'].append(m2[fname]['ssim'])
        matched += 1

    lines = []
    lines.append(f"Matched images: {matched}/{len(m1)}\n")

    lines.append("=" * 90)
    lines.append("Grouped by lighting condition")
    lines.append(f"{'Lighting':<10} {'M1_LPIPS':>9} {'M2_LPIPS':>9} {'LPIPS_diff':>10} "
                 f"{'M1_SSIM':>9} {'M2_SSIM':>9} {'SSIM_diff':>9} {'N':>5}  {'Assessment'}")
    lines.append("-" * 90)

    for lt in sorted(by_lt):
        d  = by_lt[lt]
        l1 = np.mean(d['m1_l']); l2 = np.mean(d['m2_l'])
        s1 = np.mean(d['m1_s']); s2 = np.mean(d['m2_s'])
        n  = len(d['m1_l'])
        lpips_tag = "M1" if (l2 - l1) > 0.01 else ("M2" if (l2 - l1) < -0.01 else "similar")
        ssim_tag  = "M1" if (s1 - s2) > 0.01 else ("M2" if (s1 - s2) < -0.01 else "similar")
        tag = f"LPIPS:{lpips_tag} SSIM:{ssim_tag}"
        lines.append(f"{LIGHTING_NAMES[lt]:<10} {l1:>9.4f} {l2:>9.4f} {l2-l1:>8.4f} "
                     f"{s1:>9.4f} {s2:>9.4f} {s1-s2:>8.4f} {n:>5}  {tag}")

    lines.append("\n" + "=" * 90)
    lines.append("Grouped by indoor/outdoor location")
    lines.append(f"{'Location':<10} {'M1_LPIPS':>9} {'M2_LPIPS':>9} {'LPIPS_diff':>10} "
                 f"{'M1_SSIM':>9} {'M2_SSIM':>9} {'SSIM_diff':>9} {'N':>5}  {'Assessment'}")
    lines.append("-" * 90)

    for ind, label in [(1, 'Indoor'), (2, 'Outdoor')]:
        if not by_ind[ind]['m1_l']:
            continue
        d  = by_ind[ind]
        l1 = np.mean(d['m1_l']); l2 = np.mean(d['m2_l'])
        s1 = np.mean(d['m1_s']); s2 = np.mean(d['m2_s'])
        n  = len(d['m1_l'])
        lpips_tag = "M1" if (l2 - l1) > 0.01 else ("M2" if (l2 - l1) < -0.01 else "similar")
        ssim_tag  = "M1" if (s1 - s2) > 0.01 else ("M2" if (s1 - s2) < -0.01 else "similar")
        tag = f"LPIPS:{lpips_tag} SSIM:{ssim_tag}"
        lines.append(f"{label:<10} {l1:>9.4f} {l2:>9.4f} {l2-l1:>8.4f} "
                     f"{s1:>9.4f} {s2:>9.4f} {s1-s2:>8.4f} {n:>5}  {tag}")

    output = "\n".join(lines)
    print(output)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        f.write(output)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
