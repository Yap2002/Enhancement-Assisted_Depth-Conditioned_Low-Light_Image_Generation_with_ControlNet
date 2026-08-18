"""Compare M1 and M2 edge consistency by lighting and location."""

import os
import csv
import numpy as np
from collections import defaultdict

CLASSLIST_PATH  = "./imageclasslist.txt"
M1_EC_PATH      = "./eval_results/edge_consistency_M1.csv"
M2_EC_PATH      = "./eval_results/edge_consistency_M2.csv"
OUTPUT_PATH     = "./eval_results/edge_scene_analysis_results.txt"

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


def load_ec(path):
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row['filename'] not in ('MEAN', 'STD'):
                data[row['filename']] = float(row['edge_consistency'])
    return data


def main():
    lighting_map, indoor_map = load_classlist(CLASSLIST_PATH)
    m1 = load_ec(M1_EC_PATH)
    m2 = load_ec(M2_EC_PATH)

    by_lt  = defaultdict(lambda: {'m1':[], 'm2':[]})
    by_ind = defaultdict(lambda: {'m1':[], 'm2':[]})

    matched = 0
    for fname in m1:
        if fname not in m2 or fname not in lighting_map:
            continue
        lt  = lighting_map[fname]
        ind = indoor_map[fname]
        by_lt[lt]['m1'].append(m1[fname])
        by_lt[lt]['m2'].append(m2[fname])
        by_ind[ind]['m1'].append(m1[fname])
        by_ind[ind]['m2'].append(m2[fname])
        matched += 1

    lines = []
    lines.append(f"Matched images: {matched}\n")

    lines.append("=" * 75)
    lines.append("Edge consistency by lighting condition (higher is better)")
    lines.append(f"{'Lighting':<10} {'M1_EC':>8} {'M2_EC':>8} {'M1-M2':>10} {'N':>6}  {'Assessment'}")
    lines.append("-" * 75)

    for lt in sorted(by_lt):
        d   = by_lt[lt]
        v1  = np.mean(d['m1'])
        v2  = np.mean(d['m2'])
        diff = v1 - v2
        n   = len(d['m1'])
        tag = "M1" if diff > 0.005 else ("M2" if diff < -0.005 else "similar")
        lines.append(f"{LIGHTING_NAMES[lt]:<10} {v1:>8.4f} {v2:>8.4f} {diff:>10.4f} {n:>6}  {tag}")

    lines.append("\n" + "=" * 75)
    lines.append("Edge consistency by indoor/outdoor location (higher is better)")
    lines.append(f"{'Location':<10} {'M1_EC':>8} {'M2_EC':>8} {'M1-M2':>10} {'N':>6}  {'Assessment'}")
    lines.append("-" * 75)

    for ind, label in [(1, 'Indoor'), (2, 'Outdoor')]:
        if not by_ind[ind]['m1']:
            continue
        d    = by_ind[ind]
        v1   = np.mean(d['m1'])
        v2   = np.mean(d['m2'])
        diff = v1 - v2
        n    = len(d['m1'])
        tag  = "M1" if diff > 0.005 else ("M2" if diff < -0.005 else "similar")
        lines.append(f"{label:<10} {v1:>8.4f} {v2:>8.4f} {diff:>10.4f} {n:>6}  {tag}")

    all_m1 = [v for d in by_lt.values() for v in d['m1']]
    all_m2 = [v for d in by_lt.values() for v in d['m2']]
    lines.append("\n" + "=" * 75)
    lines.append("Overall")
    lines.append(f"M1: {np.mean(all_m1):.4f} ± {np.std(all_m1):.4f}")
    lines.append(f"M2: {np.mean(all_m2):.4f} ± {np.std(all_m2):.4f}")
    lines.append(f"Difference (M1-M2): {np.mean(all_m1)-np.mean(all_m2):.4f}")

    output = "\n".join(lines)
    print(output)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        f.write(output)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
