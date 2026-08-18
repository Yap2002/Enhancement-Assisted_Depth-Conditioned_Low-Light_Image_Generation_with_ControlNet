"""Measure agreement between depth edges and generated-image gradients.

Outputs:
    eval_results/edge_consistency_results.txt
    eval_results/edge_consistency_M1.csv
    eval_results/edge_consistency_M2.csv

Requires opencv-python.
"""

import os
import csv
import cv2
import numpy as np

M1_GEN_DIR   = "./eval_results/M1_generated"
M2_GEN_DIR   = "./eval_results/M2_generated"
M1_DEPTH_DIR = "./hpc_ready_dataset/test/depths"
M2_DEPTH_DIR = "./ablation_dataset/test/depths"
OUTPUT_DIR   = "./eval_results"


def edge_consistency(depth_path, gen_path, size=512):
    """Return the mean generated-image gradient at Canny depth edges."""
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    gen   = cv2.imread(gen_path,   cv2.IMREAD_GRAYSCALE)

    if depth is None or gen is None:
        return None

    depth = cv2.resize(depth, (size, size))
    gen   = cv2.resize(gen,   (size, size))

    depth_edges = cv2.Canny(depth, 50, 150).astype(np.float32) / 255.0

    gx   = cv2.Sobel(gen, cv2.CV_64F, 1, 0, ksize=3)
    gy   = cv2.Sobel(gen, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2)

    if grad.max() > 0:
        grad = grad / grad.max()

    if depth_edges.sum() == 0:
        return None

    return float(grad[depth_edges > 0.5].mean())


def evaluate_model(model_name, gen_dir, depth_dir, output_csv):
    scores = []
    per_image = []

    fnames = sorted([f for f in os.listdir(gen_dir) if f.endswith('.png')])
    print(f"\n{model_name}: evaluating {len(fnames)} images...")

    for i, fname in enumerate(fnames):
        name       = os.path.splitext(fname)[0]
        depth_path = os.path.join(depth_dir, name + ".png")
        gen_path   = os.path.join(gen_dir,   fname)

        if not os.path.exists(depth_path):
            continue

        score = edge_consistency(depth_path, gen_path)
        if score is not None:
            scores.append(score)
            per_image.append({'filename': name, 'edge_consistency': score})

        if (i + 1) % 100 == 0 or (i + 1) == len(fnames):
            print(f"  Progress: {i+1}/{len(fnames)}")

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'edge_consistency'])
        writer.writeheader()
        writer.writerows(per_image)
        writer.writerow({'filename': 'MEAN', 'edge_consistency': np.mean(scores)})
        writer.writerow({'filename': 'STD',  'edge_consistency': np.std(scores)})

    return np.mean(scores), np.std(scores), len(scores)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    m1_mean, m1_std, m1_n = evaluate_model(
        "M1", M1_GEN_DIR, M1_DEPTH_DIR,
        os.path.join(OUTPUT_DIR, "edge_consistency_M1.csv")
    )

    m2_mean, m2_std, m2_n = evaluate_model(
        "M2", M2_GEN_DIR, M2_DEPTH_DIR,
        os.path.join(OUTPUT_DIR, "edge_consistency_M2.csv")
    )

    result = f"""
{"=" * 55}
Edge-consistency results (higher is better)
{"=" * 55}
M1 (enhanced): {m1_mean:.4f} ± {m1_std:.4f}  ({m1_n} images)
M2 (original): {m2_mean:.4f} ± {m2_std:.4f}  ({m2_n} images)
Difference (M1-M2): {m1_mean - m2_mean:.4f}
{"=" * 55}
"""
    print(result)

    with open(os.path.join(OUTPUT_DIR, "edge_consistency_results.txt"), 'w') as f:
        f.write(result)

    print(f"Per-image results saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
