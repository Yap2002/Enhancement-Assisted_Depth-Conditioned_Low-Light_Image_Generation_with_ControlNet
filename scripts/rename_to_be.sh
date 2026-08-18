#!/bin/bash
set -e
cd ~/nobackup

echo "Copying generated-image directories"
cp -rn eval_results/M1_generated eval_results/generated_enhanced
cp -rn eval_results/M2_generated eval_results/generated_baseline

echo "Copying metric CSV files"
cp -n eval_results/M1_metrics.csv          eval_results/metrics_enhanced.csv
cp -n eval_results/M2_metrics.csv          eval_results/metrics_baseline.csv
cp -n eval_results/edge_consistency_M1.csv eval_results/edge_consistency_enhanced.csv
cp -n eval_results/edge_consistency_M2.csv eval_results/edge_consistency_baseline.csv

echo "Copying loss curves"
cp -n loss_curve.png          loss_curve_enhanced.png
cp -n loss_curve_ablation.png loss_curve_baseline.png

echo "Finished. Existing files remain unchanged for verification."
ls -1 eval_results | grep -E "baseline|enhanced"
