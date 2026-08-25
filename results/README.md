# Evaluation results

This directory contains the numerical outputs retained from the final project workflow.

## Route names

- `M1` or `enhanced`: enhancement-assisted pseudo-conditions.
- `M2` or `baseline`: pseudo-conditions extracted from the original low-light images.

## File groups

- `M1_metrics.csv`, `M2_metrics.csv`: retained LPIPS and SSIM snapshots.
- `metrics_enhanced.csv`, `metrics_baseline.csv`: descriptive aliases of the same route-level measurements.
- `edge_consistency_*.csv`: per-image structural condition-adherence measurements.
- `clip_*.csv`: per-image semantic condition-adherence measurements.
- `niqe_*.csv`: per-image no-reference image-quality measurements.
- `niqe_skipped_*.txt`: samples skipped when NIQE computation did not converge.
- `scene_analysis_results.txt`, `edge_scene_analysis_results.txt`: grouped lighting and environment summaries.
- `anchor_lowerbounds.txt`: shuffled-pair and reference checks.
- `summary_baseline_enhanced.txt`: retained aggregate evaluation snapshot.

## Version and coverage warning

The retained CSV files and `summary_baseline_enhanced.txt` are not all from the same final reporting pass. The LPIPS/SSIM CSV files contain 674 rows, whereas the dissertation's final matched-membership audit reports 672 generated outputs. The retained edge CSV files also contain two more rows than the final coverage reported in the dissertation. The exact two-row reconciliation artefact was not retained, so the files are preserved without retrospective deletion.

The final dissertation reports:

| Metric | Baseline M2 | Enhanced M1 |
| --- | ---: | ---: |
| LPIPS | 0.7168 | 0.7061 |
| SSIM | 0.1796 | 0.1767 |
| Edge consistency | 0.1875 | 0.2051 |
| FID | 83.0217 | 76.1046 |
| NIQE | 5.5188 | 4.8475 |
| CLIP score | 29.1822 | 29.8411 |

Use the final table for dissertation reporting. Use the retained CSV files for transparency and inspection of the available per-image measurements. Do not describe the metric sets as having identical coverage unless their image identifiers have first been matched explicitly.
