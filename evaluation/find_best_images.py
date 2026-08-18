import csv

m1, m2 = {}, {}
with open('./eval_results/M1_metrics.csv') as f:
    for row in csv.DictReader(f):
        if row['filename'] not in ('MEAN','STD'):
            m1[row['filename']] = float(row['lpips'])

with open('./eval_results/M2_metrics.csv') as f:
    for row in csv.DictReader(f):
        if row['filename'] not in ('MEAN','STD'):
            m2[row['filename']] = float(row['lpips'])

diffs = [(f, m2[f]-m1[f]) for f in m1 if f in m2]

print("Five largest LPIPS improvements for M1:")
for f, d in sorted(diffs, key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {f}  difference={d:.4f}  M1={m1[f]:.4f}  M2={m2[f]:.4f}")

print("\nFive largest LPIPS improvements for M2:")
for f, d in sorted(diffs, key=lambda x: x[1])[:5]:
    print(f"  {f}  difference={d:.4f}  M1={m1[f]:.4f}  M2={m2[f]:.4f}")
