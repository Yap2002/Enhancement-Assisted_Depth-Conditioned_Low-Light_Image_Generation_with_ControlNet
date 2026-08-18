#!/bin/bash
#SBATCH --job-name=Eval_M1_M2
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --exclude=gpu021
#SBATCH --output=eval_log.out
#SBATCH --error=eval_log.err

module load cuda/12.4.1
source /users/fkwt0359/miniconda3/bin/activate venv_yb
export PYTHONUNBUFFERED=1

pip install lpips scikit-image --quiet

echo "============================================"
echo "Stage 4: 批量生成 + 定量评估"
echo "============================================"

echo ""
echo ">>> [1/4] M1 批量生成中..."
python batch_generate.py \
    --model_path /mnt/scratch/fkwt0359/output_model \
    --test_dir ./hpc_ready_dataset/test \
    --output_dir ./eval_results/M1_generated

echo ""
echo ">>> [2/4] M2 批量生成中..."
python batch_generate.py \
    --model_path /mnt/scratch/fkwt0359/output_model_ablation \
    --test_dir ./ablation_dataset/test \
    --output_dir ./eval_results/M2_generated

echo ""
echo ">>> [3/4] M1 定量评估中..."
python evaluate.py \
    --generated_dir ./eval_results/M1_generated \
    --gt_dir ./hpc_ready_dataset/test/images \
    --output_csv ./eval_results/M1_metrics.csv

echo ""
echo ">>> [4/4] M2 定量评估中..."
python evaluate.py \
    --generated_dir ./eval_results/M2_generated \
    --gt_dir ./ablation_dataset/test/images \
    --output_csv ./eval_results/M2_metrics.csv

echo ""
echo "============================================"
echo "全部完成! 结果在 ./eval_results/"
echo "============================================"