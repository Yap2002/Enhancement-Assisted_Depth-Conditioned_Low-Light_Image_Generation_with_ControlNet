#!/bin/bash
#SBATCH --job-name=FID_Eval
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclude=gpu021
#SBATCH --output=fid_log.out
#SBATCH --error=fid_log.err

module load cuda/12.4.1
source /users/fkwt0359/miniconda3/bin/activate venv_yb
export PYTHONUNBUFFERED=1

pip install pytorch-fid --quiet

echo "FID Evaluation: M1 vs M2"

cd /users/fkwt0359/nobackup
python compute_fid.py

echo ""
echo "Finished. Results are in fid_results.txt"
