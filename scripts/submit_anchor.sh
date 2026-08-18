#!/bin/bash
#SBATCH --job-name=eval_all
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu021
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=eval_all_%j.out
#SBATCH --error=eval_all_%j.err

set -e

module load cuda/12.4.1
source ~/.bashrc
conda activate venv_yb

cd ~/nobackup

# Install dependencies before the first run (disable after the first successful installation):
pip install --quiet pyiqa open_clip_torch pytorch-fid

python anchor_lowerbounds.py
