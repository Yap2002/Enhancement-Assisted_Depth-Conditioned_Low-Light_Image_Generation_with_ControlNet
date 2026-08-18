#!/bin/bash
#SBATCH --job-name=Test_Generation
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=16G
#SBATCH --output=test_log.out
#SBATCH --error=test_log.err

module load cuda/12.4.1
source /users/fkwt0359/miniconda3/bin/activate venv_yb

python test.py
