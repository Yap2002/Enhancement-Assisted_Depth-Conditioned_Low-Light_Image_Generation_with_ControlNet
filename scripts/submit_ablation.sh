#!/bin/bash
#SBATCH --job-name=ControlNet_Ablation
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --exclude=gpu021
#SBATCH --output=ablation_training_log.out
#SBATCH --error=ablation_training_log.err

# 1. Initialise the runtime environment
module load cuda/12.4.1
source /users/fkwt0359/miniconda3/bin/activate venv_yb

# 2. Stream logs without buffering
export PYTHONUNBUFFERED=1

python train_controlnet.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
 --controlnet_model_name_or_path="lllyasviel/sd-controlnet-depth" \
 --train_data_dir="./ablation_dataset/train" \
 --output_dir="/mnt/scratch/fkwt0359/output_model_ablation" \
 --image_column="image" \
 --conditioning_image_column="conditioning_image" \
 --caption_column="text" \
 --resolution=512 \
 --learning_rate=1e-5 \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --mixed_precision="fp16" \
 --gradient_checkpointing \
 --max_train_steps=10000 \
 --checkpointing_steps=2000