#!/bin/bash
#SBATCH --job-name=dfot_habitat_4H100_single_image_testing
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/tshu2/zwen19/diffusion-forcing-transformer/logs/slurm_%j.out
#SBATCH --nodes=1
#SBATCH --partition=h100
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8


source ~/.bashrc
conda activate dfot
nvidia-smi || true


export XDG_DATA_HOME=/scratch/tshu2/zwen19/.local/share
export XDG_CACHE_HOME=/scratch/tshu2/zwen19/.cache
export WANDB_DIR=/scratch/tshu2/zwen19/wandb
export WANDB_CACHE_DIR=/scratch/tshu2/zwen19/wandb-cache
export WANDB_IGNORE_GLOBS="**/*.ckpt"


cd /scratch/tshu2/zwen19/diffusion-forcing-transformer/ || exit 1
export CKPT=/scratch/tshu2/zwen19/diffusion-forcing-transformer/outputs/2025-11-16/08-55-40/checkpoints/epoch=23-step=32736.ckpt

python -m main +name=habitat_single_image_test_1 dataset=habitat algorithm=dfot_video_pose experiment=video_generation @diffusion/continuous load=/scratch/tshu2/zwen19/diffusion-forcing-transformer/outputs/best.ckpt 'experiment.tasks=[validation]' experiment.validation.data.shuffle=True dataset.context_length=1 dataset.frame_skip=4 dataset.n_frames=8 experiment.validation.batch_size=1 algorithm.tasks.prediction.history_guidance.name=vanilla +algorithm.tasks.prediction.history_guidance.guidance_scale=4.0