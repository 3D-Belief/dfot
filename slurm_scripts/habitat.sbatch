#!/bin/bash
#SBATCH --job-name=dfot_habitat_4H100_whole_dataset
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/tshu2/zwen19/diffusion-forcing-transformer/logs/slurm_%j.out
#SBATCH --nodes=1
#SBATCH --partition=h100
#SBATCH --ntasks-per-node=4          # one launcher per node
#SBATCH --gres=gpu:4         # 4 GPUs on this single node
#SBATCH --cpus-per-task=8

# ---- env ----
source ~/.bashrc         # if conda needs bashrc
conda activate dfot
nvidia-smi || true

# <<< W&B to scratch + no ckpt uploads >>>
export XDG_DATA_HOME=/scratch/tshu2/zwen19/.local/share
export XDG_CACHE_HOME=/scratch/tshu2/zwen19/.cache
export WANDB_DIR=/scratch/tshu2/zwen19/wandb
export WANDB_CACHE_DIR=/scratch/tshu2/zwen19/wandb-cache
export WANDB_IGNORE_GLOBS="**/*.ckpt"

# ---- launch (single node, no master addr/port needed) ----
cd /scratch/tshu2/zwen19/diffusion-forcing-transformer/ || exit 1

python -m main +name=dfot_habitat_whole_dataset_phase3 dataset=habitat algorithm=dfot_video_pose experiment=video_generation @diffusion/continuous
# python -m main +name=dfot_spoc dataset=spoc algorithm=dfot_video_pose experiment=video_generation @diffusion/continuous