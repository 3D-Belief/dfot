#!/bin/bash
#SBATCH --job-name=dfot_habitat_4H100_whole_dataset
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --nodes=1
#SBATCH --partition=h100
#SBATCH --ntasks-per-node=4          # one launcher per node
#SBATCH --gres=gpu:4         # 4 GPUs on this single node
#SBATCH --cpus-per-task=8

# ---- env ----
source ~/.bashrc         # if conda needs bashrc
conda activate dfot
nvidia-smi || true

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/.cache}"
export XDG_DATA_HOME="${XDG_DATA_HOME:-${CACHE_ROOT}/share}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"
export WANDB_DIR="${WANDB_DIR:-${CACHE_ROOT}/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${CACHE_ROOT}/wandb-cache}"
export WANDB_IGNORE_GLOBS="**/*.ckpt"

# ---- launch (single node, no master addr/port needed) ----
cd "${REPO_ROOT}" || exit 1

python -m main +name=dfot_habitat_whole_dataset_phase3 dataset=habitat algorithm=dfot_video_pose experiment=video_generation @diffusion/continuous
# python -m main +name=dfot_spoc dataset=spoc algorithm=dfot_video_pose experiment=video_generation @diffusion/continuous
