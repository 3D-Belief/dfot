#!/bin/bash
#SBATCH --job-name=dfot_habitat_4H100_single_image_testing
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --nodes=1
#SBATCH --partition=h100
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8


source ~/.bashrc
conda activate dfot
nvidia-smi || true


REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/.cache}"
export XDG_DATA_HOME="${XDG_DATA_HOME:-${CACHE_ROOT}/share}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"
export WANDB_DIR="${WANDB_DIR:-${CACHE_ROOT}/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${CACHE_ROOT}/wandb-cache}"
export WANDB_IGNORE_GLOBS="**/*.ckpt"


cd "${REPO_ROOT}" || exit 1
export CKPT="${CKPT:-checkpoints/dfot_habitat.ckpt}"

python -m main +name=habitat_single_image_test_1 dataset=habitat algorithm=dfot_video_pose experiment=video_generation @diffusion/continuous load="${CKPT}" 'experiment.tasks=[validation]' experiment.validation.data.shuffle=True dataset.context_length=1 dataset.frame_skip=4 dataset.n_frames=8 experiment.validation.batch_size=1 algorithm.tasks.prediction.history_guidance.name=vanilla +algorithm.tasks.prediction.history_guidance.guidance_scale=4.0
