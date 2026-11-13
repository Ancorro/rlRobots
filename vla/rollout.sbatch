#!/bin/bash
# Run a VLA (vision-language-action) policy on the simulated SO-101 on one OSU cluster GPU.
# Submit from the cluster checkout:  cd $SHARE/rlRobots && sbatch vla/rollout.sbatch [POLICY] [EPISODES]
#SBATCH --job-name=so101-vla-rollout
#SBATCH --account=eecs
# Several partitions dodge the eecs group GPU cap; the constraint keeps to GPUs that PyTorch 2.11 /
# CUDA 13 still supports (preempt also holds M60, GTX 980/1080 and V100 cards it cannot use).
#SBATCH --partition=gpu,ampere,dgxh,preempt
#SBATCH --constraint=a40|l40s|rtx8000|rtx6000|h100|h200|rtx2080|t4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/slurm/%x-%j.log
SHARE=${SHARE:-/nfs/hpc/share/$USER}  # your cluster share folder
set -euo pipefail
export HF_HOME=$SHARE/hf_cache
export MUJOCO_GL=egl  # headless rendering on the GPU
nvidia-smi --query-gpu=name,memory.total --format=csv
.venv-vla/bin/python vla/rollout.py "${1:-lerobot/smolvla_base}" "${2:-5}"
