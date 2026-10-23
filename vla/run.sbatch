#!/bin/bash
# Any one-off script on one OSU cluster GPU (MuJoCo rendering needs the GPU for EGL).
# Submit from the cluster checkout:  sbatch vla/run.sbatch -m so101.armcal selftest --seeds=10
# Other venv:  sbatch --export=ALL,PY=.venv-depth/bin/python vla/run.sbatch vla/depth_check.py da3-mono
#SBATCH --job-name=so101-run
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere,dgxh,preempt
#SBATCH --constraint=a40|l40s|rtx8000|h100|h200
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=runs/slurm/%x-%j.log
SHARE=${SHARE:-/nfs/hpc/share/$USER}  # your cluster share folder
set -euo pipefail
export HF_HOME=$SHARE/hf_cache HF_HUB_OFFLINE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1
${PY:-.venv-vla/bin/python} "$@"  # PY=.venv-depth/bin/python for the DA3 / MoGe depth models
