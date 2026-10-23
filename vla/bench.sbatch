#!/bin/bash
# General pick-and-place benchmark (bench_general.py) on one OSU cluster GPU, scenes split over worker processes.
# Submit from the cluster checkout:  cd $SHARE/rlRobots && sbatch vla/bench.sbatch --set=scanned --scenes=200 --tag=x
# Each worker holds DINO + SigLIP + SAM 2 (~1.5 GB GPU) and one MuJoCo scene; 8 workers fit a 48 GB card easily.
#SBATCH --job-name=so101-bench
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere,dgxh,preempt
#SBATCH --constraint=a40|l40s|rtx8000|h100|h200  # 48 GB+: 8 workers ran a 24 GB RTX 6000 out of memory (2026-10-04)
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --output=runs/slurm/%x-%j.log
SHARE=${SHARE:-/nfs/hpc/share/$USER}  # your cluster share folder
set -euo pipefail
export HF_HOME=$SHARE/hf_cache HF_HUB_OFFLINE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl  # headless rendering on the GPU
export OMP_NUM_THREADS=1
nvidia-smi --query-gpu=name,memory.total --format=csv
.venv-vla/bin/python bench_general.py --workers=8 "$@" 2>runs/slurm/bench-$SLURM_JOB_ID.err | grep -v '^{"scene"' || true
