#!/bin/bash
# GPU part of the MJWarp feasibility study (so101/g1_warp.py): one cluster GPU, own venv .venv-warp.
# Submit from $SHARE/rlRobots-warp:  sbatch warp/gpu.sbatch "gpu+--tapes=runs/warp/tapes-n100.pkl" ["speed+..."]
# Other GPU type: sbatch --partition=dgxh --constraint=h100 warp/gpu.sbatch ...
#SBATCH --job-name=warp-gpu
#SBATCH --account=eecs
#SBATCH --partition=ampere,preempt
#SBATCH --constraint=a40
#SBATCH --exclude=sail-gpu0
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/slurm/%x-%j.log
SHARE=${SHARE:-/nfs/hpc/share/$USER}  # your cluster share folder
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. PYTHONUNBUFFERED=1
export WARP_CACHE_PATH=$SHARE/rlRobots-warp/warp_cache
echo "host $(hostname) cpu $(grep -m1 'model name' /proc/cpuinfo)"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
for cmd in "$@"; do  # several commands in one job: one argument per command, words joined by "+" (the cluster sbatch splits quoted arguments)
  cmd=${cmd//+/ }
  echo "=== $cmd"
  .venv-warp/bin/python -m so101.g1_warp $cmd
done
