#!/bin/bash
# CPU part of the MJWarp feasibility study (so101/g1_warp.py) on the OSU cluster, own venv .venv-warp.
# Submit from $SHARE/rlRobots-warp:  sbatch warp/cpu.sbatch "tapes+--n=100+--procs=16" ["cpu+..."]
#SBATCH --job-name=warp-cpu
#SBATCH --account=eecs
#SBATCH --partition=preempt,share
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=runs/slurm/%x-%j.log
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. PYTHONUNBUFFERED=1
echo "host $(hostname) cpu $(grep -m1 'model name' /proc/cpuinfo)"
for cmd in "$@"; do  # several commands in one job: one argument per command, words joined by "+" (the cluster sbatch splits quoted arguments)
  cmd=${cmd//+/ }
  echo "=== $cmd"
  .venv-warp/bin/python -m so101.g1_warp $cmd
done
