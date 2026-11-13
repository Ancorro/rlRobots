#!/bin/bash
#SBATCH --job-name=reid
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere,preempt
#SBATCH --gres=gpu:1
#SBATCH --exclude=sail-gpu0,cn-r-5
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/nfs/hpc/share/%u/rlRobots/runs/slurm/reid-%j.log
# Re-identification benchmark (reid_bench.py): collect looks (DINO + SAM + SigLIP 2 + DINOv2 on one GPU), then score.
# Usage (from the code copy): ARGS="--seed=600000 --episodes=40 --tag=dev" [SCORE="--grid"] sbatch vla/reid.sbatch
SHARE=${SHARE:-/nfs/hpc/share/$USER}  # your cluster share folder
set -uo pipefail
S=$SHARE
cd ${CODE:-$S/stack2}
export MUJOCO_GL=egl HF_HOME=$S/hf_cache PYTHONPATH=.
PY=$S/rlRobots/.venv-vla/bin/python
$PY reid_bench.py collect ${ARGS:-} && $PY reid_bench.py score ${ARGS:-} ${SCORE:-}
rc=$?
echo REID-DONE rc=$rc
exit $rc
