#!/bin/bash
#SBATCH --job-name=g1-render
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere,preempt
#SBATCH --gres=gpu:1
#SBATCH --exclude=sail-gpu0,cn-r-5
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/nfs/hpc/share/%u/isaac/render-%j.log
# Videos of a recorded Isaac lift eval (g1_dexsuite.py eval --record), redrawn by MuJoCo (isaac/render_mj.py; EGL on the GPU).
# Usage: REC=.../rec-<tag>.npz OUT=.../clips [CODE=rlg1f] [KIND=lift|walkgrasp] sbatch render_cluster.sbatch
SHARE=${SHARE:-/nfs/hpc/share/$USER}  # your cluster share folder
set -uo pipefail
cd $SHARE/isaac/${CODE:-rlg1f}  # a code copy: isaac/render_mj.py and so101/assets/unitree_g1 (Menagerie G1 + Dex3)
MUJOCO_GL=egl $SHARE/rlRobots/.venv-vla/bin/python isaac/render_mj.py ${KIND:-lift} "$REC" "$OUT"
echo RENDER-DONE rc=$?
