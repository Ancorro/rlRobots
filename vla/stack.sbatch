#!/bin/bash
#SBATCH --job-name=stack-v1
#SBATCH --account=eecs
#SBATCH --partition=dgxh  # two GPUs per job: ampere (A40/L40S) allows 2 per user, gpu has only RTX 8000 (2026-10-09)
#SBATCH --gres=gpu:2
#SBATCH --exclude=sail-gpu0,cn-r-5,dgxh-1  # dgxh-1: EGL cannot open /dev/dri (MuJoCo rendering fails), GPUs shared (2026-10-21)
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/nfs/hpc/share/%u/rlRobots/runs/slurm/stack-%j.log
# Stack v1 or v2 (STACK=v2; docs/stack_v1.md, stack_v2.md) on two GPUs: JEV-9B, Gemma and DAM servers (vla/stack_serve.sh), DINO + SAM in-process.
# Usage (from the code copy): MODE=smoke sbatch vla/stack.sbatch          every model loads and answers
#   SCRIPT=ground_bench.py MODE=run ARGS=... sbatch vla/stack.sbatch       another script on the same servers
#                             MODE=run ARGS="--episodes=..." sbatch --time=12:00:00 vla/stack.sbatch
#   IMAGE=$SHARE/images/stack.sif ...      everything but Ollama from the stack image (containers/stack)
SHARE=${SHARE:-/nfs/hpc/share/$USER}  # your cluster share folder
set -uo pipefail
S=$SHARE
cd ${CODE:-$S/stack1}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
source vla/stack_serve.sh
export MUJOCO_GL=egl HF_HOME=$S/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.
PY=${IMAGE:+apptainer exec --nv --bind $S $IMAGE python3}; PY=${PY:-$S/rlRobots/.venv-vla/bin/python}  # IMAGE: the stack image
# GPU memory per process every 5 s (pid, MiB, command), for the per-layer compute table; peak per process printed at the end
GM=$S/rlRobots/runs/slurm/gpumem-$SLURM_JOB_ID.txt
(while true; do nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | while IFS=', ' read p m; do
  [ "$(ps -o user= -p $p 2>/dev/null)" = "$USER" ] && echo "$p $m $(ps -o args= -p $p | cut -c1-90)"; done >> $GM; sleep 5; done) &  # ours only
GM_PID=$!
if [ "${MODE:-smoke}" = smoke ]; then $PY vla/stack_smoke.py; else $PY ${SCRIPT:-stack_${STACK:-v1}.py} ${ARGS:-}; fi
rc=$?
kill $GM_PID 2>/dev/null
echo "GPU memory peak per process (MiB):"; sort -k1,1 -k2,2nr $GM 2>/dev/null | awk '!seen[$1]++' | sort -k2,2nr
echo STACK-DONE rc=$rc
exit $rc
