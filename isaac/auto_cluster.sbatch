#!/bin/bash
# One Isaac job for isaac/auto_rl.py (the auto RL pipeline): like walk_cluster.sbatch, but the time limit, partition and
# GPU constraint come from sbatch's command line, and the run log is auto-run-<jobid>.log.
#   CODE=rlRobots SCRIPT=isaac/g1_dexsuite.py ARGS="train ..." sbatch --time=23:00:00 isaac/auto_cluster.sbatch
#SBATCH --job-name=g1-auto
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere,preempt
#SBATCH --constraint=rtx8000|a40|l40s
#SBATCH --gres=gpu:1
#SBATCH --exclude=sail-gpu0,cn-r-5
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/nfs/hpc/share/%u/isaac/auto-%j.log
SHARE=${SHARE:-/nfs/hpc/share/$USER}  # your cluster share folder
I=$SHARE/isaac
CODE=${CODE:-rlRobots}; SCRIPT=${SCRIPT:-isaac/g1_dexsuite.py}; ARGS=${ARGS:-}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export APPTAINERENV_PYTHONNOUSERSITE=1 APPTAINERENV_OMNI_KIT_ACCEPT_EULA=YES APPTAINERENV_ACCEPT_EULA=Y APPTAINERENV_PRIVACY_CONSENT=Y
export APPTAINERENV_GIT_PYTHON_REFRESH=quiet APPTAINERENV_PYTHONUNBUFFERED=1 APPTAINERENV_PYTHONPATH=$I/pylib
export APPTAINERENV_ISAACLAB=$I/IsaacLab APPTAINERENV_SHARE=$SHARE
apptainer exec --cleanenv --writable --nv --home $I/../isaac-home --bind $SHARE $I/isaac-sb bash -c "
cd $I/$CODE && /isaac-sim/python.sh $SCRIPT $ARGS > $I/auto-run-$SLURM_JOB_ID.log 2>&1; rc=\$?
grep -aE '^(EVAL|WHY|BASE|TASK|SMOOTH|TRAIN-DONE|TIME-LIMIT|PROBE|resuming|warm start|settings)|Error|error:|Traceback' $I/auto-run-$SLURM_JOB_ID.log | grep -v WARN | tail -30
grep -a '^BENCH_RESULT ' $I/auto-run-$SLURM_JOB_ID.log | tail -1
[ \$rc = 0 ] && grep -aq '^Traceback' $I/auto-run-$SLURM_JOB_ID.log && rc=3
echo AUTO-DONE rc=\$rc; exit \$rc"
rc=$?
exit $rc
