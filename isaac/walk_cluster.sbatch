#!/bin/bash
# Walk-and-grasp G1 runs (isaac/walk_grasp.py, isaac/locomanip_walk.py) in the Isaac Sim sandbox on one cluster GPU.
#   CODE=rlg1h SCRIPT=isaac/walk_grasp.py ARGS="--shape=box --n=12" sbatch isaac/walk_cluster.sbatch   (from $I)
#SBATCH --job-name=g1-walk
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere,preempt
#SBATCH --constraint=a40|l40s|rtx8000|rtx6000
#SBATCH --gres=gpu:1
#SBATCH --exclude=sail-gpu0,cn-r-5
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/nfs/hpc/share/%u/isaac/walk-%j.log
SHARE=${SHARE:-/nfs/hpc/share/$USER}  # your cluster share folder
I=$SHARE/isaac
CODE=${CODE:-rlg1h}; SCRIPT=${SCRIPT:-isaac/walk_grasp.py}; ARGS=${ARGS:-}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export APPTAINERENV_PYTHONNOUSERSITE=1 APPTAINERENV_OMNI_KIT_ACCEPT_EULA=YES APPTAINERENV_ACCEPT_EULA=Y APPTAINERENV_PRIVACY_CONSENT=Y
export APPTAINERENV_GIT_PYTHON_REFRESH=quiet APPTAINERENV_PYTHONUNBUFFERED=1 APPTAINERENV_PYTHONPATH=$I/pylib
# The job ends with the run's exit code (audit 2026-10-23: a crashed run ended 0 and looked like a finished one), and the
# run's BENCH_RESULT line is copied into this job log, where scripts/check_day*.py read it.
apptainer exec --cleanenv --writable --nv --home $I/../isaac-home --bind $SHARE $I/isaac-sb bash -c "
cd $I/$CODE && /isaac-sim/python.sh $SCRIPT $ARGS > $I/walk-run-$SLURM_JOB_ID.log 2>&1; rc=\$?
grep -aE 'robots|lifted|held|success|pinch-to|recorded|fell|Error|error:' $I/walk-run-$SLURM_JOB_ID.log | grep -v WARN | tail -20
grep -a '^BENCH_RESULT ' $I/walk-run-$SLURM_JOB_ID.log | tail -1
[ \$rc = 0 ] && grep -aq '^Traceback' $I/walk-run-$SLURM_JOB_ID.log && rc=3  # Kit exits 0 after an uncaught exception
echo WALK-DONE rc=\$rc; exit \$rc"
rc=$?
exit $rc
