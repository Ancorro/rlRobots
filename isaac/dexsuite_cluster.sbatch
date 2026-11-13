#!/bin/bash
#SBATCH --job-name=g1-dex
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere,preempt
#SBATCH --constraint=a40|l40s|rtx8000|rtx6000
#SBATCH --gres=gpu:1
#SBATCH --exclude=sail-gpu0,cn-r-5  # apptainer cannot mount /scratch on sail-gpu0; no GPU visible on cn-r-5 (infra slurm.md)
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=/nfs/hpc/share/%u/isaac/dex-%j.log
# G1 closed-loop grasp policy (isaac/g1_dexsuite.py: Isaac Lab's Dexsuite lift task adapted to the G1 + Dex3).
# Usage: MODE=train NE=2048 ITERS=5000 OUT=$I/rlg1c/runs/isaac/dex-v1 [ARGS=--resume] sbatch --time=24:00:00 dexsuite_cluster.sbatch
#        MODE=diag NE=512 [CK=a.pt] sbatch --time=00:30:00 dexsuite_cluster.sbatch   # why episodes end
#        MODE=eval CK=a.pt[,b.pt] OUT=... [EVAL_ARGS="--objects=household --n=3"] sbatch dexsuite_cluster.sbatch
SHARE=${SHARE:-/nfs/hpc/share/$USER}  # your cluster share folder
set -uo pipefail
I=$SHARE/isaac
MODE=${MODE:-train}; NE=${NE:-2048}; ITERS=${ITERS:-5000}; OUT=${OUT:-$I/rlg1c/runs/isaac/dex-v1}; ARGS=${ARGS:-}; CK=${CK:-}
EVAL_ARGS=${EVAL_ARGS:-}; CODE=${CODE:-rlg1c}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
export APPTAINERENV_PYTHONNOUSERSITE=1 APPTAINERENV_OMNI_KIT_ACCEPT_EULA=YES APPTAINERENV_ACCEPT_EULA=Y APPTAINERENV_PRIVACY_CONSENT=Y
export APPTAINERENV_GIT_PYTHON_REFRESH=quiet APPTAINERENV_PYTHONUNBUFFERED=1 APPTAINERENV_PYTHONPATH=$I/pylib
apptainer exec --cleanenv --writable --nv --home $I/../isaac-home --bind $SHARE $I/isaac-sb bash -c "
cd $I/$CODE
if [ '$MODE' = train ]; then
  /isaac-sim/python.sh isaac/g1_dexsuite.py train --num_envs=$NE --iters=$ITERS --out=$OUT --unitree=$I/unitree $ARGS > $I/dex-train-$SLURM_JOB_ID.log 2>&1 \
    || { tail -5 $I/dex-train-$SLURM_JOB_ID.log; exit 3; }
  grep -E 'objects:|motor limits|TRAIN-DONE|Error' $I/dex-train-$SLURM_JOB_ID.log | tail -5
elif [ '$MODE' = diag ]; then
  /isaac-sim/python.sh isaac/g1_dexsuite.py diag --num_envs=$NE --checkpoint=$CK --unitree=$I/unitree $ARGS > $I/dex-diag-$SLURM_JOB_ID.log 2>&1 \
    || { tail -5 $I/dex-diag-$SLURM_JOB_ID.log; exit 5; }
  grep -E 'objects:|DIAG|Error' $I/dex-diag-$SLURM_JOB_ID.log
else
  /isaac-sim/python.sh isaac/g1_dexsuite.py eval --checkpoint=$CK --out=$OUT --unitree=$I/unitree $EVAL_ARGS > $I/dex-eval-$SLURM_JOB_ID.log 2>&1 \
    || { tail -5 $I/dex-eval-$SLURM_JOB_ID.log; exit 4; }
  grep -E 'objects:|EVAL|WHY|Error' $I/dex-eval-$SLURM_JOB_ID.log
fi
"
rc=$?  # non-zero if training or evaluation failed, so --dependency=afterok chains stop
echo DEX-DONE rc=$rc
exit $rc
