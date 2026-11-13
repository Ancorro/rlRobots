#!/bin/bash
#SBATCH --job-name=g1-grasp
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere,preempt
#SBATCH --constraint=a40|l40s|rtx8000|rtx6000
#SBATCH --gres=gpu:1
#SBATCH --exclude=sail-gpu0,cn-r-5  # apptainer cannot mount /scratch on sail-gpu0; no GPU visible on cn-r-5 (infra slurm.md)
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=/nfs/hpc/share/%u/isaac/grasp-%j.log
# G1 grasp-parameter RL (isaac/g1_grasp_rl.py): train, then evaluate every saved policy on the held-out layouts in one
# scene. Host memory is the limit (~20 MB per env: 2,880 envs ran out of the AWS box's 30 GB), hence 120G here.
# CODE=rlg1b runs the code copy in $I/rlg1b (default rlg1), so jobs already queued keep theirs.
# Usage: N=60 REPS=8 ITERS=60 OUT=runs/isaac/rl-v4 ARGS="--redraw --params=8 --snap=30" sbatch grasp_cluster.sbatch
#        POLICIES=a.pt,b.pt OUT=... [EVAL_SEED=700000] sbatch grasp_cluster.sbatch   (evaluate only; 700000+ = validation
#        layouts for picking a save or a run, never the 900000+ test; training uses 800000+)
SHARE=${SHARE:-/nfs/hpc/share/$USER}  # your cluster share folder
set -uo pipefail
I=$SHARE/isaac
N=${N:-60}; REPS=${REPS:-8}; ITERS=${ITERS:-60}; OUT=${OUT:-runs/isaac/rl-v3}; ARGS=${ARGS:-}; EVAL_N=${EVAL_N:-60}; POLICIES=${POLICIES:-}; EVAL_SEED=${EVAL_SEED:-0}; EVAL_ARGS=${EVAL_ARGS:-}; CODE=${CODE:-rlg1}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
export APPTAINERENV_PYTHONNOUSERSITE=1 APPTAINERENV_OMNI_KIT_ACCEPT_EULA=YES APPTAINERENV_ACCEPT_EULA=Y APPTAINERENV_PRIVACY_CONSENT=Y
export APPTAINERENV_GIT_PYTHON_REFRESH=quiet APPTAINERENV_PYTHONUNBUFFERED=1 APPTAINERENV_PYTHONPATH=$I/pylib
apptainer exec --cleanenv --writable --nv --home $I/../isaac-home --bind $SHARE $I/isaac-sb bash -c "
cd $I/$CODE
if [ -z '$POLICIES' ]; then /isaac-sim/python.sh isaac/g1_grasp_rl.py train --n=$N --reps=$REPS --iters=$ITERS --out=$OUT --unitree=$I/unitree $ARGS \
  > $I/grasp-train-$SLURM_JOB_ID.log 2>&1 || { tail -3 $I/grasp-train-$SLURM_JOB_ID.log; exit 3; }
  grep -E '^iter|^object state|TRAIN-DONE|Error' $I/grasp-train-$SLURM_JOB_ID.log | tail -4; fi
P=\$(ls $OUT/policy-it*.pt 2>/dev/null | sort -V | paste -sd, -); P=\${P:-$OUT/policy.pt}; [ -n '$POLICIES' ] && P=$POLICIES
/isaac-sim/python.sh isaac/g1_grasp_rl.py eval --policy=\$P --n=$EVAL_N --seed=$EVAL_SEED $EVAL_ARGS --out=$OUT --unitree=$I/unitree > $I/grasp-eval-$SLURM_JOB_ID.log 2>&1 || { tail -3 $I/grasp-eval-$SLURM_JOB_ID.log; exit 4; }
grep -E 'EVAL|Error' $I/grasp-eval-$SLURM_JOB_ID.log
"
rc=$?  # non-zero if training or evaluation failed, so --dependency=afterok chains stop
echo GRASP-DONE rc=$rc
exit $rc
