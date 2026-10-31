#!/bin/bash
#SBATCH --job-name=stack-image
#SBATCH --account=eecs
#SBATCH --partition=share,preempt
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/nfs/hpc/share/%u/rlRobots/runs/slurm/stack-image-%j.log
# Build the stack image from containers/stack/stack.def (fakeroot, no root needed) and check it imports everything.
#   sbatch containers/stack/build.sbatch            -> $S/images/stack-<date>.sif, and stack.sif pointing at it
# Code edits never need this: only a change to requirements*.txt or stack.def does.
SHARE=${SHARE:-/nfs/hpc/share/$USER}  # your cluster share folder
set -euo pipefail
S=$SHARE; C=$S/rlRobots/containers/stack; OUT=$S/images; TAG=${TAG:-$(date +%Y%m%d-%H%M)}
mkdir -p $OUT $S/images/cache
T=$S/images/tmp-$SLURM_JOB_ID; mkdir -p $T; trap "rm -rf $T" EXIT  # node /tmp may be too small for a ~20 GB build
export APPTAINER_CACHEDIR=$S/images/cache APPTAINER_TMPDIR=$T
cd $C
apptainer build --fakeroot $OUT/stack-$TAG.sif stack.def
ln -sfn stack-$TAG.sif $OUT/stack.sif
ls -la $OUT
echo IMAGE-DONE $OUT/stack-$TAG.sif
