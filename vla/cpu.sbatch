#!/bin/bash
# Any one-off CPU script on the OSU cluster (no GPU: IK, physics-only checks, result summaries).
# Submit from the cluster checkout:  sbatch vla/cpu.sbatch -m so101.g1 claw   (args go to python; log in runs/slurm/)
#SBATCH --job-name=so101-cpu
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=runs/slurm/%x-%j.log
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.${PYTHONPATH:+:$PYTHONPATH} PYTHONUNBUFFERED=1
${PY:-.venv-vla/bin/python} "$@"
