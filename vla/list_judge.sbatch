#!/bin/bash
# Match open object lists to the truth with a text model (vla/list_check.py judge), vLLM container, then report.
# sbatch --dependency=afterok:<list job> vla/list_judge.sbatch [run stems; default gemma4-12b-q4-open-off/on]
#SBATCH --job-name=so101-list-judge
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere,dgxh,preempt
#SBATCH --constraint=a40|l40s
#SBATCH --exclude=sail-gpu0  # apptainer cannot mount /scratch there
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=runs/slurm/%x-%j.log
SHARE=${SHARE:-/nfs/hpc/share/$USER}  # your cluster share folder
set -uo pipefail
export HF_HOME=$SHARE/hf_cache HF_HUB_OFFLINE=1 PYTHONPATH=. VLLM_CACHE_ROOT=/tmp/vllm-$SLURM_JOB_ID
for run in ${@:-gemma4-12b-q4-open-off gemma4-12b-q4-open-on}; do
  apptainer exec --nv --bind $SHARE $SHARE/llmserve/vllm.sif \
    python3 vla/list_check.py judge --run=$run
done
.venv-vla/bin/python vla/list_check.py report
