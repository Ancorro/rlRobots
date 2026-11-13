#!/bin/bash
# Match the open crop names to the truth (vla/crop_list.py judge = list_check's judge on runs/croplist), vLLM container;
# then report and figure. sbatch --dependency=afterok:<crop_name job> vla/crop_judge.sbatch
#SBATCH --job-name=so101-croplist-judge
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere,dgxh,preempt
#SBATCH --constraint=a40|l40s
#SBATCH --exclude=sail-gpu0
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=runs/slurm/%x-%j.log
SHARE=${SHARE:-/nfs/hpc/share/$USER}  # your cluster share folder
set -uo pipefail
export HF_HOME=$SHARE/hf_cache HF_HUB_OFFLINE=1 PYTHONPATH=. VLLM_CACHE_ROOT=/tmp/vllm-croplist-$SLURM_JOB_ID
for run in ${RUNS:-gemma4-12b-q4-open-off gemma4-12b-q4-ctx-open-off}; do
  apptainer exec --nv --bind $SHARE $SHARE/llmserve/vllm.sif \
    python3 vla/crop_list.py judge --run=$run "$@"
done
.venv-vla/bin/python vla/crop_list.py report "$@"
.venv-vla/bin/python vla/crop_list.py figure "$@"
rm -rf /tmp/vllm-croplist-$SLURM_JOB_ID
