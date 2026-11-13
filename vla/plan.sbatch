#!/bin/bash
# Plan-only planner benchmark (plan_bench.py run) for one model on one cluster GPU, in the vLLM container
# ($SHARE/llmserve/vllm.sif, vLLM 0.24.0; a pip install of newer vLLM failed to build llguidance).
# Submit from the cluster checkout:  sbatch vla/plan.sbatch --model=qwen3.5-9b --think=on
# Models over 40 GB in bf16 (qwen3.8-27b): add  --constraint="h100|h200"  before the script name.
#SBATCH --job-name=so101-plan
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere,dgxh,preempt
#SBATCH --constraint=a40|l40s|h100|h200
#SBATCH --exclude=sail-gpu0  # apptainer there cannot mount /scratch (2026-10-06)
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=runs/slurm/%x-%j.log
SHARE=${SHARE:-/nfs/hpc/share/$USER}  # your cluster share folder
set -euo pipefail
export HF_HOME=$SHARE/hf_cache HF_HUB_OFFLINE=1 PYTHONPATH=.
export VLLM_CACHE_ROOT=/tmp/vllm-$SLURM_JOB_ID  # per job: a compile cache shared across GPU types crashed Gemma 4 E4B
nvidia-smi --query-gpu=name,memory.total --format=csv
apptainer exec --nv --bind $SHARE $SHARE/llmserve/vllm.sif python3 plan_bench.py run "$@"
