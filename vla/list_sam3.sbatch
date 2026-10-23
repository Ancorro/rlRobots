#!/bin/bash
#SBATCH --job-name=list-sam3
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere,dgxh,preempt
#SBATCH --constraint=a40|l40s|rtx8000|h100|h200
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=runs/slurm/%x-%j.log
# List first, then check each listed name with SAM 3 (vla/list_sam3.py), then the listing report.
# sbatch vla/list_sam3.sbatch runs/listing/<stem>.jsonl ...   (open lists: their -judged.jsonl copies)
SHARE=${SHARE:-/nfs/hpc/share/$USER}  # your cluster share folder
set -uo pipefail
export HF_HOME=$SHARE/hf_cache HF_HUB_OFFLINE=1 PYTHONPATH=.
nvidia-smi --query-gpu=name,memory.total --format=csv
.venv-vla/bin/python vla/list_sam3.py "$@" || exit 1
.venv-vla/bin/python vla/list_check.py report > runs/listing/report-sam3.txt 2>&1
