#!/bin/bash
# Planner benchmark through Ollama (the quantized model a PC runs), one GPU. Pull the model on the login node first
# (compute nodes are offline):  OLLAMA_MODELS=... ollama serve & ollama pull gemma4:12b
# Submit from the cluster checkout:  sbatch vla/plan_ollama.sbatch --model=gemma4:12b --tag=gemma4-12b-q4 --think=on
#SBATCH --job-name=so101-plan-ollama
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere,dgxh,preempt
#SBATCH --constraint=a40|l40s|h100|h200
#SBATCH --exclude=cn-r-5
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=runs/slurm/%x-%j.log
SHARE=${SHARE:-/nfs/hpc/share/$USER}  # your cluster share folder
set -uo pipefail
B=$SHARE/ollama-bin  # upstream Ollama 0.32.0 (the cluster module's 0.11.4 skips CUDA)
export PATH=$B/bin:$PATH LD_LIBRARY_PATH=$B/lib/ollama:${LD_LIBRARY_PATH:-}
export OLLAMA_MODELS=$SHARE/ollama-models OLLAMA_NUM_PARALLEL=8 OLLAMA_CONTEXT_LENGTH=16384
export OLLAMA_HOST=127.0.0.1:$((20000 + SLURM_JOB_ID % 10000))  # own port: other jobs may share the node
nvidia-smi --query-gpu=name,memory.total --format=csv
ollama serve > runs/slurm/ollama-$SLURM_JOB_ID.err 2>&1 &
SERVE=$!
for i in $(seq 60); do curl -s -o /dev/null http://$OLLAMA_HOST/api/tags && break; sleep 2; done
PYTHONPATH=. .venv-vla/bin/python plan_bench.py ollama "$@"
kill $SERVE
