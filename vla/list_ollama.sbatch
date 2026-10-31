#!/bin/bash
# Object listing from camera pictures (vla/list_check.py): render the scenes, then ask an Ollama vision model.
# sbatch vla/list_ollama.sbatch [--scenes=200]   (renders once if runs/listing/truth.json is missing, then all asks)
# Another model: sbatch --export=ALL,MODEL=qwen3-vl:8b-instruct-q4_K_M,TAG=qwen3-vl-8b-q4,THINKS=off vla/list_ollama.sbatch
# (pulled into OLLAMA_MODELS inside the job if missing)
#SBATCH --job-name=so101-list
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere,dgxh,preempt
#SBATCH --constraint=a40|l40s|h100|h200
#SBATCH --exclude=cn-r-5
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=runs/slurm/%x-%j.log
SHARE=${SHARE:-/nfs/hpc/share/$USER}  # your cluster share folder
set -uo pipefail
export HF_HOME=$SHARE/hf_cache HF_HUB_OFFLINE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=.
B=$SHARE/ollama-bin
export PATH=$B/bin:$PATH LD_LIBRARY_PATH=$B/lib/ollama:${LD_LIBRARY_PATH:-}
export OLLAMA_MODELS=$SHARE/ollama-models OLLAMA_NUM_PARALLEL=8 OLLAMA_CONTEXT_LENGTH=8192
export OLLAMA_HOST=127.0.0.1:$((20000 + SLURM_JOB_ID % 10000))
nvidia-smi --query-gpu=name,memory.total --format=csv
[ -f runs/listing/truth.json ] || .venv-vla/bin/python vla/list_check.py render "$@" 2>runs/slurm/list-render-$SLURM_JOB_ID.err
ollama serve > runs/slurm/ollama-$SLURM_JOB_ID.err 2>&1 &
SERVE=$!
for i in $(seq 60); do curl -s -o /dev/null http://$OLLAMA_HOST/api/tags && break; sleep 2; done
MODEL=${MODEL:-gemma4:12b} TAG=${TAG:-gemma4-12b-q4}
ollama show $MODEL >/dev/null 2>&1 || ollama pull $MODEL || exit 1
for mode in ${MODES:-open closed}; do for think in ${THINKS:-off on}; do  # MODES=closed sbatch ... to rerun one kind
  .venv-vla/bin/python vla/list_check.py ask --model=$MODEL --tag=$TAG --mode=$mode --think=$think
done; done
kill $SERVE
