#!/bin/bash
#SBATCH --job-name=crop-vlm
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere,dgxh,preempt
#SBATCH --constraint=a40|l40s|h100|h200
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=runs/slurm/%x-%j.log
# Detector-first scene listing (vla/crop_list.py name) with a model served by vLLM in the container, instead of Ollama:
#   SERVER=qwen  Qwen3.8-27B (FP8) through /v1/chat/completions: closed, open, closed on the true boxes
#   SERVER=jev   JEV-9B (autotrust; Qwen3.5-9B + its System 1 adapter, vl/serve_decide.py) through /v1/decide:
#                closed and closed on the true boxes (a choice with calibrated probabilities)
# sbatch --export=ALL,SERVER=jev vla/crop_vlm.sbatch [--limit=N for a smoke test]   (open rows: then vla/crop_judge.sbatch)
SHARE=${SHARE:-/nfs/hpc/share/$USER}  # your cluster share folder
set -uo pipefail
export HF_HOME=$SHARE/hf_cache HF_HUB_OFFLINE=1 PYTHONPATH=.
export VLLM_CACHE_ROOT=/tmp/vllm-$SLURM_JOB_ID
SIF=$SHARE/llmserve/vllm.sif  # vLLM 0.24
PY=.venv-vla/bin/python
PORT=$((20000 + SLURM_JOB_ID % 10000))
export OLLAMA_HOST=127.0.0.1:$PORT  # crop_list.py's server address
nvidia-smi --query-gpu=name,memory.total --format=csv
case ${SERVER:?SERVER=qwen or jev} in
  qwen)
    MODEL=Qwen/Qwen3.8-27B-FP8
    apptainer exec --nv --bind $SHARE $SIF vllm serve $MODEL --port $PORT --max-model-len 16384 \
      --limit-mm-per-prompt '{"image": 2}' --max-num-seqs 16 > runs/slurm/vllm-$SLURM_JOB_ID.err 2>&1 &
    RUNS=("--mode=closed" "--mode=open" "--mode=closed --boxes=truth")
    ARGS=(--api=openai --model=$MODEL --tag=qwen3.8-27b-fp8);;
  jev)  # serve_decide.py needs a newer vLLM than 0.24 (vllm.entrypoints.launchers): llmserve/pull.sbatch builds it
    SIF=$SHARE/llmserve/vllm-latest.sif
    H=$HF_HOME/hub
    JEV=$(ls -d $H/models--autotrust--JEV-9B/snapshots/*)
    BASE=$H/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a  # the revision vl/serve.sh pins
    apptainer exec --nv --bind $SHARE $SIF python3 $JEV/vl/serve_decide.py --model $BASE \
      --served-model-name autotrust/JEV-9B --enable-lora --max-lora-rank 32 --lora-modules jev-decision=$JEV/vl/adapter_vllm \
      --logprobs-mode processed_logprobs --max-model-len 32768 --enable-prefix-caching --mamba-cache-mode align \
      --limit-mm-per-prompt '{"image": 8}' --max-num-seqs 8 --max-logprobs 256 --trust-request-chat-template \
      --port $PORT > runs/slurm/vllm-$SLURM_JOB_ID.err 2>&1 &
    RUNS=("--mode=closed" "--mode=closed --boxes=truth")
    ARGS=(--api=jev --tag=jev-9b --workers=8);;
esac
SERVE=$!
for i in $(seq 180); do curl -s -o /dev/null http://$OLLAMA_HOST/health && break; kill -0 $SERVE 2>/dev/null || break; sleep 5; done
curl -s -o /dev/null http://$OLLAMA_HOST/health || { echo "server did not start"; tail -30 runs/slurm/vllm-$SLURM_JOB_ID.err; exit 1; }
for r in "${RUNS[@]}"; do
  $PY vla/crop_list.py name $r "${ARGS[@]}" "$@"
done
kill $SERVE
