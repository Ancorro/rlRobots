#!/bin/bash
# Start the stack's two model servers on this GPU node (source it from a job script, docs/stack_v1.md):
#   JEV-9B (decision, layer 2): its vl/serve_decide.py in the newer vLLM container, POST $JEV_URL/v1/decide
#     JEV_QUANT=fp8 serves it with vLLM's online FP8 weights (half the bf16 size; H100/L40S); JEV_GPU_FRAC its GPU share
#   Gemma 4 12B 4-bit (planner, layer 3): Ollama, $OLLAMA_HOST
#   DAM-3B (perception's namer): vla/dam_serve.py in .venv-dam (transformers 4.51), $DAM_URL
# DINO and SAM run inside the stack's Python process. Servers stop when the job ends.
SHARE=${SHARE:-/nfs/hpc/share/$USER}  # your cluster share folder
S=$SHARE
# IMAGE=<stack .sif> (containers/stack): vLLM servers and DAM-3B run from the stack image instead of vllm-latest.sif / .venv-dam
VLLM_SIF=${IMAGE:-$S/llmserve/vllm-latest.sif}
DAM_PY=${IMAGE:+apptainer exec --nv --bind $S $IMAGE /opt/dam/bin/python}; DAM_PY=${DAM_PY:-$S/rlRobots/.venv-dam/bin/python}
JEV_DIR=$S/hf_cache/hub/models--autotrust--JEV-9B/snapshots/b63f651ce8ed64481d3f5e73ecdb05f740042f01
BASE_DIR=$S/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
PJ=$((18000 + SLURM_JOB_ID % 1000)); PO=$((19000 + SLURM_JOB_ID % 1000))  # own ports: other jobs may share the node
export JEV_URL=http://127.0.0.1:$PJ OLLAMA_HOST=127.0.0.1:$PO
LOGS=${STACK_LOGS:-$S/rlRobots/runs/slurm}; mkdir -p "$LOGS"
# two GPUs: JEV-9B alone on the first (vLLM takes JEV_GPU_FRAC of it); Gemma, DAM and the stack's own models on
# the second (on one 48 GB card they need ~44 GiB: out of memory, 2026-10-09)
G=(${CUDA_VISIBLE_DEVICES//,/ }); GJ=${G[0]}; GO=${G[1]:-${G[0]}}
if [ -n "${GEMMA_VLLM:-}" ]; then
  # Gemma 4 12B on vLLM as the decision model (so101/deciders.py GemmaVLLM): exact letter-only choice
  # (allowed_token_ids + processed log-probabilities); GEMMA_LORA=<adapter dir> adds LoRA module "gemma-decide",
  # used per request for decisions only. Takes JEV's GPU; JEV is not started.
  # 0.85: weights are 23 GiB, so 0.6 of a 46 GiB L40S left no room for the cache (2026-10-21).
  PG=$((16000 + SLURM_JOB_ID % 1000)); export GEMMA_URL=http://127.0.0.1:$PG GEMMA_MODEL=google/gemma-4-12B-it
  GDIR=$(ls -d $S/hf_cache/hub/models--google--gemma-4-12B-it/snapshots/* | head -1)
  APPTAINERENV_CUDA_VISIBLE_DEVICES=$GJ APPTAINERENV_HF_HUB_OFFLINE=1 APPTAINERENV_VLLM_CACHE_ROOT=$S/vllm_cache apptainer exec --nv --bind $S $VLLM_SIF \
    vllm serve $GDIR --served-model-name $GEMMA_MODEL --max-model-len 8192 --gpu-memory-utilization ${GEMMA_GPU_FRAC:-0.85} \
    --logprobs-mode processed_logprobs --max-logprobs 64 --enable-prefix-caching --limit-mm-per-prompt '{"image": 0}' \
    ${GEMMA_LORA:+--enable-lora --max-lora-rank 16 --lora-modules gemma-decide=$GEMMA_LORA} --port $PG > "$LOGS/gemma-$SLURM_JOB_ID.log" 2>&1 &
  JEV_PID=$!
  for i in $(seq 120); do curl -s -o /dev/null $GEMMA_URL/v1/models && break; sleep 5; done
else
APPTAINERENV_CUDA_VISIBLE_DEVICES=$GJ APPTAINERENV_HF_HUB_OFFLINE=1 APPTAINERENV_VLLM_CACHE_ROOT=$S/vllm_cache apptainer exec --nv --bind $S $VLLM_SIF \
  python3 $JEV_DIR/vl/serve_decide.py --model $BASE_DIR --served-model-name autotrust/JEV-9B \
  --enable-lora --max-lora-rank 32 --lora-modules jev-decision=$JEV_DIR/vl/adapter_vllm \
  --logprobs-mode processed_logprobs --max-model-len 8192 --enable-prefix-caching --mamba-cache-mode align \
  --limit-mm-per-prompt '{"image": 8}' --max-num-seqs 8 --max-logprobs 256 --trust-request-chat-template \
  ${JEV_QUANT:+--quantization $JEV_QUANT} --gpu-memory-utilization ${JEV_GPU_FRAC:-0.85} --port $PJ > "$LOGS/jev-$SLURM_JOB_ID.log" 2>&1 &
JEV_PID=$!
# JEV first: vLLM sizes its cache from the GPU memory free while it starts; the others load after it is up
for i in $(seq 120); do curl -s -o /dev/null $JEV_URL/v1/models && break; sleep 5; done
fi
PD=$((17000 + SLURM_JOB_ID % 1000)); export DAM_URL=http://127.0.0.1:$PD
CUDA_VISIBLE_DEVICES=$GO HF_HOME=$S/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $DAM_PY vla/dam_serve.py $PD \
  > "$LOGS/dam-$SLURM_JOB_ID.log" 2>&1 &
DAM_PID=$!
B=$S/ollama-bin  # upstream Ollama 0.32.0 (the cluster module's 0.11.4 skips CUDA)
CUDA_VISIBLE_DEVICES=$GO PATH=$B/bin:$PATH LD_LIBRARY_PATH=$B/lib/ollama:${LD_LIBRARY_PATH:-} OLLAMA_MODELS=$S/ollama-models \
  OLLAMA_NUM_PARALLEL=2 OLLAMA_CONTEXT_LENGTH=16384 ollama serve > "$LOGS/ollama-$SLURM_JOB_ID.log" 2>&1 &
OLLAMA_PID=$!
for i in $(seq 60); do curl -s -o /dev/null $DAM_URL && break; sleep 5; done
for i in $(seq 60); do curl -s -o /dev/null http://$OLLAMA_HOST/api/tags && break; sleep 2; done
curl -s -o /dev/null -w "JEV or Gemma decision server %{http_code}\n" ${GEMMA_URL:-$JEV_URL}/v1/models
curl -s -o /dev/null -w "Ollama %{http_code}\n" http://$OLLAMA_HOST/api/tags
curl -s -o /dev/null -w "DAM %{http_code}\n" $DAM_URL
trap 'kill $JEV_PID $OLLAMA_PID $DAM_PID 2>/dev/null' EXIT
export CUDA_VISIBLE_DEVICES=$GO  # the stack's own models (DINO, SAM, SigLIP)
