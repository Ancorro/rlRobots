#!/bin/bash
# Detector-first scene listing (vla/crop_list.py): replay truth boxes, Grounding DINO "object." boxes, then name each
# crop with 4-bit Gemma 4 12B in Ollama (closed, open, and closed on the true boxes). Outputs in runs/croplist/.
# sbatch vla/crop_name.sbatch [--limit=N for a smoke test]   (MODES=closed CONTEXTS=on TRUTHBOX=0 STEPS=name: one part)
#SBATCH --job-name=so101-croplist
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere,dgxh,preempt
#SBATCH --constraint=a40|l40s|h100|h200
#SBATCH --exclude=cn-r-5,sail-gpu0
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=runs/slurm/%x-%j.log
SHARE=${SHARE:-/nfs/hpc/share/$USER}  # your cluster share folder
set -uo pipefail
export HF_HOME=$SHARE/hf_cache HF_HUB_OFFLINE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=.
PY=.venv-vla/bin/python
nvidia-smi --query-gpu=name,memory.total --format=csv
STEPS=${STEPS:-truthboxes detect siglip name}
[[ $STEPS == *truthboxes* ]] && { [ -f runs/croplist/known.npz ] || $PY vla/crop_list.py truthboxes || exit 1; }
[[ $STEPS == *detect* ]] && { [ -f runs/croplist/dino-object.json ] || $PY vla/crop_list.py detect || exit 1; }
[[ $STEPS == *siglip* ]] && { $PY vla/crop_list.py siglip && $PY vla/crop_list.py siglip --boxes=truth || exit 1; }
[[ $STEPS == *name* ]] || exit 0
B=$SHARE/ollama-bin
export PATH=$B/bin:$PATH LD_LIBRARY_PATH=$B/lib/ollama:${LD_LIBRARY_PATH:-}
export OLLAMA_MODELS=$SHARE/ollama-models OLLAMA_NUM_PARALLEL=8 OLLAMA_CONTEXT_LENGTH=8192
export OLLAMA_HOST=127.0.0.1:$((20000 + SLURM_JOB_ID % 10000))
ollama serve > runs/slurm/ollama-$SLURM_JOB_ID.err 2>&1 &
SERVE=$!
for i in $(seq 60); do curl -s -o /dev/null http://$OLLAMA_HOST/api/tags && break; sleep 2; done
for ctx in ${CONTEXTS:-off on}; do  # crop alone / whole picture with the box marked + crop
  for mode in ${MODES:-closed open}; do
    $PY vla/crop_list.py name --mode=$mode --context=$ctx "$@"
  done
  [ "${TRUTHBOX:-1}" = 1 ] && $PY vla/crop_list.py name --mode=closed --boxes=truth --context=$ctx "$@"
done
kill $SERVE
