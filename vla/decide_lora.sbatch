#!/bin/bash
#SBATCH --job-name=decide-lora
#SBATCH --account=eecs
#SBATCH --partition=dgxh,ampere
#SBATCH --constraint=h100|a40|l40s
#SBATCH --gres=gpu:1
#SBATCH --exclude=sail-gpu0,cn-r-5
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/nfs/hpc/share/%u/rlRobots/runs/slurm/decide-lora-%j.log
# Gemma 4 12B decision LoRA (vla/train_decide_lora.py): decision examples from simulated runs (vla/decision_data.py,
# training seeds 800000+, held-out 810000+), then train and score base vs adapter on the held-out set.
# Usage (from the code copy): CODE=$S/stack2d sbatch vla/decide_lora.sbatch     [EVAL_ONLY=1] [SCENES=800] [TAG=lora1]
SHARE=${SHARE:-/nfs/hpc/share/$USER}  # your cluster share folder
set -uo pipefail
S=$SHARE
cd ${CODE:-$S/stack2d}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export MUJOCO_GL=egl HF_HOME=$S/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.
PY=$S/rlRobots/.venv-vla/bin/python
D=runs/decide; T=${TAG:-lora1}; mkdir -p $D
# perception-style names seen on dev runs (names only, no labels), so the model learns to follow ids
[ -s $D/names.json ] || $PY -c "
import json, glob
n = sorted({p['name'] for f in glob.glob('runs/stack_v2/dev60-*.jsonl') for l in open(f) for p in json.loads(l)['perceived']})
json.dump(n, open('$D/names.json', 'w')); print(len(n), 'perception names')"
[ -s $D/train.jsonl ] || $PY vla/decision_data.py --seed=800000 --scenes=${SCENES:-800} --shuffle=1 --names=$D/names.json --out=$D/train.jsonl || exit 4
[ -s $D/heldout.jsonl ] || $PY vla/decision_data.py --seed=810000 --scenes=150 --shuffle=0 --names=$D/names.json --out=$D/heldout.jsonl || exit 4
# training runs in the vLLM container: its transformers reads Gemma 4's checkpoint (gemma4_unified); .venv-vla's 5.5.4 does
# not. peft and accelerate go into their own folder, no deps (the container's torch and transformers stay).
C="apptainer exec --nv --bind $S --env HF_HOME=$S/hf_cache,HF_HUB_OFFLINE=1,PYTHONPATH=.:$S/pylib-lora $S/llmserve/vllm-latest.sif"
$C python3 -c "import peft, accelerate" 2>/dev/null || $C python3 -m pip install -q --no-deps --target $S/pylib-lora peft accelerate || exit 3
$C python3 -c "import transformers, torch, peft; print('transformers', transformers.__version__, 'torch', torch.__version__, 'peft', peft.__version__)"
$C python3 vla/train_decide_lora.py --train=$D/train.jsonl --eval=$D/heldout.jsonl --out=$D/$T --eval_only=${EVAL_ONLY:-0} ${EXTRA:-}
rc=$?
echo DECIDE-LORA-DONE rc=$rc
exit $rc
