#!/bin/bash
#SBATCH --job-name=crop-clip
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere,dgxh,preempt
#SBATCH --constraint=a40|l40s|rtx8000|h100|h200
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=runs/slurm/%x-%j.log
# CLIP-style naming of the detector's crops and of the true boxes (vla/crop_list.py siglip), one model per job:
#   sbatch --export=ALL,MODEL=google/siglip2-so400m-patch16-384,TAG=siglip2-so400m-384 vla/crop_clip.sbatch
#   MODEL=openclip:timm/PE-Core-L-14-336 (Meta Perception Encoder through open_clip, installed per job into
#   $SHARE/pylibs/openclip). The model downloads inside the job (HF_HOME cache).
SHARE=${SHARE:-/nfs/hpc/share/$USER}  # your cluster share folder
set -uo pipefail
export HF_HOME=$SHARE/hf_cache PYTHONPATH=.
PY=.venv-vla/bin/python
nvidia-smi --query-gpu=name,memory.total --format=csv
P=$SHARE/pylibs
if [[ $MODEL == moevie:* ]]; then  # Meta MoE-ViE: its repo's own open_clip (Triton kernels compile at start)
  [ -d $P/moe_vie ] || git clone -q https://github.com/facebookresearch/moe_vie $P/moe_vie || exit 1
  [ -d $P/moevie-deps/einops ] || uv pip install -q --python $PY --target $P/moevie-deps einops ftfy regex wcwidth --no-deps
  export PYTHONPATH=.:$P/moe_vie/src:$P/moevie-deps
elif [[ $MODEL == tulip:* ]]; then  # TULIP: Berkeley's open_clip fork + a checkpoint from their bucket
  [ -d $P/tulip_openclip ] || git clone -q https://github.com/tulip-berkeley/open_clip $P/tulip_openclip || exit 1
  [ -d $P/moevie-deps/einops ] || uv pip install -q --python $PY --target $P/moevie-deps einops ftfy regex wcwidth --no-deps
  CK=${MODEL##*:}
  [ -f $CK ] || { mkdir -p $(dirname $CK); curl -sfL -o $CK "$TULIP_URL" || exit 1; }
  [ -d $P/tulip-deps/timm ] || uv pip install -q --python $PY --target $P/tulip-deps "timm>=1.0.20" --no-deps
  [ -d $P/fgclip-deps/huggingface_hub ] || uv pip install -q --python $PY --target $P/fgclip-deps "transformers==4.57.1" "huggingface_hub==0.35.3" --no-deps
  export PYTHONPATH=.:$P/tulip_openclip/src:$P/tulip-deps:$P/fgclip-deps:$P/moevie-deps  # its tokenizer uses transformers 4.x API
elif [[ $MODEL == fgclip2:* ]]; then  # its remote code was written for transformers 4.57 (newer ones break its text side)
  [ -d $P/fgclip-deps/huggingface_hub ] || uv pip install -q --python $PY --target $P/fgclip-deps "transformers==4.57.1" "huggingface_hub==0.35.3" --no-deps
  export PYTHONPATH=.:$P/fgclip-deps
elif [[ $MODEL == openclip:* ]]; then
  L=$SHARE/pylibs/openclip
  [ -d $L/wcwidth ] || uv pip install -q --python $PY --target $L open_clip_torch ftfy wcwidth --no-deps || exit 1
  $PY -c "import timm" 2>/dev/null || uv pip install -q --python $PY --target $L timm --no-deps
  export PYTHONPATH=.:$L
fi
for b in dino truth; do
  $PY vla/crop_list.py siglip --model=$MODEL --tag=$TAG --boxes=$b || exit 1
done
