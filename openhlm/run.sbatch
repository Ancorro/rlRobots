#!/bin/bash
# OpenHLM-data: train, select on validation, score the test sample once, run the check. Two A40s (L40S as fallback), 30 minutes;
# ampere allows two GPUs per user (vla/stack.sbatch, 2026-10-09).
# Submit from the cluster checkout (data fetched first on the login node, docs/openhlm.md):
#   sbatch openhlm/run.sbatch                      the real run (TAG=<name>, default the job id; writes logs/openhlm/test-<TAG>.jsonl)
#   FAKE=1 sbatch openhlm/run.sbatch               an 8-minute dry run on generated data (checks the GPUs, the venv and the code path)
#   VAL_ONLY=1 sbatch openhlm/run.sbatch           train and select only; no test record is written
#   REFUSE_RETEST=1 ...                             stop instead of scoring the test again when a record exists (every test
#                                                   run is counted in runs/openhlm/test-history.jsonl and named in the report)
#   CAMERA=auto (default): the vision candidate (mlp-cam) sees every camera whose frame cache is complete for all sets
#                (openhlm/fetch.py builds head_image_left, left_wrist_image, right_wrist_image); CAMERA=head_image_left
#                one camera; CAMERA="" state only; FRAMES=<dir>
#   CANDIDATES=mlp-wide,mlp-cam,act,ridge ...     add ACT from the LeRobot stack (needs lerobot in $PY's venv)
#   IMAGE=$SHARE/images/stack.sif sbatch ...        run inside the stack image (containers/stack; it needs pyarrow, in its requirements)
#   PY=<python> ...                                 another venv (default .venv-vla, then .venv)
# Also runs as a plain script on any machine with the venv and the data:  bash openhlm/run.sbatch
#SBATCH --job-name=openhlm
#SBATCH --account=eecs
#SBATCH --partition=ampere,preempt
#SBATCH --constraint=a40|l40s
#SBATCH --gres=gpu:2
#SBATCH --exclude=sail-gpu0,cn-r-5
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/nfs/hpc/share/%u/rlRobots/runs/slurm/openhlm-%j.log
SHARE=${SHARE:-/nfs/hpc/share/$USER}  # your cluster share folder
set -uo pipefail
# the checkout: where sbatch was run (Slurm copies this script elsewhere), else this file's repository
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" && { [ -f openhlm/pipeline.py ] || cd ..; }
[ -f openhlm/pipeline.py ] || { echo "OPENHLM-DONE rc=2 (submit from the rlRobots checkout)"; exit 2; }
mkdir -p runs/slurm runs/openhlm logs/openhlm
export HF_HOME=${HF_HOME:-$SHARE/hf_cache} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TORCH_HOME=${TORCH_HOME:-$SHARE/torch_cache}
if [ -n "${IMAGE:-}" ]; then PY="apptainer exec --nv --bind $SHARE --env HF_HOME=$HF_HOME,HF_HUB_OFFLINE=1,PYTHONPATH=. $IMAGE python3"
else PY=${PY:-.venv-vla/bin/python}; [ -x "$PY" ] || PY=.venv/bin/python; [ -x "$PY" ] || PY=python3; fi
TAG=${TAG:-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}
BUDGET_MIN=${BUDGET_MIN:-26}   # of the 30-minute job: the rest is start-up and this script's own steps
CAMERA=${CAMERA-auto}; FRAMES=${FRAMES:-runs/openhlm/frames}
EXTRA=${EXTRA:-}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi (CPU run)"
echo "python: $($PY --version 2>&1) at $PY; tag $TAG; budget $BUDGET_MIN min"

# fail fast: the venv must import torch (CUDA), pyarrow and numpy; lerobot only for the act candidate
$PY - <<EOF || { echo "OPENHLM-DONE rc=3 (venv: $PY lacks a package; pip install -r openhlm/requirements.txt)"; exit 3; }
import numpy, pyarrow, torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count(), "GPUs; pyarrow", pyarrow.__version__)
EOF
if [ "${FAKE:-0}" = 1 ]; then  # dry run: generated data, 8 minutes, validation and test on it, everything under runs/openhlm-fake/
    F=runs/openhlm-fake; D=$F/data; [ -f $D/g1_HLM-12_humi/meta/info.json ] || $PY -m openhlm.fake_data $D --sets=5 --episodes=40 ${FAKE_VIDEOS:+--videos}
    DATA="--data $D"; TAG=fake-$TAG; BUDGET_MIN=${FAKE_BUDGET_MIN:-8}; LOGS="--logs $F/logs --out $F --cache $F/cache"; EXTRA="$EXTRA --reserve-min 2"
    [ "$CAMERA" = auto ] && CAMERA=head_image_left,left_wrist_image  # the generated sets' two cameras
    if [ -n "$CAMERA" ]; then
        $PY -c "import json,sys; f=json.load(open('$D/g1_HLM-12_humi/meta/info.json'))['features']; sys.exit(0 if any(v.get('dtype') in ('image','video') for v in f.values()) else 1)" \
            || $PY -m openhlm.fake_data $D --sets=5 --episodes=40 --videos  # the dry run needs camera frames
        for cam in $(echo "$CAMERA" | tr ',' ' '); do $PY -m openhlm.frames --camera $cam --data $D --out $F/frames --size 64 --cache $F/cache || exit 4; done
        FRAMES=$F/frames
    fi
else
    DATA=${DATA_DIR:+--data $DATA_DIR}; LOGS=""
    $PY -m openhlm.fetch --describe --no-verify ${DATA_DIR:+--data $DATA_DIR} ${SETS:+--sets $SETS} || { echo "OPENHLM-DONE rc=4 (data missing: run openhlm/fetch.py on the login node)"; exit 4; }
fi
MODE="--test"; [ "${VAL_ONLY:-0}" = 1 ] && MODE=""; [ "${REFUSE_RETEST:-0}" = 1 ] && MODE="$MODE --refuse-retest"
if [ "$CAMERA" = auto ]; then  # the frame cache decides: a camera counts when every set has <FRAMES>/<set>/<camera>/meta.json and frames
    CAMERA=""
    for cam in head_image_left left_wrist_image right_wrist_image; do
        ok=1
        for s in $(echo "${SETS:-g1_HLM-12_full_teleop,g1_HLM-12_humi,g1_HLM-12_stationary_teleop,long_g1_5_fruits_full_teleop,long_g1_5_fruits_humi}" | tr ',' ' '); do
            [ -f "$FRAMES/$s/$cam/meta.json" ] && ls "$FRAMES/$s/$cam/" | grep -q '\.npy$' || { ok=0; break; }
        done
        [ $ok = 1 ] && CAMERA="${CAMERA:+$CAMERA,}$cam" || echo "no complete frame cache for $cam under $FRAMES (openhlm/fetch.py builds it)"
    done
    [ -n "$CAMERA" ] || echo "no camera: state only"
fi
CANDIDATES=${CANDIDATES:-$([ -n "$CAMERA" ] && echo mlp-wide,mlp-cam,ridge || echo mlp,mlp-wide,ridge)}
echo "camera: ${CAMERA:-none (state only)}; candidates $CANDIDATES"
case ",$CANDIDATES," in *,act,*) $PY -c "import lerobot" || { echo "OPENHLM-DONE rc=3 (act needs lerobot in $PY)"; exit 3; };; esac
CAM=${CAMERA:+--camera $CAMERA --frames $FRAMES}

$PY -m openhlm.pipeline --tag "$TAG" $DATA ${SETS:+--sets $SETS} --candidates "$CANDIDATES" --budget-min "$BUDGET_MIN" $CAM $LOGS $MODE $EXTRA
rc=$?
R=runs/openhlm/$TAG/report.md; [ -f "$R" ] || R=runs/openhlm-fake/$TAG/report.md; [ -f "$R" ] && { echo "---- $R"; sed -n '1,60p' "$R"; }
echo "OPENHLM-DONE rc=$rc tag=$TAG"
exit $rc
