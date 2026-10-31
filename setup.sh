#!/usr/bin/env bash
# CPU-only setup for the parts of this repo that run without a GPU, Isaac Sim or model downloads:
# the SO-101 MuJoCo tasks, the planner checker, the scene memory and the smoke check.
#   bash setup.sh            -> .venv/ (Python 3.12), then: bash scripts/check_cpu.sh
# Versions match uv.lock (`uv sync` gives the same environment if uv is installed).
# Needs network access for pip. The GPU stacks (vla/requirements.txt, Isaac Lab) are not installed here.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3.12}
command -v "$PY" >/dev/null || PY=python3
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 12), f"Python 3.12 needed, found {sys.version.split()[0]}"'

[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q --index-url https://download.pytorch.org/whl/cpu "torch==2.14.0"
.venv/bin/python -m pip install -q "numpy==2.5.3" "mujoco==3.14.0" "gymnasium[mujoco]==1.3.0" \
    "stable-baselines3==2.9.0" "matplotlib==3.11.2" "tqdm==4.70.1" "rich==15.0.0" "pandas==3.0.6" \
    "psutil==7.2.2" pytest
.venv/bin/python -c 'import mujoco, gymnasium, torch; print(f"ok: mujoco {mujoco.__version__}, gymnasium {gymnasium.__version__}, torch {torch.__version__}")'
