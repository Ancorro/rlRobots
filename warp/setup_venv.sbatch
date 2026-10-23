#!/bin/bash
# Build the MuJoCo Warp venv (.venv-warp) in the cluster folder; tries the network first (compute node).
#SBATCH --job-name=warp-venv
#SBATCH --account=eecs
#SBATCH --partition=share,preempt
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=runs/slurm/%x-%j.log
SHARE=${SHARE:-/nfs/hpc/share/$USER}  # your cluster share folder
set -euo pipefail
export UV_CACHE_DIR=$SHARE/uv_cache UV_PYTHON_INSTALL_DIR=$SHARE/uv_python
UV=${UV:-$HOME/.local/bin/uv}
echo "uv=$UV host=$(hostname)"
curl -sS -m 10 -o /dev/null -w "pypi http %{http_code}\n" https://pypi.org/simple/ || echo "no network"
$UV venv -p 3.12 --python-preference only-managed .venv-warp
$UV pip install -p .venv-warp ${FIND:+--no-index --find-links=$FIND} mujoco==3.14.0 mujoco-warp==3.14.0 warp-lang==1.17.0 numpy gymnasium matplotlib
.venv-warp/bin/python -c "import mujoco, mujoco_warp, warp; print(mujoco.__version__, warp.__version__)"
