#!/bin/bash
# Build the pooled object set (so101/assets/objects/pool_g1.py) on cluster CPUs.
#   CODE=rlg1h [LIMIT=20] [WORKERS=8] [DGN_MAX=1000] sbatch so101/assets/objects/pool_cluster.sbatch   (from $SHARE/isaac)
#SBATCH --job-name=obj-pool
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/nfs/hpc/share/%u/isaac/pool-%j.log
SHARE=${SHARE:-/nfs/hpc/share/$USER}  # your cluster share folder
set -euo pipefail
I=$SHARE/isaac; D=$SHARE/datasets; V=$SHARE/envs/pool-venv
export UV_CACHE_DIR=$SHARE/uv_cache
[ -x $V/bin/python ] || { ~/.local/bin/uv venv -q -p 3.11 $V && ~/.local/bin/uv pip install -q -p $V/bin/python numpy scipy trimesh networkx coacd; }
[ -d $D/DexGraspNet/meshdata ] || tar -xzf $D/DexGraspNet/meshdata.tar.gz -C $D/DexGraspNet
OUT=$D/pool_g1${LIMIT:+_check}
cd $I/${CODE:-rlg1h}/so101/assets/objects
# ShapeNetSem left out (2026-10-13): its CoACD step killed workers, and the check kept 0 one-hand objects of 20.
$V/bin/python pool_g1.py --dgn=$D/DexGraspNet/meshdata --gso=gso_g1/hulls.npz \
  --out=$OUT --workers=${WORKERS:-8} --limit=${LIMIT:-0} --dgn_max=${DGN_MAX:-1000}
echo POOL-DONE
