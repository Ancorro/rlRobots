#!/bin/bash
# Run a CPU-only Python script with the Isaac container's python (h5py, torch, robomimic in pylib-mimic).
#   sbatch isaac/py_cluster.sbatch $SHARE/isaac/h5_tree.py sdg/nav_judge10.hdf5
#SBATCH --job-name=isaac-py
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/nfs/hpc/share/%u/isaac/py-%j.log
SHARE=${SHARE:-/nfs/hpc/share/$USER}  # your cluster share folder
I=$SHARE/isaac
K=/isaac-sim/extscache/omni.kit.pip_archive-0.0.0+69cbf6ad.lx64.cp311/pip_prebundle; ML=/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle  # numpy, torch
export APPTAINERENV_PYTHONNOUSERSITE=1 APPTAINERENV_PYTHONPATH=$I/pylib-mimic:$I/IsaacLab/source/isaaclab_mimic:$I/pylib:$K:$ML
cd $I
apptainer exec --cleanenv --home $I/../isaac-home --bind $SHARE $I/isaac-sb /isaac-sim/kit/python/bin/python3 "$@"
rc=$?  # the job ends with the script's exit code (audit 2026-10-23)
exit $rc
