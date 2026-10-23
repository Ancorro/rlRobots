#!/bin/bash
#SBATCH --job-name=isaac-setup
#SBATCH --account=eecs
#SBATCH --partition=share,preempt
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/nfs/hpc/share/%u/isaac/setup-%j.log
# Isaac Lab on the OSU cluster (el9 nodes: Isaac Sim runs from its container, isaac-sim-5.1.0.sif, pulled by pull.sbatch).
# The image keeps /isaac-sim readable by root only (Docker runs it as root; apptainer runs it as us), so it is unpacked
# into a sandbox folder we own (isaac-sb), opened up, and Isaac Lab v2.3.2 core is pip-installed into that sandbox the
# way Isaac Lab's Docker image does (/isaac-sim/python.sh -m pip install -e). Jobs then run the sandbox read-only.
# Same fixes as setup_ubuntu.sh (RUNBOOK "Isaac Lab"): setuptools<81, flatdict without build isolation, no isaaclab_mimic,
# rsl-rl-lib==3.1.2. Also unpacks the G1-29dof-Dex3 fixed-base USD copied from unitree_sim_isaaclab (g1dex3.tgz).
SHARE=${SHARE:-/nfs/hpc/share/$USER}  # your cluster share folder
set -uo pipefail
I=$SHARE/isaac
cd $I
[ -d IsaacLab ] || git clone -q --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git || exit 1
mkdir -p unitree && tar xzf g1dex3.tgz -C unitree || echo "FAILED unpacking g1dex3.tgz"
export APPTAINER_TMPDIR=${TMPDIR:-/tmp} APPTAINER_CACHEDIR=$I/cache
if [ ! -d isaac-sb/isaac-sim ]; then
  apptainer build --sandbox isaac-sb isaac-sim-5.1.0.sif || exit 1
fi
chmod -R u+rwX isaac-sb/isaac-sim isaac-sb/root 2>/dev/null
mkdir -p isaac-sb$SHARE isaac-sb/nfs/hpc/apps  # mount points: --writable cannot create them
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
# --cleanenv: the login profile's conda env leaked in and Isaac's python.sh warned; PYTHONNOUSERSITE keeps pip out of
# $HOME/.local (a mixed pip there broke with "No module named pip._vendor.packaging._structures", 2026-10-07).
export APPTAINERENV_PYTHONNOUSERSITE=1 APPTAINERENV_OMNI_KIT_ACCEPT_EULA=YES APPTAINERENV_ACCEPT_EULA=Y APPTAINERENV_PRIVACY_CONSENT=Y
apptainer exec --cleanenv --writable --home $I/../isaac-home --bind $SHARE isaac-sb bash -c '
  set -x
  PY=/isaac-sim/python.sh
  SP=/isaac-sim/kit/python/lib/python3.11/site-packages  # its pip lost pip/_vendor/packaging/_structures.py: reinstall
  $PY -m pip --version >/dev/null 2>&1 || { rm -rf $SP/pip $SP/pip-*.dist-info; $PY -m ensurepip; }
  $PY -m pip --version
  cd '$I'/IsaacLab
  $PY -m pip install -q "setuptools<81" wheel
  $PY -m pip install -q --no-build-isolation flatdict==4.0.1
  for s in isaaclab isaaclab_assets isaaclab_tasks isaaclab_rl; do $PY -m pip install -q --no-build-isolation -e source/$s || echo FAILED-$s; done
  $PY -m pip install -q rsl-rl-lib==3.1.2
  $PY -c "import importlib.metadata as m; print(\"isaaclab\", m.version(\"isaaclab\"), \"rsl-rl\", m.version(\"rsl-rl-lib\"))"
'
echo SETUP-DONE
