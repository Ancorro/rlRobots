#!/bin/bash
#SBATCH --job-name=grasp-setup
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/nfs/hpc/share/%u/isaac/grasp-setup-%j.log
# MuJoCo for the G1 grasp RL (so101.g1 builds the hand model with it), installed beside the Isaac sandbox in $I/pylib,
# not into it: a pip install inside the sandbox once deleted Isaac Sim's bundled packaging/starlette files.
SHARE=${SHARE:-/nfs/hpc/share/$USER}  # your cluster share folder
set -uo pipefail
I=$SHARE/isaac
rm -rf $I/rlg1 && tar xzf $I/rlg1-code.tgz -C $I && echo "code: $(ls $I/rlg1)"  # isaac/ + so101/ (with fetched MuJoCo assets) from the AWS box
export APPTAINERENV_PYTHONNOUSERSITE=1 APPTAINERENV_PIP_CACHE_DIR=$SHARE/pip_cache
apptainer exec --cleanenv --writable --home $I/../isaac-home --bind $SHARE $I/isaac-sb bash -c "
set -e
/isaac-sim/python.sh -m pip install -q --target $I/pylib mujoco==3.14.0
cd $I/pylib && /isaac-sim/python.sh - <<'PY'
# drop what the sandbox already has (numpy, typing_extensions, ...): pylib goes first on PYTHONPATH and must not shadow it
import importlib.util, os, shutil, subprocess, sys
for name in sorted(os.listdir('.')):
    mod = name.split('-')[0].split('.')[0]
    if name.endswith(('.dist-info', '.libs')) or mod in ('mujoco', 'bin', '__pycache__'):
        continue
    r = subprocess.run(['/isaac-sim/python.sh', '-c', f'import {mod}'], cwd='/', capture_output=True)
    if r.returncode == 0:
        print('sandbox has', mod, '- removing'); shutil.rmtree(name) if os.path.isdir(name) else os.remove(name)
PY
cd / && PYTHONPATH=$I/pylib /isaac-sim/python.sh -c 'import mujoco, numpy; print(\"mujoco\", mujoco.__version__, \"numpy\", numpy.__version__)'
cd $I/rlg1 && PYTHONPATH=$I/pylib:$I/rlg1 /isaac-sim/python.sh -c 'from so101 import g1; from so101.objects import random_objects; print(\"so101 ok\")'
"
echo GRASP-SETUP-DONE
