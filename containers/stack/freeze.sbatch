#!/bin/bash
#SBATCH --job-name=stack-freeze
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --output=/nfs/hpc/share/%u/rlRobots/runs/slurm/freeze-%j.log
# What the stack runs on today: packages in .venv-vla, .venv-dam and the vLLM container, the container's base image,
# and whether this cluster lets us build images without root. Inputs for containers/stack/stack.def.
SHARE=${SHARE:-/nfs/hpc/share/$USER}  # your cluster share folder
S=$SHARE; O=$S/rlRobots/runs/freeze; mkdir -p $O
~/.local/bin/uv pip freeze -p $S/rlRobots/.venv-vla/bin/python > $O/vla.txt
~/.local/bin/uv pip freeze -p $S/rlRobots/.venv-dam/bin/python > $O/dam.txt
apptainer exec $S/llmserve/vllm-latest.sif python3 -m pip freeze > $O/vllm.txt 2>&1
apptainer exec $S/llmserve/vllm-latest.sif sh -c 'cat /etc/os-release | head -2; python3 --version; ls /usr/lib/x86_64-linux-gnu | grep -i "libegl\|libgl\.\|libosmesa" | head' > $O/vllm-os.txt 2>&1
apptainer inspect $S/llmserve/vllm-latest.sif > $O/vllm-inspect.txt 2>&1
ls -la $S/llmserve > $O/llmserve-ls.txt; cat $S/llmserve/*.sbatch > $O/llmserve-sbatch.txt 2>/dev/null
apptainer --version > $O/apptainer.txt; apptainer build --help 2>&1 | grep -i fakeroot >> $O/apptainer.txt
apptainer build --fakeroot /tmp/probe-$SLURM_JOB_ID.sif docker://busybox:1.36 >> $O/apptainer.txt 2>&1 && echo "FAKEROOT BUILD OK" >> $O/apptainer.txt
rm -f /tmp/probe-$SLURM_JOB_ID.sif
wc -l $O/*.txt
echo FREEZE-DONE
