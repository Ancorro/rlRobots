#!/bin/bash
# The whole MJWarp feasibility study on the OSU cluster: run from $SHARE/rlRobots-warp on a login node
# (only submits jobs). N = layouts per object class (training seeds 800000+).
set -euo pipefail
module load slurm
N=${N:-100}
T=runs/warp/tapes-n$N.pkl
mkdir -p runs/slurm runs/warp warp_cache
A=$(sbatch --parsable warp/cpu.sbatch "tapes+--n=$N+--procs=16")
echo "tapes $A"
sbatch --dependency=afterok:$A warp/cpu.sbatch \
  "cpu+--tapes=$T+--scene=orig+--noslip=3+--procs=16" "cpu+--tapes=$T+--scene=orig+--noslip=0+--procs=16" \
  "cpu+--tapes=$T+--scene=simple+--noslip=3+--procs=16" "cpu+--tapes=$T+--scene=simple+--noslip=0+--procs=16" \
  "cpu+--tapes=$T+--scene=simple+--noslip=0+--friction=elliptic+--procs=16"
sbatch --dependency=afterok:$A warp/gpu.sbatch "speed+--tapes=$T+--cpu" "gpu+--tapes=$T" "gpu+--tapes=$T+--friction=elliptic" \
  "speed+--tapes=$T+--worlds=1,64,512,1024,4096,8192" "speed+--tapes=$T+--worlds=1,64,512,1024,4096,8192+--friction=elliptic"
sbatch --dependency=afterok:$A --time=03:00:00 warp/gpu.sbatch "gpu+--tapes=$T+--scene=orig" \
  "speed+--tapes=$T+--scene=orig+--worlds=1,64,512+--shape=box"
sbatch --dependency=afterok:$A --partition=dgxh --constraint=h100 warp/gpu.sbatch "speed+--tapes=$T+--worlds=1,64,512,1024,4096,8192"
sbatch --dependency=afterok:$A --partition=gpu,preempt --constraint=rtx8000 warp/gpu.sbatch \
  "speed+--tapes=$T+--worlds=1,64,512,4096+--shape=box,ball"
