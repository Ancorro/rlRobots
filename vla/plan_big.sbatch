#!/bin/bash -l
# Build the large held-out planner set (CPU), then submit every model run on it.
# Login computers are for copying and submitting only: anything longer runs as a job like this one.
# sbatch vla/plan_big.sbatch
#SBATCH --job-name=so101-plan-big
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=runs/slurm/%x-%j.log
set -euo pipefail
.venv-vla/bin/python plan_bench.py make --bench=big --seed=1 --scenes=700 --per-family=300 --max-start=3 --both-trays=0.3
.venv-vla/bin/python plan_bench.py baselines --bench=big
module load slurm
for t in off on; do
  sbatch vla/plan_ollama.sbatch --bench=horizon --model=gemma4:12b --tag=gemma4-12b-q4 --think=$t
  sbatch vla/plan_ollama.sbatch --model=gemma4:12b --tag=gemma4-12b-q4 --think=$t
  sbatch vla/plan_ollama.sbatch --bench=big --model=gemma4:12b --tag=gemma4-12b-q4 --think=$t
  sbatch vla/plan_ollama.sbatch --bench=big --model=gemma4:12b --tag=gemma4-12b-q4-schema --think=$t --schema
done
for m in qwen3.5-9b qwen3.5-4b gemma4-12b gemma4-e4b gpt-oss-20b cosmos-reason2-8b; do for t in on off; do
  sbatch vla/plan.sbatch --bench=big --model=$m --think=$t; done; done
for t in on off; do sbatch --constraint="h100|h200" vla/plan.sbatch --bench=big --model=qwen3.8-27b --think=$t; done
echo ALL SUBMITTED
