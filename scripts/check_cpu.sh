#!/usr/bin/env bash
# Smoke check of the CPU-only parts (about 30 s): unit tests, then each entry point on a tiny input.
# No GPU, display, network or model downloads. Outputs go to a temporary folder, not runs/.
#   bash scripts/check_cpu.sh        (after bash setup.sh)
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
py=${PYTHON:-$repo/.venv/bin/python}
[ -x "$py" ] || py=python3
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
export PYTHONPATH=$repo MPLBACKEND=Agg

step() { echo "== $*"; }

step "unit tests (tests/)"
"$py" -m pytest -q -p no:cacheprovider "$repo/tests"

step "decision tasks, rule decider, 2 episodes each (decide.py)"
for task in reach pick sort tabletop; do
    "$py" "$repo/decide.py" rule 2 --task=$task --fast --no-gui | tail -1
    "$py" -c "import json,sys; s=json.load(open('runs/decide/$task/rule/summary.json')); sys.exit(s['success_rate'] < 1)"
done

step "planner benchmark: make 2 scenes, oracle and random baselines (plan_bench.py)"
"$py" "$repo/plan_bench.py" make --scenes=2 --seed=0
"$py" "$repo/plan_bench.py" baselines | tee baselines.txt
grep -q "^oracle: 100.0%" baselines.txt

step "re-ID assignment against brute force (reid_bench.py)"
"$py" -c "import reid_bench; reid_bench.check_assignment()"

step "decision training data, 3 scenes (vla/decision_data.py)"
"$py" "$repo/vla/decision_data.py" --seed=800000 --scenes=3 --out=decide.jsonl --shuffle=1

step "camera images (needs an OpenGL library: OSMesa or EGL)"
if "$py" -c "from so101.pixel_pick import SO101PixelPick; SO101PixelPick().reset(seed=0)" 2>/dev/null; then
    echo "rendered"
else
    echo "not run: no OpenGL library here (apt install libosmesa6, then MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa)"
fi

echo "all CPU checks passed"
