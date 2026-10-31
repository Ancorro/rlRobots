#!/bin/bash
# Completion check: the SO-101 arm stack moves objects in MuJoCo, CPU only, no models, no network after setup.sh.
# Runs the rule decider on the four arm tasks (reach, pick, sort, tabletop) and the sort task with 30 % gripper slips,
# on held-out seeds, and requires every episode to succeed. Prints one line per task and exits non-zero on any miss.
#   bash scripts/check_arm.sh [EPISODES=5]
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
py=${PYTHON:-$repo/.venv/bin/python}
[ -x "$py" ] || py=$repo/.venv/Scripts/python.exe  # Windows venv layout
[ -x "$py" ] || py=python3
n=${1:-5}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
export PYTHONPATH=$repo MPLBACKEND=Agg
fail=0
for task in reach pick sort tabletop sort-drop; do
    "$py" "$repo/decide.py" rule "$n" --task=$task --fast --no-gui > "$task.out" 2>&1 || { echo "$task: crashed"; tail -3 "$task.out"; fail=1; continue; }
    "$py" - "$task" "$n" <<'EOF' || fail=1
import json, sys
task, n = sys.argv[1], int(sys.argv[2])
s = json.load(open(f"runs/decide/{task}/rule/summary.json"))
ok = s["episodes"] == n and s["success_rate"] == 1.0
extra = f", skills {s['skill_success_rate']:.0%}" if "skill_success_rate" in s else ""
print(f"{task:10s} {s['episodes']} episodes, success {s['success_rate']:.0%}{extra}  {'ok' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
EOF
done
[ $fail = 0 ] && echo "ARM CHECK PASSED" || { echo "ARM CHECK FAILED"; exit 1; }
