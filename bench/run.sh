#!/bin/bash
# The walk_carry goal on Slurm, days 1 to 3 (bench/README.md), as one driver. Run it on the cluster login node from this
# repository checkout (module load slurm first if sbatch is not on PATH). Every step is idempotent; job ids go to
# bench/jobs.tsv and results to bench/results/, where scripts/check_day*.py read them.
#
#   bench/run.sh sync                     copy the code where walk_cluster.sbatch expects it ($SHARE/isaac/$CODE)
#   bench/run.sh day1                     the scripted baseline on validation (6 shapes x N)
#   bench/run.sh zero                     wiring check: the learned-grab code path with a zero policy (must score like scripted)
#   bench/run.sh v8 POLICY.pt [NAME]      the learned grab (g1_grasp_rl.py policy file) on validation, with --noise 0.4.
#                                         STAND=0.11 instead stops at the fixed-base distance (grab name NAME-stand0.11).
#                                         A different --stand changes the benchmark: compare it only with a scripted run at
#                                         the same value (WALKARGS="--stand 0.11" bench/run.sh day1), and settle on one
#                                         value before `select` (the checks pair against the latest scripted val record)
#   bench/run.sh collect                  job states, exit codes; BENCH_RESULT lines -> bench/results/; check_day1
#   bench/run.sh select                   scripts/select_grab.py --write, then check_day2
#   bench/run.sh day3                     test, once: the selected grab (bench/selection.json; POLICY=file stages it) and
#                                         scripted on --split=test; then `collect` and scripts/check_day3.py
#   bench/run.sh status                   squeue for the jobs in bench/jobs.tsv
#   bench/run.sh archive                  move bench/results/*.json and jobs.tsv to results/old-<date>/ (after a script
#                                         change: scripted and learned records must come from the same set-down script)
#
# Settings (environment): SHARE (default /nfs/hpc/share/$USER), CODE (folder under $SHARE/isaac, default rlRobots),
# N (layouts per shape, 16), WALKARGS (extra walk_grasp.py args), DRY=1 (print the sbatch commands, submit nothing).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
SHARE=${SHARE:-/nfs/hpc/share/$USER}
I=$SHARE/isaac
CODE=${CODE:-rlRobots}
N=${N:-16}
SHAPES="box,cylinder,lying-cylinder,ball,capsule,egg"
JOBS=$HERE/jobs.tsv
RESULTS=$HERE/results
WALK_BASE="--carry --rand_layout --shape=$SHAPES --n=$N --split=val ${WALKARGS:-}"
mkdir -p "$RESULTS"
PY=$REPO/.venv/bin/python; [ -x "$PY" ] || PY=python3  # for the CPU checks (standard library only)

say() { echo "[bench] $*"; }

submit() {  # submit NAME ARGS: one walk_grasp.py job (isaac/walk_cluster.sbatch); records NAME, job id, log path
    local name=$1 args=$2
    local cmd=(env SHARE="$SHARE" CODE="$CODE" SCRIPT=isaac/walk_grasp.py ARGS="$args" sbatch "$REPO/isaac/walk_cluster.sbatch")
    if [ "${DRY:-}" = 1 ]; then
        say "DRY: ${cmd[*]}"
        return
    fi
    local out jid
    out=$(cd "$REPO" && "${cmd[@]}") || { say "sbatch failed for $name: $out"; return 1; }
    jid=$(echo "$out" | grep -oE '[0-9]+$' | tail -1)
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date +%FT%T)" "$name" walk_carry "$jid" "$I/walk-$jid.log" >> "$JOBS"
    say "$name -> job $jid ($I/walk-$jid.log)"
}

staged() {  # staged FILE NAME: copy a policy file under $SHARE (the container binds $SHARE only); prints the path the job uses
    local dst=$I/$CODE/bench/policies/$2.pt
    mkdir -p "$(dirname "$dst")"
    cp "$1" "$dst"
    echo "$dst"
}

zero_policy() {  # a freshly initialised (zero-adjustment) policy file: the learned-grab path with no learning in it
    local out=$HERE/zero_policy.pt py
    [ -s "$out" ] && { echo "$out"; return; }
    for py in "$REPO/.venv/bin/python" "$REPO/.venv-vla/bin/python" python3; do
        "$py" -c "import torch" 2>/dev/null || continue
        "$py" - "$out" <<EOF
import sys, torch
sys.path.insert(0, "$REPO/isaac")
from grab_policy import Policy
torch.save(Policy(14, 8).state_dict(), sys.argv[1])
EOF
        echo "$out"
        return
    done
    say "no python with torch found to write the zero policy (tried .venv, .venv-vla, python3)"
    return 1
}

case ${1:-} in
sync)
    say "code -> $I/$CODE (walk_cluster.sbatch runs isaac/walk_grasp.py there)"
    mkdir -p "$I/$CODE" "$RESULTS"
    rsync -a --delete --exclude .venv --exclude '.venv-*' --exclude runs --exclude .git --exclude '__pycache__' "$REPO/" "$I/$CODE/"
    ;;
day1)
    submit day1-walk-scripted "$WALK_BASE"
    ;;
zero)
    P=$(staged "$(zero_policy)" zero)
    submit day2-walk-zero "$WALK_BASE --grab zero --policy $P"
    ;;
v8)
    [ -s "${2:-}" ] || { say "usage: bench/run.sh v8 POLICY.pt [NAME]; policy file not found: ${2:-}"; exit 2; }
    NAME=${3:-v8}${STAND:+-stand$STAND}
    P=$(staged "$2" "$NAME")
    submit "day2-walk-$NAME" "$WALK_BASE --grab $NAME --policy $P --noise 0.4${STAND:+ --stand $STAND}"
    ;;
day3)
    SEL=$HERE/selection.json
    [ -s "$SEL" ] || { say "no bench/selection.json: run day 2 (select) first"; exit 2; }
    WNAME=$("$PY" -c "import json; print(json.load(open('$SEL')).get('walk_carry', {}).get('policy', ''))")
    TEST="--carry --rand_layout --shape=$SHAPES --n=$N --split=test ${WALKARGS:-}"
    submit day3-walk-scripted "$TEST"
    if [ -n "$WNAME" ]; then
        P=$I/$CODE/bench/policies/$WNAME.pt
        [ -n "${POLICY:-}" ] && P=$(staged "$POLICY" "$WNAME")  # POLICY=file: stage the selected policy file now
        [ -s "$P" ] || { say "staged policy $P missing: POLICY=/path/to/policy.pt bench/run.sh day3"; exit 2; }
        SHA=$(sha256sum "$P" | cut -c1-64); WANT=$("$PY" -c "import json; print(json.load(open('$SEL'))['walk_carry']['checkpoint'])")
        [ "$SHA" = "$WANT" ] || { say "policy file sha256 $SHA is not the selected checkpoint $WANT"; exit 2; }
        submit "day3-walk-$WNAME" "$TEST --grab $WNAME --policy $P --noise 0.4"
    else
        say "no learned grab selected; scripted only"
    fi
    say "when done: bench/run.sh collect; $PY scripts/check_day3.py"
    ;;
archive)
    d=$RESULTS/old-$(date +%Y%m%d-%H%M)
    mkdir -p "$d"
    mv "$RESULTS"/*.json "$d"/ 2>/dev/null || true
    [ -f "$JOBS" ] && mv "$JOBS" "$d"/jobs.tsv
    say "moved old records to $d (the checks read only bench/results/*.json and logs at its top level)"
    ;;
status)
    [ -s "$JOBS" ] || { say "no jobs submitted yet"; exit 0; }
    squeue -u "$USER" -o "%.10i %.12j %.8T %.10M %R" 2>/dev/null || true
    column -t "$JOBS"
    ;;
collect)
    [ -s "$JOBS" ] || { say "no jobs in $JOBS"; exit 1; }
    while IFS=$'\t' read -r when name bench jid log; do
        [ "$jid" = job ] && continue  # a header line
        state=$(sacct -j "$jid" -n -o State,ExitCode -X 2>/dev/null | head -1 | tr -s ' ' || true)
        for cand in "$log" "$I/$log" "$HERE/logs/$log" "$HERE/logs/$(basename "$log")"; do  # absolute, or copied under bench/logs/
            [ -f "$cand" ] && { log=$cand; break; }
        done
        if [ ! -f "$log" ]; then
            say "$name job $jid: no log yet ($state)"
            continue
        fi
        rc=$(grep -aoE 'WALK-DONE.*rc=[0-9]+' "$log" | tail -1 | grep -oE '[0-9]+$' || true)
        if [ -n "$rc" ] && [ "$rc" != 0 ]; then
            say "$name job $jid: FAILED rc=$rc ($state); first error:"
            grep -aE '^(Traceback|[A-Za-z]*Error|RuntimeError)' "$log" | head -3 | sed 's/^/    /'
            continue
        fi
        if grep -q '^BENCH_RESULT ' "$log"; then
            grep '^BENCH_RESULT ' "$log" | tail -1 | sed 's/^BENCH_RESULT //' > "$RESULTS/$name-$jid.json"
            say "$name job $jid: record -> results/$name-$jid.json ($state)"
        else
            say "$name job $jid: no BENCH_RESULT in $log ($state); last lines:"
            tail -5 "$log" | sed 's/^/    /'
        fi
    done < "$JOBS"
    "$PY" "$REPO/scripts/check_day1.py" || true
    ;;
select)
    "$PY" "$REPO/scripts/select_grab.py" --write
    "$PY" "$REPO/scripts/check_day2.py"
    ;;
*)
    sed -n '2,24p' "$0"
    exit 2
    ;;
esac
