# The walk_carry goal: records, selection, checks

Goal (`scripts/goal_common.py`): beat the development score of the walking G1 on `walk_carry` (walk to a table, pinch
one of six primitive shapes, carry it to a second table, set it down; best 19/96 on 2026-10-22) on a benchmark fixed
first, with the learned grab chosen on validation and the test split scored once. The contract is `isaac/bench_spec.py`
(seed splits, success tests, the result record); the CPU checks are `scripts/check_day1.py`, `check_day2.py`,
`check_day3.py` (or `check_goal.py` for all three). They read this folder by default.

## Status (2026-10-29)

| day | state | numbers (validation, RTX 3090, 6 shapes x 32 layouts) |
|---|---|---|
| 1 benchmark fixed, scripted baseline | passed | scripted 92/192 |
| 2 learned grab chosen on validation | passed | v8 119/192 vs scripted 92/192, only-v8 41 / only-scripted 14, McNemar p = 0.0004; `selection.json` names v8 |
| 3 test, once | passed | test, 6 shapes x 16 layouts, cluster: v8 48/96 vs scripted 41/96, only-v8 18 / only-scripted 11, McNemar p = 0.265; above the development best 19/96 |

Noise floor, measured: the same scripted code on the same 96 layouts differs by 8-9 layouts between runs (cluster RTX
8000 vs RTX 3090, and zero-policy vs scripted on one machine): GPU PhysX is not deterministic. Records carry
`settings.host`; the checks pair a candidate with the scripted record from the same machine. Treat a margin under
~10 layouts at n = 96 as noise.

## What goes here

- `results/`: one file per run: the run's `BENCH_RESULT` line saved as `.json` (or the Slurm job log, which the cluster
  script copies the line into). The checks find records by their `bench`, `split`, `policy`, `layout`, `objects` fields.
  Old records go to `results/old-*/`, which the checks ignore.
- `selection.json` (day 2): `{"walk_carry": {"policy": "<name>", "checkpoint": "<sha256>", "candidates": [...]}}`,
  written by `scripts/select_grab.py --write`. Every candidate needs a val record; the chosen one must be the val best;
  its `checkpoint` must equal the val record's.
- `jobs.tsv`: the driver's job table.

## The driver: `bench/run.sh` (Slurm)

On the cluster login node, from this checkout (`module load slurm` if needed; `SHARE=` if your share is not
`/nfs/hpc/share/$USER`, `CODE=` if the checkout under `$SHARE/isaac` is not `rlRobots`):

```bash
bench/run.sh sync          # code -> $SHARE/isaac/rlRobots (what walk_cluster.sbatch runs)
bench/run.sh day1          # scripted baseline on val: 6 shapes x N (N=16 default; N=32 halves the uncertainty)
bench/run.sh zero          # wiring check: the learned-grab code path with a zero policy (must score like scripted)
bench/run.sh v8 <policy.pt> v8     # the learned grab on val (--noise 0.4); STAND=0.11 for the fixed-base distance
bench/run.sh status        # squeue + the job table
bench/run.sh collect       # when jobs finish: BENCH_RESULT lines -> results/, then check_day1
bench/run.sh select        # select_grab.py --write (wiring check reported, zero never selected), then check_day2
POLICY=<policy.pt> bench/run.sh day3    # test, once: scripted and the selected grab; then collect, check_day3
```

`DRY=1` prints the sbatch commands without submitting. After any change to `walk_grasp.py`'s script (the set-down is
part of the benchmark): `bench/run.sh archive`, then `day1` and `zero` again before any learned run.

Running the walk jobs elsewhere (the same code on a local GPU): the arguments are
`--carry --rand_layout --shape=box,cylinder,lying-cylinder,ball,capsule,egg --n=32 --split=val|test`
plus `--grab <name> --policy <file> --noise 0.4` for a learned grab; save each run's `BENCH_RESULT` line as
`results/<name>.json`.

## The learned grab (day 2)

`walk_grasp.py --grab <name> --policy <file>` applies `g1_grasp_rl.py`'s adjustment policy (v8 and kin;
`isaac/grab_policy.py` is the shared definition) to the scripted pinch from where each robot stops; `--noise 0.4` gives
it the perception error it trained with. Zero adjustment is exactly the scripted pinch, so `bench/run.sh zero` is the
wiring check: if it differs from scripted on more than the noise floor, the learned path is not reproducing the
scripted pinch and nothing downstream is comparable.

Known difference from the fixed-base training: the walk stops with the pelvis `--stand` (0.18 m) from the edge where
the fixed-base robot stood at 0.11 m, so the policy sees objects ~7 cm further out than in training. It transferred
anyway (above); `--stand 0.11` is the alternative if a future policy does not.

## Seeds

| split | range | walk_carry base |
|---|---|---|
| train | 800000-899999 | 860000 |
| val | 700000-799999 | 760000 |
| test | 900000-999999 | 960000 (the historic walk runs) |

Other records: `isaac/box_walk.py` (two-hand box) and `walk_grasp.py --objects=gso` (scanned household objects) print
records too (`bench` box_carry, `objects` gso); they are not part of this goal and the checks ignore them.
