# Auto RL pipeline: walking G1 that grasps, carries and sets down (2026-10-28)

One command on the cluster login node drives the whole thing; the GPU work is Slurm jobs. Budget: **10 GPU-hours**
(the sum of the jobs' time limits; `--budget_h`), about 4.5 hours wall. RTX 8000 / A40 / L40S (the Isaac container
needs RT-core GPUs, see RUNBOOK).

```bash
module load slurm
cd $SHARE/isaac/rlRobots                      # the checkout the jobs run (bench/run.sh sync puts it there)
python isaac/auto_rl.py plan                  # the jobs, GPU-hours, whether the pool and the warm start are found
nohup python isaac/auto_rl.py run > runs/auto_rl/run.log 2>&1 &     # or in tmux; Ctrl-C and rerun resumes
python isaac/auto_rl.py status
```

Settings worth knowing: `--budget_h 10`, `--gpus 2` (parallel training runs; their hours are cut to fit the budget),
`--train_hours 3`, `--patience 150` (iterations without a better success window before a run stops), `--init` (the
warm-start checkpoint; default `$SHARE/isaac/rlg1g/runs/isaac/dex-v6b/model_4999.pt`), `--unitree`, `--pool` (skipped
when the folder is missing), `--partition`, `--constraint`.

Where this starts from (measured): dex-v6b on the walking base with the waist frozen and `--ema 0.4` (A40, 72 val
layouts, 2026-10-28): pelvis kick 11 deg median (22 without), closest pinch-to-object 0.139 m, lifted 0/72. The
fine-tune has to close that 14 cm; the pipeline stops on its own if it does not.

## When it stops by itself

- A training run stops at its time limit, or earlier when its success reward's 50-iteration window has not beaten the
  best by 5 % for `--patience` iterations (plateau), or has fallen to half its early value (diverged). It prints
  `LEARN it=... success=...` every 10 iterations and one `TRAIN-STOP reason=...` line; the report shows both.
- After selection: if no checkpoint lifted a single validation object, the task stage is skipped and the report says
  so ("stopped early"). Nothing is scored that was not run.
- A job that fails twice stops the pipeline with its log tail (`--max_tries`).

## What it does

| stage | jobs | what | decides |
|---|---|---|---|
| knobs | 3 x 0.25 h | dex-v6b on the walking base with waist frozen / strong action filter / both + rate cap (probe 2026-10-27: the policy's own motion pitched the base 22 deg) | the setting with the smallest pelvis kick among those that get the hand within 8 cm |
| train | N x (budget-fitted, ~3 h) | `g1_dexsuite.py train --base=walking --init=dex-v6b --walk_dr=1`: PPO fine-tune on the standing/walking base; after 3.5 s of each episode the legs get random carry motions (back off, turn, walk) so the policy learns to hold the object while the base moves; primitives or the object pool; two smoothness weights | — |
| select | (N+1) x sets x 0.25 h | the last 2 checkpoints of each run, and the warm start as control, on validation layouts (primitives, pool) | the checkpoint with the most objects kept through the shake test |
| task val / test | sets x 0.4 h each | `g1_dexsuite.py task`: walk in 1 m, grasp, lift, back off 0.45 m, turn 90 deg, walk 0.9 m to table B holding the object, lower it, open, step back; judged by `bench_spec` (lifted, on B, upright, hand clear, standing) | test once |
| report | — | `runs/auto_rl/report.md`; the task records also land in `bench/results/auto-task-*.json` | — |

## What "done" looks like, and what it does not promise

The report gives: the chosen knobs, every checkpoint's validation score beside the warm start's, the selected
checkpoint, and the end-to-end success on validation and test for primitives (and pool objects). Compare the task
numbers with the scripted + v8 walk (`bench/results`); they are different scenes (fixed layout here), so read them as
"the learned grasp carried through a walk" rather than a like-for-like replacement. The task records are written with
layout "fixed" and policy "dex:...", so the goal checks (which look for random-layout records) ignore them.

Honest limits: the walk is scripted (AGILE + commands), not learned; the grasp policy is the learned part. Of the new
code, only the `knobs` eval has run on a GPU (2026-10-28, A40); training with `--walk_dr`, the warm start and the task
mode have not, so the first `run` is also their smoke test. If a stage's jobs fail, `status` shows which, the logs are
`$SHARE/isaac/auto-run-<job>.log`, and after a fix the failed jobs are retried (`--max_tries`) or can be removed from
`runs/auto_rl/state.json` (or the whole file) to retry.

## Design notes

- Warm start, not from scratch: `--init` loads weights and observation normalizers from the fixed-base policy with a
  fresh optimizer; `--walk_dr` gives 10 s episodes and wide object bounds so the base can walk away from the table.
- The base kick is fought with action-side knobs chosen by data (stage knobs), and with the smoothness penalties the
  training variants sweep; the selected policy is whichever keeps the most objects, not whichever trained longest.
- Time limits: a training job stops itself at its time limit (or earlier at a plateau) and saves on the main thread;
  checkpoints every 50 iterations besides; the select stage tolerates runs that produced nothing.
- The orchestrator is standard-library Python, idempotent per job name, and tested end to end on fake logs
  (`tests/test_auto_rl.py`).

## Testing without a GPU

- `tests/fake_isaac.py` stands in for Isaac Lab so that `isaac/walk_grasp.py` runs end to end on the CPU with N robots
  (every tensor op, every path: scripted, learned grab, primitives, scanned objects, the carry and levelled set-down,
  the record); `tests/test_walk_grasp_fake.py`.
- `isaac/task_logic.py` holds the tensor logic of `g1_dexsuite.py`'s walking-base additions (walk_dr schedule, the
  action filter, table B geometry, set-down goal, judging); `tests/test_task_logic.py` checks shapes and numbers with
  N environments. The script calls these functions, so what is tested is what runs.
- `tests/test_auto_rl.py` drives the orchestrator through every stage on fake job logs.
What is still only exercised on the GPU: the Isaac config classes (spawners, action terms, the Dexsuite env subclass)
and rsl_rl's runner calls. The `knobs` stage is deliberately the first and shortest job for that reason.
