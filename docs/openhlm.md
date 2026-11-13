# OpenHLM-data benchmark (2026-10-31)

Goal: `scripts/check_openhlm.py` passes: a sample of exactly 20 distinct episodes of the Hugging Face dataset
`OpenHLM/OpenHLM-data` scored at 50 % success or better. The data is five LeRobot-format sets of a real Unitree G1
humanoid under teleoperation (`g1_HLM-12_full_teleop`, `g1_HLM-12_humi`, `g1_HLM-12_stationary_teleop`,
`long_g1_5_fruits_full_teleop`, `long_g1_5_fruits_humi`: three recording modes, an "HLM-12" task set and a long
five-fruit task, going by the names; `openhlm/fetch.py` prints what the tasks are). Nothing in them is simulated and no simulator of their scenes
exists, so no policy (ours or anyone's) can be run closed loop on them; what can be scored is how well a policy predicts
what the teleoperator did next, on episodes it never saw. That is the benchmark here, turned into a per-episode
pass/fail. It is the offline proxy used for real-robot data (validation loss in LeRobot, action L1 in OpenVLA), stated
as a contract before any run.

Everything is in `openhlm/`: the contract (`spec.py`), the reader (`data.py`), policies (`policy.py`,
`act_policy.py`), the trainer (`train.py`), the pipeline (`pipeline.py`), the fetch and frame tools (`fetch.py`,
`frames.py`), a stand-in dataset (`fake_data.py`), the cluster job (`run.sbatch`) and the tests (`tests/test_openhlm.py`).

## The contract (`openhlm/spec.py`)

**Test.** At the start of every chunk the policy sees HISTORY_S = 1/3 s of `observation.state` (10 frames at 30 fps)
and must write the next HORIZON_S = 1 s of `action` (30 frames). Chunks tile the episode without overlap from the first
frame that has a full history; the last partial chunk is not scored. At every chunk start the recorded history is given
(teacher forcing), so the chunks are independent open-loop predictions, what an action-chunking policy (ACT, pi0,
SmolVLA) does between re-plans.

**Error.** Mean over chunks, steps and action dimensions of |predicted - recorded| / sigma_d, sigma_d the training
split's standard deviation of action dimension d. Dimensions that do not move (std under 1 % of the largest) are left
out, so a parked hand cannot dominate the number.

**Reference.** The hold baseline: every step of the chunk filled with the robot's current position, a_{t+k} = s_t for
each action dimension that is also a state dimension (matched by name in `meta/info.json`; the G1 recordings name their
joints), the last recorded action a_{t-1} for the rest. It is what a controller that commands nothing new would send.

**Success.** skill = 1 - error(policy) / error(hold); an episode succeeds when **skill >= 0.5**: the policy removes at
least half of the error of standing still, over the whole episode. The level was fixed on 2026-10-31 before any run and
is not tuned on validation; runs also report the rate at 0.25 and 0.75, the mean and median skill, a Wilson 95 %
interval, and the paired comparison (exact McNemar, as in `scripts/goal_common.py`) against the ridge and hold references
on the same episodes.

**Split.** Each episode's role comes from a hash of `<set>/<episode_index>` with a salt, so the assignment is fixed
before the download and does not depend on the order episodes are listed. Per set, in hash order: the first 4 episodes
are test (5 sets x 4 = 20, the check's sample), the next 15 % (at least 2) validation, the rest training. Validation
chooses between candidates; the test sample is meant to be scored once, and every test run is counted: the pipeline
moves an earlier record to `runs/openhlm/old/`, appends the run to `runs/openhlm/test-history.jsonl` and names it in
the report ("Test (run #k)"; `--refuse-retest` stops instead). With fewer sets the per-set test share grows so the
sample stays 20.

**Inputs the policy may use.** The state history, the set's task index (the instruction, known to any policy at run
time; a one-hot) and, with `--camera`, the current camera frame. It does not see past actions: in these recordings the
action is the teleoperator's command and leads the measured state, so a policy that saw a_{t-1} would get the first
part of every chunk for free and the hold-state reference would no longer be fair. Episodes shorter than one chunk
(history + horizon frames) are left out before the split; they cannot be scored by anyone.

## Policies (`openhlm/policy.py`)

| candidate | what | trained on |
|---|---|---|
| `hold` | the reference (skill 0 by definition) | nothing |
| `ridge` | ridge regression, normalised history + task one-hot -> residual chunk on the hold baseline (closed form) | CPU, seconds |
| `mlp` | 3 x 512 MLP (LayerNorm, GELU, dropout 0.1) on the normalised history + task one-hot, residual on hold, L1 loss on the metric's dimensions, AdamW, cosine decay over the budget, input noise 0.02, head zero-initialised (starts as hold) | one GPU, its budget share |
| `mlp-wide` | 4 x 1024, dropout 0.2, weight decay 1e-3, input noise 0.05 | same |
| `mlp-cam` | `mlp-wide` plus a small convolutional encoder (4 strided convs, GroupNorm, spatial softmax, random-shift augmentation) on the current frame of the camera(s), from the frame cache; batch 512; gets twice a state-only candidate's share of the budget | same; only when a frame cache exists |
| `ens` | the mean chunk of the learned candidates, scored on validation at selection time | nothing |
| `act` | ACT from our LeRobot stack (`openhlm/act_policy.py`: LeRobot 0.6.1's `ACTPolicy`, CVAE, ResNet18 per camera, small transformer), fed the history as its environment state, residual on hold | same; needs `lerobot` in the venv |

Vision is a property of a candidate (`vision` in `policy.CANDIDATES`): with `--camera KEY` only `mlp-cam` and `act`
see the frames, so a state-only candidate always competes and a camera can only help the selection.

Every candidate is scored the same way (`policy.score`), and every checkpoint carries the training statistics, the
hold map, names, fps, history and horizon, so a test run cannot use other statistics than the training split's.

## The pipeline (`openhlm/pipeline.py`) and its budget

One command: split, train every candidate on every set, select per set on validation, score the test sample once,
write the records and run the check. Learned candidates are spread round-robin over the GPUs (one worker per GPU, units
run one after another); ridge runs on the CPU beside them. Each unit gets an equal share of the training budget (the
wall budget minus a 3-minute reserve for the test and start-up); the unit stops at its deadline, at `--max-steps`, or
after 8 validation passes without improvement, and keeps the checkpoint with the most validation successes (ties: mean
skill). Selection per set uses the same key.

Default (`openhlm/run.sbatch`): 2 GPUs, 30-minute job, `--budget-min 26`, candidates `mlp,mlp-wide,ridge`, 5 sets:
10 learned units, 5 per GPU, about 4.3 minutes each (minus 20 s of start-up), ridge on the CPU; the test and the report
take seconds. Per-episode records go to `logs/openhlm/test-<tag>.jsonl` (the check's input), everything else to
`runs/openhlm/<tag>/`: `split.json`, `plan.json`, `<set>/<candidate>.{pt,json,-val.jsonl}`, `logs/`, `selection.json`,
`test-refs.jsonl` (hold and ridge on the test episodes), `report.md`.

## Running it

On the cluster login node (network), once:

```bash
cd $SHARE/rlRobots                                   # the checkout the job runs (openhlm/run.sbatch cds to its own repo)
.venv-vla/bin/pip install -r openhlm/requirements.txt         # pyarrow (and huggingface_hub, av) next to the CUDA torch
# (installing torch itself needs a few GB of temporary space: TMPDIR=/somewhere/big pip install torch ...)
export HF_HOME=$SHARE/hf_cache
.venv-vla/bin/python -m openhlm.fetch                # tables + head-camera videos of the five sets; the table cache; the frame cache
mkdir -p runs/slurm                                  # the job log goes there (#SBATCH --output)
```

`fetch` prints, per set, the episodes, frames, fps, the resolved state / action keys and their dimensions, cameras,
tasks, whether the tables load and the inferred hold map, then the split with the 20 test episode ids, then builds
`runs/openhlm/cache/<set>-<id>.npz` (the tables, read once) and `runs/openhlm/frames/<set>/head_image_left/`
(every 3rd frame at 96 px, uint8: about 16 GB for the five sets; it drops to 64 px by itself when the disk is short;
decoding 600k PNG rows takes about 10 minutes on 8 CPUs). `--videos ""` skips the frames (the job then runs state
only); `--videos head_image_left,left_wrist_image` adds a camera. In OpenHLM-data the frames are PNG bytes inside the
parquet rows, so nothing beyond the tables is downloaded. PROBLEM lines and a non-zero exit mean
something to understand before submitting.

Then the job (two A40s, 30 minutes):

```bash
FAKE=1 sbatch openhlm/run.sbatch          # 8-minute dry run on generated data (with a generated camera): GPUs, venv, code path
sbatch openhlm/run.sbatch                 # the real run: train, select, test (counted), check; TAG=<name> to name it
cat runs/openhlm/<tag>/report.md; python scripts/check_openhlm.py logs/openhlm
```

The job detects the frame cache (`CAMERA=auto`): with it the candidates are `mlp-wide`, `mlp-cam`, `ridge` (and the
ensemble), without it `mlp`, `mlp-wide`, `ridge`. `CAMERA=""` forces state only, `CAMERA=<key>` another camera,
`CANDIDATES=...` any list (`act` needs `lerobot` in the venv: ACT's ImageNet ResNet weights come from torchvision on
first use, so run `python -c "import torchvision; torchvision.models.resnet18(weights='IMAGENET1K_V1')"` on the login
node with `TORCH_HOME=$SHARE/torch_cache` first, or it trains the backbone from scratch and says so). `VAL_ONLY=1`
trains and selects without a test; `SETS=a,b` limits the sets; `BUDGET_MIN` (26) is the pipeline's wall budget inside
the job's 30 minutes; `REFUSE_RETEST=1` stops when a test record exists.

Without Slurm, the same script runs on any machine with the venv and the data: `bash openhlm/run.sbatch`
(`PY=` the python, `DATA_DIR=` a folder with `<set>/meta/info.json` instead of the Hugging Face cache).

## Protocol: development on validation, counted tests

Anything compared or tuned (state only vs a camera, ACT in or out, other candidates, other budgets) is compared on
validation with `VAL_ONLY=1` runs (their `report.md` has the validation table and no test section), then the chosen
configuration runs with the test. The default job is that run for the default configuration (state only,
`mlp,mlp-wide,ridge` and their ensemble, selection per set on validation). The test sample never changes, so every
test run is one more look at it: the pipeline counts them (`runs/openhlm/test-history.jsonl`, "run #k" in the report,
the earlier record kept in `runs/openhlm/old/`), and a write-up must give the number of test runs next to the number.

## What a run reports

`runs/openhlm/<tag>/report.md`: the split sizes; per set every candidate's validation successes, mean skill, steps and
seconds, the selected one marked; the test: successes of 20 with the Wilson interval, mean and median skill, the
count at each skill level, per set, the ridge and hold references on the same episodes with the paired counts and
McNemar p, and every test episode's numbers. The job log ends with the check's output and `OPENHLM-DONE rc=`.

The check (`scripts/check_openhlm.py`) reads `logs/openhlm/`: every record names an OpenHLM set, each episode once,
exactly 20 distinct episodes, a success value each, and at least 50 % succeed. Online it also checks the episode ids
against each set's `meta/episodes.jsonl`; `--offline` skips that (the pipeline runs it offline on the compute node).

## The data as found (fetch of 2026-11-02, snapshot fe7f131)

| set | episodes | frames | episode length | tasks |
|---|---|---|---|---|
| g1_HLM-12_full_teleop | 480 | 522,181 | 441-2,221 | 12 ("Walk forward to the table, put the red can on the orange mouse pad with the right hand, turn around and walk back." ...) |
| g1_HLM-12_humi | 480 | 410,925 | 173-2,221 | 12 (the same) |
| g1_HLM-12_stationary_teleop | 480 | 464,722 | 441-2,221 | 12 (the same) |
| long_g1_5_fruits_full_teleop | 120 | 294,427 | 1,746-3,362 | 20 ("Pick the banana with the right hand, pick the peach with the left hand, put them on the shelf." ...) |
| long_g1_5_fruits_humi | 199 | 122,459 | 133-3,322 | 20 (the same) |

All 30 fps, LeRobot v2.1, `robot_type unitree_g1`. The columns are **`state` and `actions`** (not LeRobot's
`observation.state` / `action`), the cameras **`head_image_left`, `left_wrist_image`, `right_wrist_image`** (not
`observation.images.*`). That broke the first run (`KeyError: 'Field "observation.state" does not exist in schema'`,
no training ran); the reader now resolves the keys (`spec.STATE_KEYS` / `ACTION_KEYS`, or `--state-key` /
`--action-key`) and takes cameras by their own keys. The split (hash, fixed): test episodes full_teleop
[119, 120, 267, 300], humi [62, 164, 235, 353], stationary [59, 121, 137, 180], long full_teleop [24, 27, 71, 104],
long humi [22, 76, 85, 108]; validation 71 / 71 / 71 / 17 / 29 episodes.

Whether the joints are named, and the state and action dimensions, were not in the first run's log (the describe
stopped at the missing keys). Without names the hold map is **inferred from the training episodes**
(`data.infer_hold_map`): an action dimension is held from the state dimension that tracks it, at the best lag up to
0.5 s, within 0.25 of its std (median absolute difference); unmatched dimensions (commands nothing in the state tracks,
a base velocity say) are held at the last recorded command by the reference and the policies' base alike. The map
and its source are in every checkpoint, the training log and the report; `fetch` prints it on 8 episodes.

**The camera frames are in the parquet rows** (2026-11-04, run #3): the three cameras are `image` features (PNG bytes
per row, Hugging Face datasets' Image column), not `video` mp4 files, so there is no `videos/` folder to download;
that is also why the "tables" took 40 minutes to fetch and why reading whole tables killed a 32 GB job (a set's
tables are tens of GB with the images). The reader now reads only the state, action and index columns; `frames.py`
decodes the `image` column in 64-row batches (PIL, resized) and still handles `video` features for other LeRobot sets.

Still assumed: frame i = row i for the cameras (a shorter video is padded with its last frame); the frame cache of a
set fits on the GPU (522k / 3 frames at 96 px = 4.8 GB; `--max-frames-gb 16` drops training episodes beyond that).

## Status

- 2026-10-31: contract, pipeline, tests and job written; the whole path (fake data -> frames -> train -> select -> test
  -> check) runs on the CPU in the tests (`tests/test_openhlm.py`, about a minute) and as `FAKE=1 bash openhlm/run.sbatch`.
- 2026-11-02: first run on the data (one Quadro RTX 8000): `fetch` downloaded the five sets' tables in 40 minutes and
  found the column names above; the reader refused them, so no training ran and no record was written. Fixed
  (key resolution, cameras by key, inferred hold map); the stand-in data now copies the real schema (`fake_data.py
  --schema openhlm`, the default), so the tests and `FAKE=1` exercise the same path.
- **2026-11-04, test run #1: 14/20 (70 %), the check passes.** One Quadro RTX 8000 (not the two A40s planned), 26-minute
  budget, state only, seed 0, candidates `mlp`, `mlp-wide`, `ridge`. Details in the result section below.
- 2026-11-04, run #3 (no test): the first vision attempt found no `videos/` folder (the frames are `image` columns in
  the parquet rows) and a 32 GB job was OOM-killed reading whole tables; no training ran. Fixed: column-selective
  reads, frames decoded from the parquet rows, `fetch` no longer fails on that, the job falls back to state only
  where frames are missing.
- **2026-11-04, test run #2 (head camera): 14/20 again, mean skill 0.540 (was 0.529).** One A40, 128 GB, 8 CPUs;
  `fetch` decoded the 605k head-camera rows in 137 s; candidates `mlp-wide`, `mlp-cam`, `ridge`, ensemble. (The
  harness ran in a fresh checkout, so its report says "run #1"; it is the second look at the test episodes.)
  Details below.
- **2026-11-04, test run #3 (head and both wrist cameras): 15/20, mean skill 0.539.** One A40, 26-minute budget, seed 0;
  candidates `mlp-wide`, `mlp-cam`, `ridge`, ensemble. The third look at the same 20 test episodes (its report also says
  "run #1": a fresh run directory again). Details below.

## Result of test run #2 (2026-11-04, head camera)

| set | selected | val: ens / mlp-cam / mlp-wide (successes, mean skill) | test |
|---|---|---|---|
| g1_HLM-12_full_teleop | ens | 62/71 0.551 / 59/71 0.542 / 56/71 0.537 | 3/4 |
| g1_HLM-12_humi | mlp-cam | 47/71 0.526 / 48/71 0.518 / 46/71 0.511 | 2/4 |
| g1_HLM-12_stationary_teleop | ens | 42/71 0.512 / 42/71 0.505 / 37/71 0.494 | 4/4 |
| long_g1_5_fruits_full_teleop | ens | 11/17 0.532 / 11/17 0.527 / 9/17 0.510 | 2/4 |
| long_g1_5_fruits_humi | ens | 19/29 0.550 / 19/29 0.543 / 19/29 0.542 | 3/4 |

Test: 14/20 (the same 14 episodes as run #1), mean skill 0.540 (run #1: 0.529), median 0.538; ridge 1/20, hold 0/20.

- The camera helps, a little, everywhere: `mlp-cam` beats `mlp-wide` on validation in all five sets (+0.001 to
  +0.017 mean skill) with 7.7-10k steps against 20-23k for the state-only MLP, and the ensemble of the two beats
  both in four sets. Pooled validation 182/259 = 70 % (run #1: 66 %).
- All six failing test episodes moved up (0.434 -> 0.445, 0.438 -> 0.449, 0.440 -> 0.463, 0.458 -> 0.478, 0.467 ->
  0.483, 0.469 -> 0.473) and none crossed 0.5: the gap to 15/20 or more is 0.02-0.05 of skill on those episodes.
- `mlp-cam` was still improving at the end of every unit (best step = last step; full_teleop 0.523 at 3k steps ->
  0.542 at 7.7k); on one GPU it had about 180 s per set. Training is the lever again.
- Decoding the PNG rows is cheap (137 s per camera on 8 CPUs), the frame cache 16 GB per camera at 96 px.

Changes after run #2: `fetch` decodes **all three cameras** (head and both wrists: the wrist views show the hand and
the object, which is what the next second of a grasp depends on) and the job gives `mlp-cam` every camera with a
complete cache (`CAMERA=auto`); `mlp-cam` trains under **bf16 autocast** with cuDNN autotuning (the convolutions were
the cost); the report has a **"where the error is"** table (error by chunk step and by action dimension, policy
against hold, on validation) and marks candidates that were still improving; the reserve drops to 2.5 min.

## Result of test run #3 (2026-11-04, head and both wrist cameras)

One A40, 128 GB, 8 CPUs, 26-minute budget, seed 0; `fetch` decoded each camera in 130-139 s; candidates `mlp-wide`,
`mlp-cam` (all three cameras), `ridge`, ensemble. Records: `openhlm/results/20261104-225700/`.

| set | selected | val: ens / mlp-wide / mlp-cam (successes, mean skill) | mlp-cam steps | test |
|---|---|---|---|---|
| g1_HLM-12_full_teleop | ens | 60/71 0.538 / 56/71 0.535 / 33/71 0.489 | 1,270 | 3/4 |
| g1_HLM-12_humi | mlp-wide | 43/71 0.504 / 46/71 0.511 / 23/71 0.432 | 684 | 3/4 |
| g1_HLM-12_stationary_teleop | ens | 41/71 0.511 / 38/71 0.494 / 33/71 0.484 | 2,295 | 4/4 |
| long_g1_5_fruits_full_teleop | ens | 12/17 0.534 / 9/17 0.510 / 12/17 0.523 | 7,737 | 2/4 |
| long_g1_5_fruits_humi | ens | 19/29 0.553 / 19/29 0.542 / 19/29 0.541 | 12,049 | 3/4 |

Test: **15/20 (75 %, Wilson 95 % 0.53-0.89), mean skill 0.539, median 0.537** (run #2: 14/20, 0.540); ridge 1/20, hold
0/20 on the same episodes.

- The one new success is humi 353 (0.513; runs #1 and #2: 0.467, 0.483), scored by the state-only `mlp-wide`; the
  other five failures stay below 0.5 (0.440-0.489). Mean skill did not move.
- Three cameras slowed `mlp-cam` to a sixth of run #2's steps on the HLM-12 sets, and it fell behind the state-only
  MLP on validation there (33/71 vs 59/71 with one camera on full_teleop). Pooled validation of the selected candidates:
  178/259 (run #2: 182/259).
- The test episodes have now been scored three times. What is left on this line is training time per camera model (a
  longer budget or two GPUs), compared on validation (`VAL_ONLY=1`) before any further test run.

## Result of test run #1 (2026-11-04)

| set | train / val | selected | val successes (mean skill) | test |
|---|---|---|---|---|
| g1_HLM-12_full_teleop | 405 / 71 | mlp-wide | 57/71 (0.536); mlp 54/71, ridge 0/71 (0.184) | 3/4 |
| g1_HLM-12_humi | 405 / 71 | mlp | 46/71 (0.507); mlp-wide 45/71 (0.516), ridge 0/71 | 2/4 |
| g1_HLM-12_stationary_teleop | 405 / 71 | mlp-wide | 39/71 (0.496); mlp 30/71, ridge 0/71 | 4/4 |
| long_g1_5_fruits_full_teleop | 99 / 17 | mlp-wide | 9/17 (0.508); mlp 9/17, ridge 0/17 | 2/4 |
| long_g1_5_fruits_humi | 166 / 29 | mlp-wide | 19/29 (0.539); mlp 17/29, ridge 1/29 | 3/4 |

Test: **14/20 succeed (70 %, Wilson 95 % 0.48-0.85), mean skill 0.529, median 0.523**; 20/20 at skill >= 0.25,
0/20 at >= 0.75. On the same 20 episodes ridge 1/20 (mean skill 0.233) and hold 0/20; the selected policy is better
than both on 20/20 episodes (McNemar p = 2e-4 and 1e-4). Pooled validation: 170/259 = 66 %, consistent with the test.

What the run showed, in order of importance:

1. **The pass is real but thin.** Every policy's skill sits in a narrow band around the pass level: test skills
   0.43-0.65, validation means 0.50-0.54 with standard deviations 0.05-0.13 per set. The 0.5 level is at the median of
   that band, so the success *rate* is a knife-edge statistic: the same policies with another seed could give 11/20 or
   16/20. The robust statements are the mean skill (~0.53: the policies remove half of the hold error) and the paired
   wins over ridge and hold (20/20). Raising the mean skill by 0.03 moves roughly 15-20 % of the episodes across the
   level; that is the lever.
2. **Half of every training unit was spent reading data.** The first validation pass came 63 s into each 120-130 s
   unit: 405 parquet files per set from network storage. In 4 of 5 sets the best checkpoint was the last one and the
   validation curve was still rising (full_teleop mlp-wide: 0.526 at 4.7k steps, 0.536 at 15k). Fixed: the pipeline
   now reads each set once (threaded) into one `.npz` cache (`runs/openhlm/cache/`), and the units start in about a
   second. On the two A40s the job was written for, that is roughly 4x the training of this run per unit.
3. **The state and the action are the same 34 joints with command-measurement offsets.** Means and stds line up
   dimension by dimension (`ridge.pt` statistics), but many dimensions carry a constant offset of 0.05-0.26 rad between
   the commanded and the measured position (gravity sag; the hands closing on objects: dims 15, 19, 25 have the largest),
   so the hold map matched only 6-10 of 34 dimensions by the contract's rule (median |a - s| within 0.25 std). The
   other dimensions were therefore held at the **last command** by the reference and by the policies' residual base
   alike. That is a stricter reference than "hold the current position" and the same for every candidate, so the
   comparison is fair and the contract was applied as written; but the contract's text ("the robot's current position")
   describes only a quarter of the dimensions in this data. The test records now carry each set's hold map and source.
   Not changed for now (changing the reference after the test would be a new benchmark version: "hold the last command
   for every dimension" would be the simpler, stricter v2; see the options below).
4. **Nonlinearity matters**: ridge reaches skill 0.18-0.26, the MLPs 0.50-0.54; the wider MLP wins 4 of 5 sets.
   The task index, the 1/3 s of history and a 1 s horizon are enough for half the hold error; what is left is mostly
   what the state cannot know (when the operator starts the next move, where the object is), which is what a camera
   would add.
5. The dimensions with near-zero motion in the HLM-12 sets (dim 7: std 0.0; dims 29, 31: 0.025) are left out of the
   error by the 1 % rule in two sets and kept in others, as designed.

Changes after run #1 (no change to the contract): the table cache (2); the **ensemble** of the learned candidates
(mean chunk) is scored on validation and competes in the selection (free; usually a small gain); a second test run is
no longer refused but **counted**: the old record goes to `runs/openhlm/old/`, every test run is appended to
`runs/openhlm/test-history.jsonl` and listed in the report ("Test (run #k)"); `REFUSE_RETEST=1` keeps the old
behaviour. Validation-only runs (`VAL_ONLY=1`) remain the way to compare configurations without spending a test.

## Options next, ranked

| option | expected effect on mean skill | cost | risk to the protocol |
|---|---|---|---|
| A. Rerun as is on the two A40s with the cache (about 4x the training per unit; curves were still rising) and the ensemble | +0.01-0.03 (enough to move several episodes) | one 30-minute job | none: same contract, test run #2 is counted |
| B. Vision (**done; run #2 measured it**): `fetch` decodes every 3rd frame of the three cameras at 96 px from the PNG rows; `mlp-cam` (the wide MLP plus a small CNN per camera) competes with `mlp-wide` on validation with twice its budget share; ACT (`CANDIDATES=...,act`) uses its ResNet | +0.01 mean skill with the head camera at 180 s of training per set; wrist cameras and more steps next | 137 s of decoding per camera on 8 CPUs, 16 GB of disk per camera | none: a state-only candidate still competes on validation; the test is counted |
| C. Contract v2 (only if wanted): hold the last command for every dimension (no inference), and the report's `skill` as the headline with the rate as secondary | makes the reference uniform and strictly harder than v1 for the 6-10 matched dims | a docs change and a new salt/version tag; v1 results stay as recorded | a new benchmark version: v1 and v2 numbers are not comparable |
| D. Model variants on validation: 1 s of history (a contract constant today), a smooth output basis (DCT coefficients instead of 30 raw steps), two seeds | +0.00-0.02 each | cheap, selection on validation | history is a contract constant; changing it is v2 |

Recommended order: A and B are now one job (the default run trains `mlp-wide`, `mlp-cam`, `ridge` and their
ensemble and selects per set on validation); then C only if a uniform reference is wanted.
