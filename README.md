# rlRobots: Simulated Robot Arm and Humanoid Pick, Carry and Place

This research repository started as a toy reinforcement learning (RL) project in
MuJoCo (Multi-Joint dynamics with Contact), the physics simulator, using the
SO-101, a small open-source six-motor robot arm. It grew in two directions:

1. **A Unitree G1 humanoid with Dex3 three-finger hands** that walks to a table,
   picks an object up, carries it to a second table and sets it down. This part
   runs in NVIDIA Isaac Lab, a robot-learning framework built on the Isaac Sim
   simulator, with many robots simulated side by side on one graphics processing
   unit (GPU).
2. **A perception-and-decision stack for the SO-101**: cameras and off-the-shelf
   vision models find objects, a large language model (LLM) turns a free-text
   request into moves, a small decision model picks the next skill, and scripted
   skills move the arm.

Much of what was tried did not work, and the negative results are kept here next
to the positive ones. Everything is simulation only: no real robot has run this
code.

- **Author:** Steven Cleasby-Mayeda (GitHub: Ancorro), Oregon State University

## Headline Results

Every number below is taken from the results write-up (`docs/results.md`) with the date
of its entry. "Held-out" means seeds or layouts kept apart from development and
training. Rows marked "development" were measured on the same scenes that were
used to tune the method, so treat them as optimistic.

### Humanoid (Unitree G1 + Dex3, Isaac Lab)

| Result | Number | Date | Caveats |
|---|---|---|---|
| Walk, one-hand pinch, carry, set down: 6 shapes x 64 = 384 robots in one GPU job (5.2 min) | 144/384 set down upright with the robot standing (box 32, lying cylinder 27, ball 24, capsule 24, upright cylinder 21, egg 16, each of 64); a repeat with more angular damping gave 146/384 | 2026-10-22 | Development run. Scripted grasp, no learning. The "egg" is a capsule with an egg's length and width (no ellipsoid shape available). Round objects get artificial angular damping because the physics engine has no rolling friction. |
| Same pipeline, random layouts: each robot its own table distances, offsets and turn (6 shapes x 16) | 19/96 set down upright, 0 falls; the fixed layout re-checked at 27/96 | 2026-10-22 | Development run. Walking and turning hold up in new scenes; the loss is at the grab. |
| walk_carry as a fixed benchmark (random layouts, strict success test, seed splits): scripted walk vs the v8 grasp-parameter policy applied inside the walk, 6 shapes x 32 validation layouts | val: scripted 92/192, v8 119/192 (only-v8 41, only-scripted 14, McNemar p = 0.0004); test, once: scripted 41/96, v8 48/96 (p = 0.265) | 2026-10-29 | Selected on validation, scored once on test; both above the development best 19/96. Measured noise floor: 8-9 of 96 layouts flip between identical runs. |
| Closed-loop fixed-base grasp policy (dex-v6b) on the walking base, diagnosed | 0/72: its own arm motion pitches the base 22 deg within 0.1 s; waist frozen 11 deg, still 0/72 | 2026-10-28 | Dynamics, not perception (`isaac/obs_compare.py`). Fine-tuning pipeline (`isaac/auto_rl.py`): first stage run, training not run. |
| Same pipeline, earlier round, one object size per shape | 8/24 placed upright (box 6/12, cylinder 2/12) | 2026-10-15 | Development seeds. A spawn bug found 2026-10-22 meant all earlier walk-and-grasp runs used one object size per shape. |
| Two-hand box carry (box 0.22 x 0.14 x 0.20 m): scripted squeeze, walk to a second table, set down | 31/60 (held 36, fell 0, dropped 2) | 2026-10-22 | Development: the same 60 scenes were reused to compare script settings, and this is the best setting. Every scripted tweak landed near 30/60. |
| Two-hand box, learned grab by behaviour cloning (BC) instead of the script, same walk | 5/60 vs 18/60 for the script on the same scenes | 2026-10-22 | The learned grab hands over to the walk too rarely (37 of 60 never handed off). |
| Two-hand box, standing only: BC grab and carry, scripted set-down | 17/20 | 2026-10-21 | 20 rollouts; the BC alone reached 5/20 because it never let go. |
| Fixed-base G1 lift test, 360 held-out layouts of 6 primitive shapes: scripted pinch | 155/360 lifted | 2026-10-07 | Corrected from 225/360 after self-collision was turned on. A later fix to a reach motion that knocked about 10% of objects off the table raised the script from 161 to 206 in a re-run (logged between the 2026-10-07 and 2026-10-08 entries); all earlier numbers carried that loss. |
| Same test, scripted pinch + learned grasp adjustments (policy gradient, "v8") | 282/360 lifted (302 with 2 retries) | 2026-10-08 | Trained and tested with realistic settings: perception error, real G1 motor torque limits, a shake test. |
| Same test, closed-loop proximal policy optimization (PPO) policy driving waist, arm and hand joints ("dex-v6b") | 345/360 lifted and kept; v8 lifted 284, kept 281 on the same layouts | 2026-10-09 | The checkpoints were scored on the test layouts by mistake before selection; selection then used validation layouts only. On 342 held-out household objects it is worse than v8: kept 40 vs 58. |

### SO-101 arm and the decision stack (MuJoCo)

| Result | Number | Date | Caveats |
|---|---|---|---|
| PPO on `SO101Reach-v0` (move the gripper tip to a target) | success 0% to 100% | 2026-10-02 | Toy task, trained on a central processing unit (CPU). |
| Camera-only pick-and-place benchmark (detector + segmenter + depth, scripted grasp), held-out seed 8, 200 scenes x 2 tasks | 369/400 placed (random primitives), 341/400 (scanned household objects) | 2026-10-03 | Two simulated depth cameras with perfect calibration; realism tests on the same benchmark show calibration error is what hurts. |
| G1 (fixed base) learned grasp chooser, MuJoCo lift test, 360 held-out layouts | scripted 154, chooser v3 212, v4 222 | 2026-10-06 | In the camera-only benchmark the chooser's lead mostly vanished (primitives 53 vs 47 placed of 200, p = 0.38). |
| Planner LLM (Gemma 4 12B) on 3,600 held-out plan-only requests | 99.5% right with thinking (4-bit copy 99.4%) | 2026-10-06 | No physics: plans are checked against exact goals on scene text from the simulator. |
| Full stack v1 (perception, planner, decision model, scripted skills), pre-registered test, 100 requests | 62/100 done (75/100 with true object names) | 2026-10-09 | Failures: perception 21, decision 9, skill 8. |
| Stack v2 with a low-rank adaptation (LoRA) fine-tuned Gemma 4 12B as decision model, 60 development requests | 39/60 done; stack v1 on the same 60: 30/60 | 2026-10-21 | Development set only; the frozen final v2 test has not been run. |

## Negative Results

These are kept because they shaped the project, not because they are finished.

- **Box PPO never learned to put the box down** (2026-10-21). Three reward
  versions learned to grab, lift and carry the box over the goal, but placing
  stayed at 0%: the policies held the box over the goal until time-out.
- **The fixed-base grasp policy did not transfer to the walking robot**
  (2026-10-15, 2026-10-16). dex-v6b (345/360 standing at a fixed base) lifted
  0/72 on the walking G1, and two 100-iteration fine-tunes on the walking base
  still lifted 0/72.
- **Residual RL on the MuJoCo grasp chooser gave no gain** (2026-10-06): 211 vs
  212 of 360.
- **Vision-only SmolVLA, a 450M-parameter vision-language-action (VLA) model,
  picked 0/20** after fine-tuning on 200 scripted demos (2026-10-02): it learned
  the motion but not which cube to go to.
- **Porting Isaac Lab's Dexsuite lift task to the G1 stalled repeatedly**
  (runs dex-v1 to dex-v5, 2026-10-08 to 2026-10-09) before lifting appeared;
  fixes to resets, reward points, gravity and action smoothing, not more
  training, made lifting learnable.
- **Isaac Lab Mimic demo generation from the steering-wheel demos produced 0
  successful box demos** across five box and crate shapes (2026-10-21); the
  recorded rim grip does not carry over to a box.
- **Walking with the Dex3 model from scratch failed** (2026-10-07), and
  Unitree's released walking policy fell in 96-100% of runs zero-shot in this
  setup. The walking used here is NVIDIA's pretrained AGILE leg policy, used as
  is (Isaac Lab's published G1 walking checkpoints also walked, but on a
  different G1 model without Dex3 hands).
- **A navigation BC policy's early 4/20 was a scoring flaw** (2026-10-20): under
  a strict place check it placed 0/20.
- **GPU MuJoCo (MuJoCo Warp) was fast but not faithful enough** for the G1 grasp
  (2026-10-06): 141x one CPU core, but only 76.4% per-layout agreement with CPU
  physics.

## Repository Layout

| Path | Contents |
|---|---|
| `so101/` | SO-101 and fixed-base G1 MuJoCo tasks, scripted skills, perception, planner, scene memory, decision-model wrappers, grasp sampler and chooser |
| `so101/assets/` | Scenes, the vendored SO-101 model, and fetch scripts for the G1 model and scanned objects |
| `isaac/` | Isaac Lab scripts: G1 lift, grasp RL, walk-grasp-carry, two-hand box, batched inverse kinematics (IK), BC data generation, MuJoCo redraw of recorded runs, cluster batch scripts; `bench_spec.py` (benchmark contract), `grab_policy.py` (the learned grab shared by the trainer and the walk), `auto_rl.py` (the auto RL pipeline), `obs_compare.py` / `task_logic.py` (CPU-testable parts) |
| `bench/` | The walk_carry goal: result records, the selection, the Slurm driver `run.sh` (`bench/README.md`) |
| `vla/` | GPU stack: LeRobot rollouts and demo recording, perception and naming checks, planner and decision-model jobs, LoRA training |
| `warp/` | MuJoCo Warp (GPU MuJoCo) feasibility study jobs |
| `containers/` | Apptainer image definition and requirements for the stack |
| `docs/` | Plans, research notes, stack write-ups and saved source documents (`docs/sources/`) |
| `tests/`, `scripts/check_cpu.sh`, `setup.sh` | CPU-only unit tests (including a fake Isaac Lab that runs `walk_grasp.py` end to end), smoke check and pip setup |
| `scripts/check_day1.py` .. `check_day3.py`, `check_goal.py`, `select_grab.py` | The goal's CPU checks and the validation-based selection |
| top level | `train.py`, `watch.py`, `decide.py`, `bench_general.py`, `plan_bench.py`, `stack_v1.py`, `stack_v2.py` and other benchmarks |
| `RUNBOOK.md`, `docs/results.md`, `docs/project.md` | Setup and run instructions, dated results, project overview |

`runs/` is gitignored: no checkpoints, datasets, recordings or videos ship with
this repository.

## Requirements and Setup

### CPU part (MuJoCo, SO-101)

Python 3.12 and [uv](https://docs.astral.sh/uv/). `pyproject.toml` pins a
CPU-only PyTorch build.

```bash
uv sync
uv run python -c "import gymnasium, stable_baselines3, torch, mujoco"
```

Without uv, `bash setup.sh` makes `.venv/` with pip, using the same versions as
`uv.lock`. `bash scripts/check_cpu.sh` then runs the unit tests in `tests/` and each CPU entry point on a
tiny input (about 30 seconds). It needs no GPU, display, network or model downloads. The camera-image step
needs an OpenGL library (`MUJOCO_GL=osmesa` with libosmesa6) and is reported as not run without one.

Optional assets (not stored in the repository):

```bash
python so101/assets/fetch_g1.py                # Unitree G1 + Dex3 model from MuJoCo Menagerie
python so101/assets/rebuild_g1.py              # the same model without GitHub access (from the committed scene + a PyPI wheel)
python so101/assets/objects/fetch_gso.py       # the 50 Google Scanned Objects used by the SO-101 benchmark
```

### GPU part (perception, VLA, LLM planner and decision models)

A separate environment from `vla/requirements.txt` (LeRobot 0.6.1) with a CUDA
(NVIDIA Compute Unified Device Architecture) build of PyTorch; see `RUNBOOK.md` for the exact command. The full stack loads
several models at once (about 44 GiB together for stack v1), so it ran on
cluster GPUs.

### Isaac Lab part (G1 humanoid)

The setup used here is Linux only. It needs an NVIDIA GPU with ray-tracing (RT) cores, Isaac Sim 5.1 and Isaac
Lab v2.3.2 (not Isaac Lab main), plus `rsl-rl-lib==3.1.2`. `isaac/setup_ubuntu.sh`
shows the install used on an Amazon Web Services (AWS) machine; it assumes that
machine's paths. Running Isaac Sim requires accepting NVIDIA's licence terms.

### Cluster scripts

The `*.sbatch` files assume a Slurm cluster and, for Isaac and the stack, an
Apptainer container. They hard-code the author's cluster account, storage paths
and GPU types, so they must be adapted before use.

## Quick Start

```bash
uv run python train.py                              # PPO on InvertedPendulum-v5, live viewer
uv run python watch.py                              # replay the trained agent
uv run python train.py SO101Reach-v0 500000         # SO-101 reach task
uv run python watch.py SO101Reach-v0
uv run python decide.py rule 10                     # rule decision model in the viewer
```

With the GPU environment, the camera-only pick-and-place benchmark in the viewer:

```bash
python so101/assets/objects/fetch_gso.py            # needed for --set=scanned
python bench_general.py --gui --scenes=3 --workers=1 --set=scanned --objects=3
```

Isaac scripts are run with Isaac Sim's Python inside the Isaac Lab setup; each
script's top docstring lists its options (for example `isaac/walk_grasp.py`,
`isaac/box_walk.py`, `isaac/box_scripted.py`, `isaac/batch_ik.py`).
`isaac/render_mj.py` redraws recorded Isaac runs with MuJoCo, because Isaac Sim
5.1's renderer crashed on the drivers available.

## Status and Caveats

- This is research code written quickly while trying ideas. Many scripts are
  exploratory, superseded, or depend on cluster state; they are kept as a record.
- Results are development runs unless the table says held-out or
  pre-registered. Seeds were reused across variants when comparing settings.
- Automated tests (`tests/`, `scripts/check_cpu.sh`) cover the CPU parts: the SO-101 tasks with the rule decider,
  the planner checker, the scene memory, the benchmark contract, the learned-grab maths, the auto RL orchestrator on
  fake job logs, and `isaac/walk_grasp.py` end to end on a fake Isaac Lab (tensor shapes, every code path). The Isaac
  configuration classes and the training loop itself run only on a GPU.
- The real-arm code (`so101/robot.py` `RealArm`, arm calibration on hardware,
  RealSense camera) was never run on hardware.
- Perception, planner and decision models are downloaded at runtime from their
  publishers and are not included.
- Cluster scripts (`*.sbatch`) assume Slurm and Apptainer and name one site's
  account and partitions; `RUNBOOK.md` lists what to change.

## Citation

Citation File Format (CFF) metadata is in [`CITATION.cff`](CITATION.cff).

```bibtex
@misc{cleasbymayeda2026rlrobots,
  author = {Cleasby-Mayeda, Steven},
  title  = {rlRobots: Simulated Robot Arm and Humanoid Pick, Carry and Place},
  year   = {2026},
  note   = {Research code, Oregon State University}
}
```

## License and Third-Party Material

Original project code and documentation are released under the Massachusetts
Institute of Technology (MIT) License; see [`LICENSE`](LICENSE). Robot models,
scanned objects, simulators, libraries, models and saved documents keep their
own terms. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
