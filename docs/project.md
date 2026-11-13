# Project overview

## Purpose

rlRobots began as a small reinforcement learning (RL) exercise in MuJoCo (Multi-Joint dynamics with Contact) and grew into a simulation study of robot manipulation with two robots:

- the **SO-101**, a low-cost six-joint arm with a two-finger jaw, used for camera-only tabletop pick-and-place and for a layered "perception, planner, decision model, skills" stack;
- the **Unitree G1** humanoid with Dex3 three-finger hands, used for learned grasping, then walking pick, carry and place.

The current goal is a G1 that walks around an environment and moves objects of varied shape and size, built from parts that have been shown to work.

All work so far is in simulation. Code for a real SO-101 arm and webcam exists but has not been run on hardware.

## Scope

- Simulators: MuJoCo on CPU (central processing unit), MuJoCo Warp on the GPU (graphics processing unit), and NVIDIA Isaac Sim 5.1 with Isaac Lab 2.3.2.
- Learning methods tried: PPO (Proximal Policy Optimization), REINFORCE, supervised success models for choosing grasps, behaviour cloning (BC) from scripted and generated demonstrations, and LoRA (low-rank adapter) fine-tuning of a language model used as a decision model.
- Off-the-shelf perception (Grounding DINO, SigLIP 2, SAM 2.1, Depth Anything, DAM-3B) and open language models as planners (Gemma 4, Qwen, gpt-oss and others). No perception model was fine-tuned.
- Evaluation: fixed layout seeds per role (training, validation, test), paired comparisons on the same layouts, and, for headline results, protocols written down before the run. Results, including negative ones and bugs that changed numbers, are in `docs/results.md`.

## Architecture

The SO-101 stack has four layers, numbered bottom-up (agreed design in `docs/plan.md`):

| Layer | Job | Used here |
|---|---|---|
| 0. Skills | carry out "pick object" or "place at" | scripted pick and place planned from camera points (`so101/grasp.py`, `bench_general.py`) |
| 1. Perception | camera images to a list of objects with positions | `so101/see.py` (detector, mask, depth), scene memory `so101/memory.py` |
| 2. Decision | choose the next skill and object with calibrated confidence | JEV-9B, or Gemma 4 12B with a decision LoRA (`so101/deciders.py`) |
| 3. Planner | turn a free-text request into moves; re-plan on low confidence | Gemma 4 12B (`so101/planner.py`) |

Stack versions and their tests: `docs/stack_v1.md`, `docs/stack_v2.md`.

The G1 work runs mostly in Isaac Lab:

- Fixed-base grasping: Isaac Lab's Dexsuite lift task adapted to the G1 + Dex3 (`isaac/g1_dexsuite.py`), and a grasp-parameter policy over a scripted pinch (`isaac/g1_grasp_rl.py`). An earlier MuJoCo version is in `so101/g1*.py`.
- Walking: Isaac Lab's `Isaac-PickPlace-Locomanipulation-G1-Abs-v0` (NVIDIA's pretrained AGILE leg policy plus IK (inverse kinematics) arms). `isaac/walk_grasp.py` does one-hand pick, carry and place for mixed shapes; `isaac/box_*.py` covers a two-hand box task (scripted, BC and RL variants); `isaac/batch_ik.py` solves arm IK for many robots at once on the GPU.

## Repository map

| Path | Contents |
|---|---|
| `train.py`, `watch.py` | toy RL training and playback in MuJoCo's viewer |
| `so101/` | SO-101 and MuJoCo G1 environments, perception, grasp planning, planner, deciders, memory, real-arm plug-in |
| `vla/` | vision-language-action and perception checks, stack serving, Slurm job scripts |
| `isaac/` | Isaac Lab tasks, training, evaluation and rendering for the G1 |
| `containers/stack/` | Apptainer image definition for the stack |
| `stack_v1.py`, `stack_v2.py`, `ground_bench.py`, `reid_bench.py`, `plan_bench.py`, `bench_general.py` | stack and benchmark entry points |
| `docs/` | design notes, results, research notes |

Setup and commands are in `RUNBOOK.md`. Related notes: `docs/baselines.md` (published RL-on-VLA results, checked against saved sources), `docs/grasp_research.md` and `docs/carry_research.md` (literature notes), `docs/environments.md` (object sets and indoor environments considered).
