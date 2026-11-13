# Plan: layered SO-101 control with a fast decision model (agreed 2026-10-02)

## Architecture

Layers are numbered bottom-up, robotics convention (renumbered 2026-10-06; earlier notes call the planner layer 0, skills layer 3).

| Layer | Job | Rate | Model | Industry name |
|---|---|---|---|---|
| 0. Skill policy | carry out "pick green_cube" from camera images | ~30 Hz | SmolVLA (450M), command-conditioned | visuomotor policy (Helix / GR00T "System 1", pi0) |
| 1. Perception | camera images -> short text scene description (objects, rough positions, what is held) | per skill | object detector + depth (sim: exact positions first) | perception / state estimation |
| 2. Decision | pick next skill + object, with calibrated confidence | every few seconds | Laya (open, 421M); Jev (closed API) as reference | task planner (SayCan; Helix / GR00T "System 2") |
| 3. Reasoner ("System 3", optional, later) | turn open requests ("tidy the table") into sub-goals; handle low-confidence escalations and repeated skill failures | rarely (per sub-goal or escalation) | a real LLM (large language model), local or API | high-level reasoner (Gemini Robotics-ER over a VLA; SayCan / Inner Monologue LLM planners) |

## Skill API (decision model's options)

- Questions per step, Jev/Laya schema: which skill (`pick`, `place`, `push`, `open_gripper`, `wait`, `done`); which object or place (one option per scene item); optional "did the last skill succeed?".
- No coordinates in commands. Skills return done/failed; perception re-reads the scene after each skill.
- Low confidence (e.g. <70%) -> stop or ask instead of acting; with layer 3 present, escalate to the LLM, which re-plans or rewrites the sub-goal.
- Layer 3 -> layer 2 interface: the LLM sets the current sub-goal (text) that the decision model reads with the scene; it never issues skills directly, so the fast loop never waits on it.

## Training

- Decision layer: RL (reinforcement learning) on multi-step task success, Laya via its option logits (RLCD-style proper-scoring rewards, then policy gradient). Baselines: rule, random, zero-shot Laya.
- Skill layer: scripted skills as teachers -> randomized sim demos (LeRobot dataset) -> SFT (supervised fine-tuning) SmolVLA -> per-skill RL in sim -> 50-100 real SO-101 demos, sim+real co-training.
- Generalization: domain randomization (objects, colours, positions, lighting, camera, physics); held-out objects/positions for every evaluation.

## Build order

1. 4060 Ti installed; both GPUs verified.
2. Skill API + scripted skills; multi-step tasks (sort two cubes, stack, recover from drop); every model shown in the GUI (`decide.py`).
3. Demo recorder with randomization -> SmolVLA SFT on 4060 Ti; swap skills in one by one.
4. RL: decision layer locally; skill policy on OSU cluster.
5. Perception: replace exact positions with detector + depth; vision decider (`decider-2b-vision`) as comparison only.
6. Optional layer 3: LLM reasoner for long requests and escalations; measure whether it beats the decision model alone.
7. Parallel track (2026-10-24): direct LLM end-effector control, then SFT and RL tuning, compared against this stack; plan in `docs/llm_control.md`.

## Known risks

- Small calibrated decision models for robot planning are new (Sept 2026) and untested; this is the research question.
- RL from 0% success gains nothing (SimpleVLA-RL Table 7): warm-start each layer first.
- Silent skill failures mislead the planner: skills must report outcome.
- Sim-to-real: LeRobot joint units match in shape, not exactly in angle; real calibration needed.

## State at agreement

- Reach (decision env): rule 100%, random 5%, zero-shot Laya 0/5. Pick-place rule baseline 67% (87% grasp). 200/200 after the 2026-10-26 fixes (results.md).
- SmolVLA runs on 1070 Ti (1.08 GB) and cluster; base checkpoint is fine-tune-only.
