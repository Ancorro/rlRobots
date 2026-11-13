# Plan: direct LLM arm control, then tuning (stack 3)

A new stack beside the layered one in `docs/plan.md`. A multimodal language model drives the SO-101 directly through end-effector tool calls, with no perception, decider or scripted skills in between. It follows Leet et al., "The Bitter Lesson Comes to Robotics" (arXiv 2610.25287, copy in `docs/sources/`). The paper tests frontier API models zero-shot on a real bimanual arm. This stack adds a sim harness, open models, and then tuning.

## Question

On the same tabletop tasks and held-out layouts, how does direct LLM control compare with the layered stack, both zero-shot and after tuning?

| Arm | Controller | Reference number |
|---|---|---|
| A. Layered rule | detector -> rule decider -> scripted skills | tabletop-detect 38/40 (2026-10-03) |
| B. Layered model | detector -> Laya / Jev decider -> scripted skills | from `decide.py` / `stack_v2.py` |
| C. Direct, API | frontier LM -> `move_to` tool calls | new |
| D. Direct, open | Qwen3.5 / Gemma 4 -> `move_to`, zero-shot | new |
| E. Direct, tuned | D after SFT (supervised fine-tuning), then RL (reinforcement learning) | new |

## Paper setup and the SO-101 version

| Item | Paper (I2RT YAM, real) | Here (SO-101, MuJoCo first) |
|---|---|---|
| Arms | two, 6-DoF plus gripper | one, 5 joints plus gripper |
| Action | `move_to` per arm: (x, y, z, yaw, pitch, roll, gripper), absolute, arm base frame | `move_to(x, y, z, pitch, roll, gripper)`: yaw is set by the base pan toward (x, y), so 5 joints cannot reach a free 6-DoF pose |
| IK (inverse kinematics) | general pose solve | `so101/arm.py:ik_jaw_down` holds the jaw down with optional roll; extend it to take a pitch target, or v1 fixes pitch at jaw-down and exposes only roll |
| Other tools | `done`, `give_up(reason)` | same |
| Budget | 40 actions | 40 actions |
| Guards | clip to limits; cap distance per action | `so101/robot.Guard`: clips limits, caps speed per tick (the arm moves slower, not shorter), floor and reach stops. Add a per-action tip-distance cap (e.g. 10 cm) to match the paper |
| Observations | wrist cameras plus overhead, and proprioception | front camera (exists in `scene_tabletop.xml`); add `wrist` and `top` cameras. Proprioception as text: tip xyz (mm), pitch, roll, gripper opening, last guard event |
| Scoring | LM judge on footage, cumulative subtasks | simulator ground truth, cumulative subtasks; best subtask reached counts, as in the paper |

Still open in the paper: image resolution, frames returned per action, and how IK failures and guard rejections are reported. Task 0: check whether Inspect Robots (arXiv 2610.06306) has public code, and copy its prompt and observation format if it does. Otherwise, the starting point is 640 px images, one frame per camera after each move settles, and IK or guard failures returned as tool-result text.

## Tasks (existing scenes first, no new scenes yet)

| Task | Subtasks (score) |
|---|---|
| tabletop pick | object lifted (50), in place (100) |
| tabletop two goals | goal 1 lifted, placed, goal 2 lifted, placed (25 each) |
| sort two cubes | per cube: lifted, in its tray (25 each) |
| stack (`stack_v1.py` scene) | top block lifted, on the base block and resting (50, 100) |

## Metrics

- Mean task score; subtasks per hour (inference time plus sim time; sim time reported separately, since sim runs faster than real time); dollars per subtask for API models (token counts times provider prices).
- The paper's two diagnostics: empty-grasp rate (gripper closed with nothing between the jaws) and actions per grasp.
- Unrecoverable errors: object off the table, guard stop.
- Each run lists its settings (model, reasoning effort, image size, cameras, budget, perturbation, seeds) beside its result.

## Realism knobs

Sim leaves out the arm sag and camera parallax that separated strong and weak models in the paper, so clean sim will overstate scores. Every arm gets one clean row and one perturbed row, using knobs that already exist: `SimCameras(pose_err=..., depth_noise=..., light=True)` and `SimArm(joint_err=..., play=...)`.

## Phases

All runs go on the cluster with small samples (about 10 to 20 trials per cell, the same held-out seeds for every arm), plus 4 GUI replays per tested version.

0. **Harness and oracle.** `so101/llm_env.py`: tool schema, observation builder, cumulative scorer, per-action guard cap. `llm_bench.py`: runs any client (OpenAI-compatible: OpenRouter, vLLM, Ollama) and writes one JSONL line per trial. Oracle: the scripted pick and place, re-expressed as `move_to` calls through the same API, must score about 100%, which checks the harness the same way `plan_bench.py baselines` checks the checker.
1. **Zero-shot API models.** Three models (one each from Anthropic, OpenAI and Google) x 2 tasks x 10 trials, through OpenRouter. The paper's cost was $0.38 to $4.51 per subtask, so set a hard dollar cap in the harness (stop when it is reached).
2. **Zero-shot open VLMs (vision-language models).** Qwen3.5-4B / 9B and Gemma 4 12B (already keyed in `plan_bench.py MODELS`) on cluster vLLM. This baseline is what tuning starts from.
3. **SFT.** Expert traces: scripted pick and place recorded as `move_to` call sequences with each step's images and proprioception, with injected slips and drops so recovery appears in the data. Reuse `vla/record_picks.py` recording and the LoRA (low-rank adaptation) setup in `vla/train_decide_lora.py`. Train on train layouts and evaluate on held-out ones.
4. **RL.** Reward is the subtask score minus a per-action cost. The paper's bottleneck was efficiency (actions burned while lining up a grasp), not confusion, so the action cost is the main knob. Use GRPO-style (group relative policy optimization) group rollouts over sim seeds, starting from the SFT checkpoint; RL from 0% success gains nothing (SimpleVLA-RL Table 7).
5. **Real arm (optional).** Comes only after the audit's real-arm safety fixes (first `Guard` command not speed-limited; `RealArm.close` drops torque).

## Risks

- Fixed jaw-down pitch makes some grasps impossible; pitch support in IK may be needed before scores mean much.
- Sending images every action is slow on 4 to 12B models; measure seconds per action in phase 2 before planning RL rollout counts.
- Sim contact (MuJoCo) does not reproduce pen-cap launches or thin-wall grips, so failure types will differ from the paper's.
