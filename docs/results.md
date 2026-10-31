# Results

Findings from the rlRobots project, September to October 2026, grouped by topic. Each entry is dated with the day the result was recorded. Everything is in simulation; no real robot has been run.

Robots: the SO-101, a small six-joint hobby arm with a two-finger jaw; and the Unitree G1, a humanoid, here with Unitree's three-finger Dex3 hands. Simulators: MuJoCo (Multi-Joint dynamics with Contact) on CPU (central processing unit), MuJoCo Warp (MJWarp, MuJoCo on the GPU, graphics processing unit), and NVIDIA Isaac Sim / Isaac Lab (PhysX physics on the GPU).

## Conventions

- **RL** is reinforcement learning; **PPO** is Proximal Policy Optimization; **BC** is behaviour cloning (supervised learning from demonstrations); **IK** is inverse kinematics (joint angles from a hand pose).
- **Layout seeds (G1 lift tests).** Training layouts use seeds 800000+ (also 400000+ and 500000+ for later on-policy rounds), validation layouts 700000+, test layouts 900000+. A seed range is only used for the role it was given; where this rule was broken, the entry says so.
- **Paired comparisons.** Two methods run on the same layouts. "A vs B 63 / 33" means 63 layouts only A solved and 33 only B solved. p-values are exact McNemar tests on those counts. "Wilson 95 % range" is a binomial confidence interval.
- **Pre-registered** means the protocol, the headline number and the comparison were written down before the run.
- "Placed of 400" in the SO-101 benchmark is 200 scenes x 2 tasks.
- Bugs found along the way are kept, with the results they changed. Some early numbers were later shown to carry a bug; the caveat sits next to the number.

---

## 1. MuJoCo SO-101: first RL and scripted skills

### Toy RL (2026-09-27)

- PPO on Gymnasium's `InvertedPendulum-v5`: average reward 24 to 1000 (the maximum) in 50,000 steps, about one minute on CPU.
- With the live viewer on: reward 23 to 1000 in 45 seconds. `watch.py` uses MuJoCo's own viewer because Gymnasium's viewer crashed on camera moves with MuJoCo 3.14.

### SO-101 reach and decision tasks (2026-10-02)

- PPO on the state-based `SO101Reach-v0` (gripper tip to a random target, success within 2 cm): success 0 % to 100 % over 20 episodes, 500,000 steps, 7.75 min on CPU.
- Reach as a decision task (`so101/decide_env.py`, the model picks moves and IK executes): rule 100 %, random 5 %, zero-shot Laya 0/5 (it moves away). Laya is an open 421M-parameter "System 1" decision model that takes a state and fixed options and returns calibrated probabilities.
- Pick-place decision task (`so101/decide_pick.py`): rule baseline 67 % (87 % grasp). Three fixes on 2026-10-26: the step waits for the jaw to stop before the state is read (a half-closed jaw had counted as a grasp and the arm lifted early); the line-up test gets 3 mm of hysteresis once descending (cubes half a step off the tip had dithered for ever); approach and carry height 5 cm instead of 8 cm (8 cm is outside the jaw-down workspace near the base). Rule on held-out seeds 1000-1199, same code path: 124/200 before, 200/200 after, mean decisions 49.8 to 22.3. Decision-model numbers on this task from before the change are not comparable with it.
- Skill-level sort task (`so101/skill_env.py`, skills pick / place / done): scripted pick 50/50. Sort, 20 held-out episodes: rule 100 %, random 0 %, zero-shot Laya 0 %; with 30 % slips, rule 100 %.
- Sort task on the GPU (`so101/gpu_sort.py`, physics, batched IK and skills all on the GPU): rule success matches the CPU on the same 512 layouts (94 %); about 30x one CPU core at 2048-4096 worlds. Laya warm start on 256 GPU worlds: 93 % (rule 94 %); with 30 % slips 233/256 (rule 234/256). The decision layer is at the rule's ceiling; the remaining failures are pick misses. Laya inference ran at about 68 decisions/s, slower than the simulator (about 250/s).
- Physics throughput, sort scene, MJWarp on an RTX 4060 Ti: CPU one core 78k steps/s; GPU 256 worlds 0.61M, 1024 worlds 1.55M, 4096 worlds 2.05M (best), 8192 worlds 1.90M. Rule baseline, seeds 1000+: CPU one core 512 episodes in 252 s, 481/512; GPU 512 worlds 22 s, 483/512; 2048 worlds 36 s, 1936/2048, 1.8 GB card memory; 4096 worlds 63 s, 3.1 GB. Per-world contact buffers `nconmax=24, njmax=96` work; giving totals (n x 12) asked for 27 GB.

Bug found: an IK bug where joints pinned at a limit stalled the solve (7 mm error at the edge of reach).

---

## 2. VLA and the perception-decision stack (SO-101)

A VLA (vision-language-action model) maps camera images and a text command to arm motions. The project first tried one as the skill layer, then replaced it with off-the-shelf perception plus scripted skills, and built a layered stack: perception, a planner LLM (large language model), a fast decision model, and scripted skills.

### 2.1 SmolVLA as a learned pick skill (2026-10-02 to 2026-10-03)

- First SmolVLA pick fine-tune: 200 scripted demos, 2000 steps at batch 16, 4.5 GB, about 1.6-2 steps/s, 20 min, loss 0.55 to 0.053. Simulated picks on held-out seeds 900000+: 0/20; it stops 7-9 cm from the cube. On training frames it beats "hold position" on lift / elbow / wrist / gripper but is worse on shoulder pan (1.7 vs 1.2): it learned the generic motion, not where the cube is. That run used 32k samples; LeRobot's docs default to 20k steps x batch 64 (1.28M) and the SmolVLA paper's simulation runs 100k x 64. Front-camera cubes are about 8 px wide.
- The colour word barely changes the pan (swap moves it 1.7 units); predictions sit near the midpoint of the two cubes. A grasp needs about 2 pan units of accuracy. GPU-recorded demos replay on CPU 195/200, which rules out a physics gap.
- Start-frame diagnostics (`vla/start_check.py`, 40 held-out layouts x 4 seeds, pan error at the start; a constant guess scores 28.4): vision-only model 23.3, regression weights on the named / other cube 0.46 / 0.41 (it finds both cubes and cannot tell which is meant). With the target's x, y in the state ("oracle" hint): error 3.1, weight 0.99.
- Evaluation noise mattered: the same checkpoint and layouts scored 12/20 and 7/20 because policy sampling was unseeded. `rollout.py` now seeds each episode.
- Oracle-hint model, 40 held-out picks: hint off by 1 cm per axis made no difference (8/20 vs 7/20 clean). About 10 of 13 misses close on air about 1 cm off the 2.4 cm cube. Replanning every 10 / 25 / 50 ticks: 13 / 20 / 16 of 40, no clear effect. Training on to 4000 steps (loss 0.045 to 0.035): 20 / 21 / 16 of 40 at 2000 / 3000 / 4000 steps. Training layouts were as bad as held-out ones (picks 14/40), so more layouts alone would not help.
- Noise-free flow-matching sampling (`--noise-scale=0`): start-pan error 2.2 (57 % within 2 units), picks 25/40 vs 20/40 (10 gained, 5 lost). A bias of about 2 units remained that more training did not remove.
- Training with a noisy hint (1 cm), noise-free evaluation: hint off 1 cm 14/40; exact hint 19/40 (clean-trained model: 13/40 and 25/40). It learned to trust the hint less but not to find the cube with the cameras. The frozen vision-language backbone is the suspect. This line was then dropped for off-the-shelf perception.

### 2.2 Off-the-shelf perception with a scripted pick (2026-10-03)

- `vla/detect_check.py`, 40 held-out layouts, front camera at 640 px, pixel projected to the table plane at cube height: Grounding DINO tiny + colour check 100 % right cube, 0.03 cm median error, 36/40 scripted picks; OWLv2 100 %, 0.15 cm, 35/40; Grounding DINO alone 88 % right cube, 33/40; perfect simulator mask 37/40; exact position 35/40. Moving the true position by 0.3 / 0.6 / 1.0 cm gives 35 / 29 / 18 of 40. About 200 ms per image on an RTX 4060 Ti. Perception is inside the precision budget; the remaining misses belong to the scripted pick.
- Scripted-pick misses (100 held-out layouts) were all at far reach (28-31 cm). The jaw-down IK constraint was loose above 5 cm and strict below, so the wrist swung upright just above the cube and pushed it. Fix in `so101/arm.py`: the strictest jaw-down weight that still reaches within 3 mm. 91/100 to 99/100; 198/200.
- Tabletop task (`so101/tabletop.py`: red cube, blue ball, standing yellow cylinder, lying green bar; two trays and a 6 cm shelf). Exact positions, 40 layouts: pick 40/40 cube, 40/40 ball, 40/40 cylinder, 39/40 bar; pick + place 240/240. Detector (`so101/perception.py`): all objects found in 40/40 scenes, xy error median 0.13-0.22 cm (max 1.4), bar heading median 3 deg, 240 ms per scene. Full two-move tasks, rule decider, 40 held-out: exact positions 39/40, detector 38/40.
- Fixes found on the way: a 10 cm jaw opening hit the shelf and, near the base, the arm's own shoulder, so it opens 0.6 rad (about 6 cm); objects are kept at least 15 cm from the base.

### 2.3 General camera-only pick-and-place (2026-10-03 to 2026-10-05)

Stack with no learned policy in the loop: two RGB-D (colour plus depth) cameras, then Grounding DINO boxes, a SigLIP 2 choice and a SAM 2.1 mask, giving 3D points (`so101/see.py`); a grasp planner over jaw headings (`so101/grasp.py`); scripted pick and place with retries (`bench_general.py`). Objects: random primitives (`so101/objects.py`) and 50 Google Scanned Objects (GSO).

- Right object chosen: 49/50 primitives, 41/45 scanned objects (SAM 3 alone: 37/45, 2x slower).
- Primitives, 100 scenes x 2: 169/200 base; MuJoCo `noslip_iterations=3` 191/200 (now the default). Cutting reach to 23 cm raised round-object first-try picks from 17 to 22/24.
- Scanned objects 142/200 to 248/300 (seed 7) after a list of fixes: release 4 mm above the surface, footprint inside the place, off-centre hold correction, aiming the object's bottom when the jaw tilts at far reach, grasping tall things lower, dropping robot and fixture pixels from masks, look-alike objects sharing a name. Requiring two-camera box agreement made perception worse (72 vs 87/90) and was dropped.
- **Held-out result** (seed 8, 200 scenes x 2 tasks): primitives 369/400 placed (92 %; eggs 51/69 worst, boxes 56/56), scanned household objects 341/400 (85 %). Remaining failures: eggs and flat round things squeezed out, thin cards tipping, about 5 % wrong object.
- Perception cost on an RTX 4060 Ti: 2.4 GB weights, 3.3 GB peak, about 1.2 s per look (two cameras, fp32).

**Realism matrix** (seed 8, 200 scenes x 2, placed, primitives / scanned): one camera 360 / 327; + depth noise 353 / 330; + calibration error 0.5 cm, 0.5 deg 354 / 307; 1 cm, 2 deg 234 / 245; + random light 225 / 254; two cameras with all of it (1 cm, 2 deg) 292 / 275. Camera calibration accuracy matters; noise and lighting barely do.

**No depth camera** (Depth Anything V2 depth scaled by the known table plane, one camera): object depth error median 4-5 mm in simulation; scanned failures are mostly "no grasp fits" because estimated depth smears edges and objects look 2-3 cm too big. Trimming mask edges by 2 px made objects look too small and was worse ("jaw closed on nothing" 243 to 432): undersizing is worse than oversizing. 1 px trim kept. Held-out seed 8: 337 / 270 of 400 (84 % / 68 %; depth camera in the same setup 354 / 307).

**Arm as a plug-in and servo realism** (`so101/robot.py`, 2026-10-03). A safety layer (`Guard`) sits in front of a simulated or real arm. Seed 8, 200 scenes x 2, placed: new-code baseline 372 / 345; 0.5 deg dead band + encoder steps 372 / 338; joint calibration error 1 deg 363 / 332 (tip off by median 5 mm), 2 deg 306 / 307 (11 mm), 3 deg 241 / 262 (16 mm). Gripper torque at LeRobot's 50 % limit (1.47 N m) 386 / 350, now the default; 0.95 N m 374 / 335. Joint calibration, like camera calibration, is what matters: 2 deg costs about 15 %.

**Camera-based arm calibration** (`so101/armcal.py`, 2026-10-04): an ArUco marker on the fixed jaw, about 20 planned poses, one Levenberg-Marquardt fit of joint offsets, camera pose and marker mount.
- Self-test, 120 setups (joints 2 deg, camera 1 cm / 2 deg, tape 2 mm / 3 deg): 0 failures, fit 0.2 px; camera-to-arm error at targets median 18 mm to 0.33 mm (max 0.63). Not separable: pan offset vs camera yaw about the base, and roll offset vs marker rotation.
- Benchmark (seed 8, one camera, depth noise, fitted table plane, servo play): no errors 368 / 341; joints 2 deg + camera 1 cm / 2 deg 175 / 185; the same with `--arm-calib` 377 / 319. About 6 s per calibration in simulation.
- Run-to-run spread on scanned objects is about +/-14 of 400. Extra wrong-object looks after calibration came from masks covering the shelf or tray (45 of 71 wrong masks), because the believed camera pose draws known geometry slightly off. Fix: drop masks that are more than half known pixels (`known_reject`). With it: calibrated errors 320 / 71 to 329 / 27 (scanned placed / wrong masks); noise floor 327 / 23. The calibrated arm then matches the no-error floor.

**Single-camera depth models** (`vla/depth_check.py`, 2026-10-04; scanned objects, object error median / p90 mm, edge smear): Depth Anything V2 Base 7.8 / 17.1, 16 %; V2 Large 7.6 / 14.1, 22 %; Depth Anything 3 mono 7.5 / 17.2, 26 %; MoGe-2 L 22.2 / 34.8, 24 %. Benchmark, seed 8, scanned placed of 400: V2 Base 287, V2 Large 275. V2 Base kept. Newer models do not fix edge smear.

**Mask models** (2026-10-05; `vla/mask_check.py`, 14 scanned objects, same detector box, IoU (intersection over union) median / p10): SAM 2.1 tiny 0.968 / 0.628, SAM 2.1 large 0.976 / 0.858, SAM 3 0.975 / 0.904; 27 / 80 / 173 ms per mask. Pre-registered rule: keep only if more than 14 better somewhere and nothing more than 14 worse. Placed of 400, seed 8, depth camera, primitives tiny / large / SAM 3: 374 / 375 / 376; scanned 341 / 336 / 341. Webcam-only scanned: 287 / 297 / 285; seed 9: tiny 289, large 305. SAM 3: no gain, off. SAM 2.1 large: +26 of 800 webcam-only, wrong-object looks 84 to 57; now the default. A plain `from_pretrained("facebook/sam3")` left SAM 3's box-prompted mask decoder random (IoU 0.007); fixed by key mapping.

### 2.4 Planner LLMs (2026-10-05 to 2026-10-06)

`so101/planner.py` + `plan_bench.py`: plan-only benchmark (no physics), 569 requests over 60 scenes, 12 request families (single, two, all-but, superlative, category, not-category, sort, conditional, clear, swap, already-done, impossible), exact goals, scene text from simulator truth. Oracle 100 %, random 4.4 %. Models served with vLLM.

- % right (median output tokens): Qwen3.8-27B thinking 99.8 (258), Gemma 4 12B thinking 99.1 (943), gpt-oss-20b medium 98.2 (367), Qwen3.8-27B 97.7 (25), Gemma 4 12B 97.4 (30), Qwen3.5-9B thinking 95.6 (3121), Cosmos-Reason2-8B reasoning 94.9 (310), Qwen3.5-4B thinking 94.9 (1631), Qwen3.5-9B 94.2, Gemma 4 E4B 93.8 / 92.6, gpt-oss low 92.1, Qwen3.5-4B 91.4, Cosmos no reasoning 89.3.
- Pick for a 16 GB card: Gemma 4 12B (about 7 GB at 4-bit). Remaining errors: near-miss names taken for missing objects, superlatives without thinking.
- Prompt fixes that mattered (all models rerun after them): "a place holds several objects, that is not stacking"; "Move ... to <place>" wording; the shelf described as "top 6 cm above the table" (gpt-oss read "6 cm high" as a height limit); cut-off thinking counted as no answer.
- 4-bit Gemma 4 12B (Ollama Q4_K_M) on the 569 requests: no thinking 97.9 %, thinking 98.9 % (full precision 97.4 / 99.1: 4-bit costs nothing). With a forced answer schema (constrained decoding): 97.2 / 92.6; the schema hurts with thinking.
- Horizon test (lists of 1-32 explicit moves, 50 each, 4-bit Gemma): plans exactly right, no thinking, 100 % up to 8 moves, 96 % at 12-16, 74 % at 24, 44 % at 32. Thinking is worse on long lists (36 % at 32; it hits the 8,192-token limit).
- Large held-out set (3,600 requests), % right thinking / no thinking: Qwen3.8-27B 99.9 / 97.4, Gemma 4 12B 99.5 / 97.9, Gemma 4 12B 4-bit 99.4 / 97.4, gpt-oss-20b 98.9 / 90.5, Cosmos-Reason2-8B 95.2 / 88.2, Gemma 4 E4B 94.8 / 91.2, Qwen3.5-9B 94.7 / 91.2, Qwen3.5-4B 93.9 / 88.3. 95 % ranges about +/-0.3-0.6 points. Same ranking as the small set.

### 2.5 Listing and naming objects from images (2026-10-06)

400 scenes; "found" is the share of visible objects named, "wrong" the share of names that match nothing (simple shapes | household objects).

- Whole-picture listing with 4-bit Gemma 4 12B: open, no thinking, found 66 % / 50 %, wrong 30 % / 43 %; thinking 70 % / 53 % found. Not good enough for the planner's scene list.
- Detector first (Grounding DINO tiny, prompt "object.", then each crop named), found / wrong / count right %: Gemma closed list 82.3 / 23.3 / 67.5 | 55.2 / 48.1 / 61.0; SigLIP 2 closed 76.7 / 35.6 / 15.0 | 73.0 / 34.3 / 50.5. Detector recall 99.8 / 99.6 % (IoU at least 0.5), 1.1-1.3 junk boxes per scene. Misses are naming errors. Caveat: filter level and merging were set after looking at these scenes.
- More namers on detector crops, found / wrong %: Qwen3.8-27B FP8 closed 84.9/17.0 | 70.8/27.9; Qwen3.5-9B 83.4/29.6 | 69.8/30.6; JEV-9B 68.8/41.5 | 55.9/45.8; SigLIP 2 giant-384 73.5/30.2 | 80.5/20.8 (best household, 0.2 s/scene); PE-Core-L 71.2/40.1 | 78.4/30.4; FG-CLIP 2 So400m 76.0/35.9 | 80.2/22.6; TULIP G 73.6/37.3 | 71.4/34.4.
- Region captioners (open naming, judged by Qwen3.8-27B): DAM-3B on detector boxes 92.4/26.2 | 78.0/38.0, on true boxes 95.5/4.5 | 87.1/12.7, about 15 s per scene; the best namer without a name list, with wrong names coming from junk boxes. PAM-3B and Rex-Omni scored poorly (setup may be at fault; not investigated).

### 2.6 Decision models (2026-10-02, 2026-10-07, 2026-10-21)

A "Jev-style" decision model reads a state and a fixed list of options and returns calibrated probabilities over them. JEV-9B is an open model of this kind (`autotrust/JEV-9B`); SubQ is a paid API (application programming interface) offering the same kind of decision request.

- Sort task, 50 held-out episodes: JEV-9B untuned 47/50 sort, 47/50 sort-with-drops; rule 50/50; Laya (tuned warm start) 50/50. JEV-9B's 3 failures answer "place green cube" instead of naming a tray.
- After restricting the object options to what the chosen skill accepts: JEV-9B untuned tabletop 49/50 (rule 50/50; Laya untuned 0/5), sort 50/50. JEV-9B was adopted because it works untuned while Laya needed task-specific training.
- **Gemma 4 12B as the decision model** (2026-10-21): options lettered, one masked forward pass, probabilities read over the allowed letters, one question per step over valid (skill, object) pairs. Untrained: tabletop 50/50, sort 50/50, but it never answers "done".
- Decision LoRA (low-rank adapter, r16, 65.6M trainable parameters, 1 epoch, about 40 min on one GPU) on simulated decision examples (train 3,327 steps from 800 scenes, held-out 692 from 150 other scenes): untrained 66.3 % right (mean p(right) 0.63; never "done", 0/129; strong position bias: letters A-C 82-93 % right, D 14 %); with the LoRA 99.7 % (690/692), mean p(right) 0.996. Caveat: held-out data comes from the same generator.
- In the stack (dev 60, vision grounding, section 2.8): LoRA Gemma 39/60 done with 0 decision failures; SubQ decider 37, JEV-9B 33, untrained Gemma 21. Decisions take 0.4-0.5 s per request vs 3.1 s for SubQ.
- Serving detail: vLLM returns one-hot log-probabilities when processed log-probabilities are requested at temperature 0; requesting temperature 1 and taking the argmax gives the softmax over the allowed letters.

### 2.7 Stack v1 (2026-10-09)

Free-text request, then Grounding DINO + SAM 2.1 + DAM-3B listing, then a Gemma 4 12B plan, then JEV-9B decisions, then scripted skills, looking again after each skill. Escalation: a skill choice with confidence below 0.7 sends the planner back to re-plan. Details and the pre-registered protocol are in `docs/stack_v1.md`.

- Development (seed 400000): 8/12 done after fixes (merge the two cameras' views of one object, list a held object as "in the gripper", keep ids for moved objects).
- **Test** (seed 420000, 100 episodes, frozen system): 62/100 requests done (primitives 35/50, household 27/50). Causes: perception 21, decision 9, skill 8. Perception finds every object (345/345, 0 junk), but DAM's descriptions miss the asked-for word: plans right 79/100, 100/100 with true names.
- By request family: impossible 10/10, superlative 7/8, category 10/14, conditional 11/18, two 8/12, single 7/12, sort 5/10, not-category 2/4, all-but 2/12.
- References on the same 100 episodes: true names 75/100 (paired 15 vs 2, p = 0.002); rule decider 67/100 (5 vs 0, p = 0.06). Caveat: "decision" failures mix stopping too early with running out of steps after failed grasps, which are really skill failures.
- Bug found: JEV's server refuses a choice with one option (HTTP 400). The client now treats a single option as the answer.

### 2.8 Stack v2: scene memory, grounding, timing (2026-10-10 to 2026-10-21)

- **Re-identification test** (pre-registered, seeds 620000+, 100 episodes, 3,441 detections), identity accuracy: SigLIP 2 appearance + position 0.984, position only 0.912, v1's nearest-within-4-cm rule 0.776. Twins 0.981 vs 0.906 (168 vs 4, p = 1e-44); moved objects 0.975 vs 0.810. Test 0.984 vs development 0.979, so the settings did not overfit. Scene memory (`so101/memory.py`) went into v2.
- **Grounding test** (`ground_bench.py`, seeds 660000+, 100 scenes, plans right): true names 99, vision (the planner sees object crops) 91, SigLIP 2 phrase match at 0.3 74, DAM names 67, Gemma closed choice 58. Vision vs DAM 24 vs 0, p = 1e-7.
- Dev 60 (seed 400000, episodes 0-59), requests done: v1 30/60; v2 with DAM names 29, vision 33, vision + SubQ decider 37; plans right 39 / 44 / 45 (true names 60). Remaining causes for vision + SubQ: perception 14, skill 9, decision 0.
- Timing (v2, vision + SubQ, 12 requests): mean 60.8 s per request = looks 23.8 (DAM names 11.1 of it), arm motion 10.6, planner 8.9 + re-plans 1.3, skill-time looks 3.5, decisions 3.1, grasp planning 0.6. GPU memory peaks on 40 GB H100s: JEV vLLM 32.0 GiB (its 85 % reservation), Ollama Gemma 4-bit 10.2, DAM-3B 7.5, the stack process (DINO, SAM 2.1, SigLIP 2, MuJoCo) 4.5. Stack v1 needed about 44 GiB in all and did not fit one 48 GB card.
- Bug found: a held box seen more than 6 cm from the jaw was matched as a new sighting and lost its grasp.
- **Test, LoRA Gemma decider** (2026-10-26, seed 420000, the 100 episodes of v1's test; not the pre-registered 200-episode frozen-v2 test): 72/100 requests done (primitives 40/50, household 32/50), mean 40.4 s per request. v1 on the same episodes 62, rule decider 67, v1 with true names 75. Plans right 93/100 (true names 98). Causes: skill 17, perception 6, decision 2, disturbed 2, planner 1. Skill failures are now the largest cause.
- By request family: impossible 10/10, conditional 11/18, category 12/14, single 10/12, two 10/12, superlative 6/8, sort 6/10, all-but 5/12, not-category 2/4.
- **Perception probe** (`percept_probe.py`, planner only, ground_bench dev scenes seed 640000, 60 scenes), plans right: vision 53, best crop from each camera 54, DAM names hidden 49, both 44, true names 60. Against vision: two views 4 vs 3 (p = 1), names hidden 2 vs 6 (p = 0.29), both 5 vs 14 (p = 0.06), true names 7 vs 0 (p = 0.016). The DAM names still help the planner; a second crop adds nothing.

### 2.9 SO-101 grasp stability (2026-10-09)

`bench_general.py`, seed 7, 100 scenes. Primitives placed / drops of 200: base 198 / 17, slow first 2 cm of lift 200 / 6, lower grasp 198 / 17, smooth carry 198 / 17, neighbour clearance 197 / 17. Household objects (98 tasks): base 76, slow lift 72, lower 82; about 15 % fail after release and about 21 % knock a neighbour. The slow first lift is now the default.

---

## 3. G1 grasp learning (fixed base, one hand)

The G1 stands welded in place at the table; the right arm and Dex3 hand grasp one object. Three lines of work: a learned grasp chooser in MuJoCo (3.2), grasp-parameter RL in Isaac Lab over a scripted pinch (3.5), and closed-loop PPO on Isaac Lab's Dexsuite task (3.6).

### 3.1 MuJoCo G1 setup and scripted baseline (2026-10-06)

- Model: MuJoCo Menagerie `unitree_g1` with Dex3 hands. Pelvis welded so the table top is at hip height; gravity compensation on all bodies; left arm parked.
- The Dex3 is used as a two-jaw gripper: the thumb pinches against the curled index finger, middle finger tucked away (index + middle as the fixed jaw let 2-4 cm things slip through: 6/24 lifted vs 11/24). Opening 0.8-7.3 cm.
- The IK ignored collisions, so the elbow or forearm against the torso stopped the hand 3-5 cm short. Contact checks at candidate poses plus 4 IK postures: clean reach objects 109 to 150/180, red tray 17 to 39/54, shelf 27 to 48/54.
- Scripted pick and place, no cameras (8 small shapes x 3 spots): lifted 14/24 (boxes 9/9, balls 5/6, cylinders 0/9), placed 8/24.
- Round things fail because of the fingers: the Dex3 finger tips are rounded and the jaw faces close in a V (index face 20 deg away from the thumb, thumb face 27-32 deg toward the index).
- Camera-only benchmark, scripted grasp (seed 8, 100 scenes x 2, front camera, depth noise; placed / picked of 200): G1 primitives 47 / 74, scanned 90 / 127. SO-101 on the same scenes: 183 / 197 and 172 / 188. G1 failures are mostly "jaw closed on nothing" and, for scanned objects, "no grasp fits" (the hand opens 7.3 cm, the SO-101 jaw 9.5 cm).

### 3.2 Learned grasp chooser in MuJoCo (2026-10-06)

**Held-out lift test** (protocol written before any sampling or training): 6 shape classes (box, upright cylinder, lying cylinder, ball, capsule, egg), 60 layouts each, seeds 900000+. One object; reach, close, lift 10 cm, hold 1 s; success from simulator truth (object at least 9 cm above its start and within 5 cm of the pinch site). This replaces the earlier claw test as the baseline.

**Pre-registered criterion:** a learned method must beat the scripted grasp on at least one class, with its lower 95 % bound above the scripted rate.

- Grasp sampler (`so101/g1_grasps.py`): random grasps (tilt 0-80 deg, heading, height 20-80 %, clearance, margin), executed and labelled; successes re-run 3 times with 0.4 cm aim error and 5 % size error. Lifted: box 100/170, upright cylinder 104/165, lying cylinder 20/105, ball 17/175, capsule 18/102, egg 17/93. Lifts fall with approach tilt for round shapes (balls 15/149 at 0-20 deg, 2/71 at 20-40). The earlier 0/9 for cylinders came from a grasp-detection rule and the IK tilt schedule, not from the hand.
- Unitree's real G1 + Dex3 teleoperation recordings (13 `unitreerobotics/G1_Dex3_*` datasets, about 1,600 pick episodes used): joints map one-to-one by name to the Menagerie model. Real closing is a wrap (index and middle tips +0.57 / +0.62 rad, thumb barely moves; one direction explains 72 % of closing moves). Used as a second hand-closing mode in the sampler: 0/545 on every shape (aimed with the pinch's aim point, which may not suit a wrap). Off.
- **Results, lifted of 60 per class** (box, upright cylinder, lying cylinder, ball, capsule, egg; total of 360):

| Method | Box | Cyl. | Lying cyl. | Ball | Capsule | Egg | Total |
|---|---|---|---|---|---|---|---|
| Scripted grasp | 40 | 37 | 24 | 6 | 26 | 21 | 154 |
| Chooser v1 | 36 | 43 | 28 | 10 | 23 | 16 | |
| Chooser v2 | 49 | 50 | 31 | 12 | 26 | 16 | 184 |
| Chooser v3 | 49 | 53 | 39 | 17 | 34 | 20 | 212 |
| Chooser v4 | | | 45 | | | 26 | 222 |

- v1: an MLP (multilayer perceptron) success model on 1,355 executed grasps, picking the best predicted of 256 draws. Not a win by the criterion.
- v2: also trained on 1,200 grasps sampled around the scripted grasp (2,575 in all); ranks wide draws, nearby draws and the scripted grasp. **Criterion met**: lower bounds above the scripted rate for boxes (70 vs 67 %), cylinders (72 vs 62 %) and balls (12 vs 10 %; marginal, one layout). Paired: cylinders 16 / 3 (p = 0.004), all classes 63 / 33 (p = 0.003); eggs 5 / 10 (worse). Compute: 14 CPU-hours. v2 was overconfident on eggs (predicted 0.95-1.0 for grasps that missed); training now stops early on held-out layouts.
- v3: v2 plus on-policy data (its own picks on 900 fresh training layouts) plus early stopping; 3,470 grasps, ranking AUC (area under the ROC curve) 0.85. Paired vs scripted 84 / 26 (p < 0.001); vs v2 52 / 24 (p = 0.002). The best epoch was 1-2 of 300 in every chooser.
- v4: one more on-policy round (4,658 grasps): 222 vs v3 212 (p = 0.17, not significant); lying cylinders 45 vs 39 (p = 0.031).
- **Aim error** (0.4 cm per axis, 5 % sizes, same layouts), lifted of 360: scripted 126, v2 146, v3 164. Cylinders are the most sensitive (v3 53 to 36). The chooser was trained on true positions.
- **Residual refinement** on v3 (REINFORCE, linear-Gaussian corrections to five grasp parameters, 15 iterations x 96 layouts): 211 vs v3's 212 of 360. No gain; stopped as pre-registered.

**Chooser in the camera-only benchmark** (seed 8, 100 scenes x 2, placed / picked of 200): primitives scripted 47 / 74, chooser v3 53 / 75 (p = 0.38); scanned 90 / 127 vs 91 / 134. No real gain. Two grasp-check faults hid good grasps for both: the stepped hand close ended at about 1e-17 rather than exactly closed while the check tested equality, and a 1.5-2 cm object between the tips reads like an empty hand. With a touch-based hold check (both thumb and index tips on the object): scripted 73 / 138, chooser 87 / 139 (placed 31 / 17, p = 0.059). Why the lift-test lead shrinks: camera centre error median 0.85 cm (twice the tested 0.4 cm), 15 % of looks pick the wrong object, 3-5 objects per scene vs 1 in training.

- After fixing the hand-close rounding (seed 8, placed of 200): primitives scripted 71, v3 88 (p = 0.033); scanned 104 vs 97.
- **Chooser v4b** (pre-registered; v3 data plus 5,108 camera-benchmark attempts on other seeds): primitives 87 vs v3 88 (p = 1.0), scanned 116 vs 97 (43 / 24, p = 0.027). Passes. Balls 4/48 and eggs 4/33 stay near zero.

**Reproduction without the Menagerie download (2026-10-27).** `so101/assets/rebuild_g1.py` rebuilds the G1 + Dex3 model from the committed `scene_g1_cull.xml` (the whole model is embedded there) and the Unitree meshes in the `example-robot-data` PyPI wheel. Checks against the numbers above: reach red tray 39/54, blue tray 47/54, shelf 48/54 (identical; objects 160 vs 150/180); scripted held-out lift 154/360 (identical). Sampler 1,200 wide-prior + 1,200 near-scripted grasps (4 CPU cores, 3 h); chooser v2 trained on them: **212/360** on the held-out test vs scripted 154 (paired 82 / 24, p < 0.001; cylinders 50 vs 35, lying cylinders 41 vs 24, capsules 38 vs 27); with the 0.4 cm aim error 167 vs 124. The finger-gap table reads ~1.7 mm tighter than the stored constants (a mesh revision), without visible effect.

### 3.3 MuJoCo Warp feasibility: no-go (2026-10-06)

Go/no-go rules fixed in code beforehand: at least 20x one CPU core, at least 90 % per-layout lift agreement with the CPU scene. Speed passed: an A40 ran 201 lift trials/s at 8192 worlds, 141x one core; about 85 % of the CPU executor's 4 s per trial was IK and planning, not physics. Fidelity failed: MJWarp has no noslip solver, and without noslip cylinders lift 0/79 even on CPU. The best stand-in (elliptic friction cones, impratio 10) gave 76.4 % agreement. Moving the whole G1 study onto elliptic-cone physics (CPU and GPU agree 97.9 %) was considered but not done.

### 3.4 Isaac Lab port (2026-10-06 to 2026-10-07)

- Throughput, Unitree's `unitree_sim_isaaclab` G1-29dof-Dex3 scene, Isaac Sim 5.1 + Isaac Lab 2.3.2, headless, cameras off, zero actions, A10G 24 GB: 1 / 64 / 512 / 2048 robots = 54 / 2,724 / 13,135 / 34,324 control steps/s (10 ms each), GPU memory 1.8 / 2.1 / 3.8 / 9.5 GB.
- RL loop check (Isaac-Lift-Cube-Franka-v0, rsl_rl PPO, 4096 envs): 300 iterations, 29.5M steps in 4 min 54 s (about 98k steps/s); mean reward 0.7 to 30.5.
- **Scripted pinch planned in Isaac**, 360 held-out layouts: first reported 225/360 lifted, but self-collision was off (Unitree's default). With self-collision on, Unitree's 145 mm hand camera-mount collider fights the palm (about 12,000 N) and was removed. **Corrected baseline: 155/360 lifted** (box 27, capsule 27, cylinder 26, lying cylinder 26, ball 26, egg 23); 187 held.
- What the port needed: objects spawned at twice the env offset (bug); hand drives matched to MuJoCo (kp 500, kv 4.5, armature 0.01; stiffness 20 gave 1/60 vs 35/60); torsional friction, friction combine max, 32 solver iterations; closing direction chosen by smallest turn. Unitree's Dex3 USD matches Menagerie's to 0 mm.
- Rendering: Isaac Sim 5.1's RTX renderer crashed at startup on drivers newer than its tested 580; Isaac Sim 6.1 worked for walking replays. Lift clips are redrawn in MuJoCo (`isaac/render_mj.py`).

### 3.5 Grasp-parameter RL in Isaac Lab (`isaac/g1_grasp_rl.py`, 2026-10-07 to 2026-10-08)

A policy picks 5 (later 8) adjustments to the scripted pinch (offsets, height, heading, closing amount; later hand tilt and closing speed) once per grasp, trained with REINFORCE and a per-shape baseline. This is open loop: the adjustments are chosen once, then the scripted motion runs.

- v1 (60 iterations x 360 envs, 45 min): held-out lifted 242/360 vs scripted 162/360 in the same run (107 vs 27, p = 1.8e-12).
- Parallel copies per layout did not raise the score: v1 242, repeat v1b 232, v3 (8 copies) 235, v4 (8 copies) 242. Run-to-run spread is about 10 lifts. Re-scoring the same policy also moves it (v1 242 to 244, v4 242 to 236). v2's drop (218) was mostly a timing mismatch between training and test reach speed.
- v5 (1,440 objects x 2, new poses every round, 150 rounds) and v6 (v5 + 3 more adjustments). Validation: v1 244, v5 258, v6 266. Pre-registered test: v6 249, v5 236, v1 244 (v6 vs v1 p = 0.69). No reliable test gain.
- Closed-loop retries (a failed 3 cm test lift opens, re-aims at the object's current pose, tries again; 2 retries): v6 262/360 vs 249 (p = 0.024). Retries that turn 90 deg (264) or go 6 mm lower (263) did no better.
- **Bug that changed every earlier number:** 63 of the 98 v6 misses were objects falling hundreds of metres, knocked off or through the table, about 10 % of envs in every evaluation. The reach swung straight at the target and the fingers swept objects; the IK moved the waist, which lowered the left hand onto the table. Fix: reach up 15 cm, across while turning, then down; left arm parked and folded. After the fix: scripted 206 (was 161), v6 274 (was 249), v6 + 2 retries 286/360 = 79 %.
- v7 (v6's setup retrained in the fixed scene, 120 rounds, pre-registered): 295/360, + 2 retries 305/360.
- **Realism audit** found: every egg was a ball (Isaac Lab gives sphere meshes a bounding-sphere collider); a 1000 N m waist used up to 20x the real G1 waist torque; hand colliders had no material; the policy saw exact object state. Fixes: 0.4 cm perception error + 5 % size error; real G1 torques; object friction 0.5-1.0; convex-hull eggs; a 1 m/s depenetration cap. Each evaluation now adds a shake test after the hold (3 cm at 2 Hz along x, y, z) and a failure reason.
- Ablation (test, lifted of 360, scripted / v6 / v7): old settings 206 / 275 / 294; + perception error 200 / 252 / 278; + real torques 203 / 277 / 297; + friction 210 / 280 / 304; + egg hull 219 / 276 / 289; + depenetration cap 205 / 275 / 295; all fixes 200 / 242 / 268; all + 2 retries 235 / 267 / 293. Only perception error costs much; the learned policies leaned on exact positions.
- Forces: hand on object peak median about 10 N, p90 35-50 N; arm and wrist motors reach their real limits in most episodes.
- "Arm off target" (scripted 91/360 with all fixes): an arm joint, mostly wrist yaw, sits at a limit at the end of the descent. Joint-limit-aware IK variants did worse (194, 68, 70 vs 200).
- Width-aware scripted pinch: scripted 200 to 218 (p = 0.005); v7 268 to 254.
- **v8** (pre-registered; v7's setup trained with all fixes and the shake in the reward): 282/360 lifted, 277 kept; v7 268 / 261; scripted 198 / 195. With 2 retries: v8 302/360 (84 %). Household objects (114 objects x 3 poses = 342, never trained on): lifted / kept scripted 41 / 38, v7 39 / 38, v8 58 / 58 (v8 vs v7 p = 2e-5).
- v9 (v8 + width pinch): 270/360 vs v8 282 (p = 0.058). Not better.
- Household failures (v8): arm off target 144, slipped 93; 86 of 342 objects tip over while settling. The household set (`so101/assets/objects/gso_g1.py`) keeps 114 of the 1,030 Google Scanned Objects the Dex3 pinch could hold.
- Summary: scripted 162 to learned about 245-265 before the fall fix; the open-loop adjustment idea plateaus near 70 %.

### 3.6 Closed-loop PPO on Isaac Lab Dexsuite (`isaac/g1_dexsuite.py`, 2026-10-08 to 2026-10-18)

Isaac Lab's Dexsuite lift task (KUKA arm + Allegro hand) adapted to the G1 + Dex3: PPO moves 16 joints (waist yaw/pitch, right arm, right hand) every 20 ms. Real G1 torques, MuJoCo-matched hand drives, our table, one primitive per env, friction 0.5-1.0, perception error on the object points. It is a teacher policy: it sees object orientation and 64 surface points every step, more than a camera would give.

- dex-v1 stopped at iteration about 650: not learning (objects out of bounds 51 % to 71 %, never a thumb + finger contact). Cause: 11-13 % of objects were launched at reset because they started floating against the parked left hand or the right wrist. Fix `--reset=rest`: objects start resting on the table.
- dex-v2 stuck: the policy hovered. Targets 5-22 cm above the table sat near the resting object, so the ungated success reward paid for leaving it alone; in zero gravity any touch sent it off.
- dex-v3 (higher targets; success reward only while thumb + finger touch): still no contact.
- **Bug:** the hand points used by the reach reward were link origins, which on Dex3 fingers sit 5.2 cm back from the tips; the reward pulled finger joints, not pads, toward the object. Fix: palm + finger pads.
- dex-v4b (pads + full gravity from the start) was the first run with any thumb + finger contact (0.0004 vs KUKA control 0.044).
- dex-v5 (reach reward on the thumb and index pads, 0.3 N contact threshold; v5a 7 hand joint deltas, v5b one pinch-closing value): contact 0.15 / 0.13 at iteration 600. Learning to grasp, but motion was erratic: arm jerk RMS (root mean square) 2950 / 4490 rad/s^3.
- Smoothing: LCP (Chen et al. 2024, a penalty on the policy's gradient with respect to its input, |grad_s log pi(a|s)|^2) at 0.002 did nothing measurable. dex-v6b (action low-pass filter, an EMA (exponential moving average) with factor 0.4, + joint acceleration and action-change penalties): acceleration 5x lower, jerk 11x lower, and by about 2000 iterations it was carrying objects to targets while v5a / v5b stayed at curriculum 0. Smoothness made lifting learnable.
- **dex-v6b result** (pre-registered selection: validation kept 348 at iteration 3000, 342 at 4999, within 10, lower jerk picks 4999): test 345/360 lifted, 345 kept; v8 284 / 281 on the same layouts (72 vs 11 lifted, p = 4e-12). Household (342, never trained on): v6b lifted 43, kept 40; v8 58 / 58 (kept 22 vs 40, p = 0.03): v6b is worse on household objects. Caveat: one validation pass ran without its seed flag and scored all saves on the test layouts; selection was redone on validation layouts.
- dex-v7 (object pool, 100 iterations from v6b with jerk and posture penalties): primitives lifted 63/72, kept 63/72; pool objects kept 333/720. Object pool: GSO 114 + 557 DexGraspNet objects = 671 (train / val / test 522 / 77 / 72), split by object name hash, not by source set. v6b with the same pool settings kept 293/720.
- Solver iterations vs throughput (walking-base flags, RTX 8000): env steps/s 2k envs 6.4k, 4k 9.6k, 8k 11.6k, 16k 12.1k; at 8k envs, object solver iterations 8 + self-collision off gave 29.5k. Grasps get worse with fewer iterations for a policy trained at 32: pool kept 32 iterations 333, 16 288, 8 259. A v7 fine-tune at 16 iterations, 8192 envs: pool kept 362/720 (confounded: 100 more iterations and more samples).
- V100 GPUs cannot run the Isaac Sim container used here ("no kernel image is available").

---

## 4. Walking G1: pick, carry and place

### 4.1 Walking with Dex3 hands (2026-10-07)

- Unitree's G1-29dof velocity task on the free-base G1 + Dex3, 1,500 iterations from scratch: failed (the policy learned to fall in 0.1 s). A plain G1 from scratch in the same setup learns, so the Dex3 whole-body model was the problem.
- Cause found: the Dex3 whole-body USD (Universal Scene Description) model file has 4 sensor rigid bodies with mass 0, so PhysX assigned about 6.7 kg of phantom mass (41.3 kg vs 34.6 kg authored). A mass override (`isaac/dex3_massfix.py`) fixed the mass, but training from scratch still fell: mass was one bug, not the only one.
- Unitree's released G1 walking policy (an ONNX, Open Neural Network Exchange, file), zero-shot: plain G1 96 % fell, Dex3 100 %. Not usable as-is.
- Isaac Lab's published `Isaac-Velocity-Flat/Rough-G1-v0` checkpoints: 512 robots x 20 s, 0 falls; +1.4 kg per arm/hand link 0 falls. But that G1 has a different arm and hand model.
- **Working:** `Isaac-PickPlace-Locomanipulation-G1-Abs-v0` (Unitree 29-dof G1 + Dex3, NVIDIA's pretrained AGILE leg policy taking velocity and hip-height commands, plus Pink IK arms). Command 0.5 m/s: 0.47 m/s, 0 falls; standing 0 falls; wrists held to 1 mm. Gotchas: import pinocchio before AppLauncher; qpsolvers 4.4.0 (4.13 breaks pink 3.1.0); IK world targets are env-local.

### 4.2 Scripted walk, grasp, carry, place (`isaac/walk_grasp.py`, 2026-10-09 to 2026-10-15)

- Walk 1.2 m to a table, stop, scripted pinch: 3 x 12 robots all arrived, 0 falls, stop within about 1.5 cm. With hands raised 20 cm before walking (they had swept objects off at table height): box 10/12, cylinder 5/12, ball 11/12 lifted; 0 falls.
- AGILE ignores a standing turn command below about 1 rad/s (yaw 0.5 gives 0.00 rad/s; 1.0 gives 1.16). Turning is bang-bang at 1.0 rad/s.
- With `--carry` (turn, walk to table B, put down), round 4 (12 per shape, development seeds): box lifted 10, carried 9, placed 8, upright 6; cylinder lifted 5, carried 5, placed 3, upright 2. Full success 8/24, 0 falls. Balls were dropped from the task here because they roll off a flat table. Bugs found: the 19.2 s episode limit reset robots mid-carry; table B had to be widened because robots drift about 0.3 m sideways.
- Parking the arms down made 7/12 robots fall: AGILE reads arm joint angles and cannot balance far outside its training pose. Matched hand drives: 9/24 vs 8/24, not adopted.
- Note: all walk_grasp numbers before 2026-10-22 used one object size per shape (see section 6).

**Fixed-base grasp policy on the walking robot.** dex-v6b zero-shot on a free-base walking G1: lifted 0/72, 0 falls; the hand never got nearer than about 0.18 m (median). 100- and 200-iteration fine-tunes on the walking base (v6b and v7 starting points): 0/72 lifted in each. The fixed-base grasp does not carry over to the walking base, and short fine-tunes barely move it.

### 4.3 Isaac Lab's locomanipulation imitation pipeline (2026-10-16 to 2026-10-21)

Stock steering-wheel task: NVIDIA's annotated G1 demos, Isaac Lab Mimic to generate 1,000 demos, robomimic BC-RNN (BC with a recurrent network), and `locomanipulation_sdg` to add walking between tables.

- **Manipulation BC** (2000 epochs, L1 loss 0.0077 to 0.000033): 45/50 rollouts succeed (Isaac Lab docs report 75-85 %).
- **Navigation data generation**: crashed at startup in the RTX renderer whenever a camera or the mobility_gen extension pulled in the viewport. Loading only mobility_gen's path planner and dropping the camera fixed it: 10/10 placed on a randomly placed end table, robot upright 10/10.
- BC pick + scripted walk back 0.4 m + BC place: lifted 10/10, held 10/10, placed 5/10. Controls: pause 6 s without walking 7/10; walk with RNN reset 5/10. Most of the loss comes from interrupting BC at all.
- **Navigation BC** (215 of 233 generated demos kept, 1000 epochs): first reported 4/20, but the judging was flawed twice. (1) An early stop by the environment's own drop term was missed because observations were read after the auto-reset. (2) "On the end table's footprint above 0.5 m" also passed with the wheel still in the hand. Re-scored with the stock placement check moved to the end table: **0/20**. Early drops were one-handed grasps (right wrist never comes in).
- Hybrid (manipulation BC picks, navigation BC walks): 0/20; walk commands went nowhere once the navigation BC no longer drove the arms. Removed.
- **Modular** (manipulation BC picks, the generator's pure-pursuit path follower walks, manipulation BC places): with the waypoint tolerance at 0.1 m the robot parked 0.10-0.14 m short (about 0.1 m/s commands are below what AGILE acts on). At 0.2 m: lifted 19, reached the goal 19, **placed 5/20** (strict check). With a looser "released at rest on the table" score: released 10/20. A calmer place start (standing still, ramping wrist targets) did not help. Not bit-for-bit repeatable across jobs: +/-2 of 20 is noise.
- Retraining the place BC on demos that start where the place starts: place-only tails released 28/50 (same within noise), whole demos 12/50 (worse). The BC-RNN resets its hidden state every 10 steps, so a "mid-task RNN reset" was never the issue.

---

## 5. Two-hand box task (2026-10-21 to 2026-10-22)

### 5.1 Mimic from the wheel demos: zero successes

- The wheel is lifted by the left hand alone at its near rim. Mimic on a solid box (slab 8.4 cm, thin tray 3 cm): 0 successes in about 2,400 tries each. Bug: the box's frame was at its centre while the wheel's is at its bottom, so the box sat half inside the table. After fixing it: still 0 of about 380 tries per shape. Open-loop replays: wheel 3/4, box rises 0-1.4 cm in all 8: the recorded hand hooks the wheel's thin rim, and a solid box gives it nothing to hook.
- Open-top crates (walls 1.2 cm): open-loop lifts 2/8 to 5/8 but none succeed; Mimic on a turned 27 cm crate 0 of 2,055 tries. Five shapes, 0 Mimic successes.

### 5.2 Scripted two-palm squeeze, standing (`isaac/box_scripted.py`)

- First tries 0/12: the open thumb stuck 6.2 cm inward and pushed the box away. Thumbs on the near face, fingers on the sides: lifted 5/6 to about 7 cm, none succeed; the wrists lagged because the box was out of comfortable reach. Centred carry, slower moves, hands 5 cm below the top: first real lift and carry.
- A stricter set-down check (box back at its start height in the goal region, slow, wrists 5 cm clear) replaced the stock check, which passed a box still held mid-carry. 4/6 succeed (box 0.22 x 0.14 x 0.20 m).
- Recorded demos: 227 of 400 succeed (57 %); 397/400 lifted, so misses are in the set-down. Later 593/1,024 (58 %).
- Batched robots with the stock Pink IK (CPU, one robot at a time): 1 robot 8.6 s per trial, 8 robots 4.2 s, 32 robots 2.9 s; success rate unchanged (64-75 %).
- **Batch IK** (`isaac/batch_ik.py`, one damped least-squares step for all robots from PhysX Jacobians, sim on the GPU), 1,024 robots in about 100 s: with the waist free, 429 fell; waist held and step capped at 0.02 rad, 520/1,024 succeeded (51 %), 10 fell. 0.1 s per try vs Pink's 2.9 s.
- Size sweep (scripted pick + set-down, 1,024 robots each; base 0.22 x 0.14 x 0.20 m = 51 %): 0.16 x 0.10 x 0.10 1024/1024; 0.30 x 0.20 x 0.30 72 %; 0.22 x 0.14 x 0.10 56 %; 0.22 x 0.20 x 0.20 55 %; 0.22 x 0.14 x 0.30 53 %; 0.16 x 0.14 x 0.20 53 %; 0.30 x 0.14 x 0.20 32 %; 0.22 x 0.10 x 0.20 28 % (thin boxes rise only about 6 cm and run out of time).

### 5.3 Box BC, standing

- BC on 227 scripted demos: 0/20 (loss 7e-7, so it fits the demos; the failure is drift). On closer look: 20/20 lifted and held, carried toward the goal but drifted 20-30 cm forward and never lowered.
- BC on 820 demos: 3/20, then 5/20 with a 1600-step horizon; several end in the goal region about 2 cm too high, still held.
- **BC carries, script sets down** (handoff after 100 steps): 10/20. Every failure was the same state: the box flung at reset. Cause: the arm controller starts from the previous rollout's pose and swings through the box on the first step. Holding the arms at their reset pose for 20 steps before putting the box back: no flings, **17/20**, 20/20 lifted.
- Wider start spread (+/-6 cm) data: the resulting BC scored 0/20 at the usual start and was not used.
- Size-aware BC (9 box sizes plus cylinders, object size as an input, 2000 epochs): 0/80 over four sizes with Pink IK, and 0 successes again after switching playback to the batch IK the demos were recorded with (20 per size; it lifted 8/20 only on the biggest box). The single-size BC control lifted 16/20 and handed off 7. Dead end.

### 5.4 Box RL, standing (`isaac/box_rl.py`)

PPO with AGILE legs frozen and batch IK; actions are wrist target moves plus a grip value; about 21,000 steps/s at 4,096 robots on one GPU.
- v1 (400 iterations): learned to grab, lift and carry over the goal; place 0 %. Carry was paid only while up, and nothing paid between carry and set-down.
- v2 (carry gated on "was lifted", new lower and release rewards): neither run lowers; lift and lower rewards cancelled over the goal.
- v3 (lift reward off over the goal, lower weight 8): place 0 % in both. Paused.

### 5.5 Walk with the box (`isaac/box_walk.py`)

Pick in the locomanipulation scene, walk a planned path to a second table, set down. Success: box at rest on the end table at its start height, wrists 5 cm clear.

- First rounds (9-12 trials): 1/9, then 0/12. Lifts miss 1 in 3 here vs about 1 in 100 standing. Variants (slower walk, wrists pulled in): no help. The misses repeat per scene.
- Turning stalled near the goal (commands under about 0.1 rad/s do nothing). A 0.5 rad/s floor and a 400-step limit: 3/12, 0 timeouts (baseline 2/12, 5 timeouts).
- Lowering by the box's measured height (round 6, 60 trials): **22/60 (37 %)**, 48 lifted, 39 reached. 12 of the reached failures had the box wedged low with the hands slid to its top.
- BC grab + walk (walk7, same seeds): 12/60. The "lifted" check counted a box tipped on its side (rise exactly 0.064 m).
- walk8: the new `held` check measured wrist-to-box-centre distance, but the grip puts the wrists about 0.15 m in front of the centre, so real grips failed it (BC 0/60, script 11/60). Fixed: the box centre must project between the wrists.
- walk9: script 18/60, BC 5/60. The scripted grab is the better base; of held boxes 18/25 were set down, so the grab is the bottleneck.
- walk10 (upright limit 60 deg): 20/60, 0 falls. Rejected boxes were tilted 82-180 deg. Lopsided starts (box about 0.2 m to one side) 0/17 vs centred 20/43.
- Side-stepping to centre the box first: 0/60 (walk11, with a 50-step settle), 0/60 again without the settle. A bisect found the cause: the centring loop reused a variable that held the grip half-width, so every centred trial squeezed at the wrong width. After the fix centring gave 16/60 vs 19/60 without, so it stays off. (An earlier explanation, base momentum after the side-step, was wrong.)
- Closed-loop squeeze (re-aim both hands at the box every step): 22/60; the left/right wrist imbalance is gone.
- Thumb offset 0.09 m: **31/60**, the best. Squeeze 0.02 m 18/60; both 25/60. Grip height at the box centre: 28-29/60; 3 cm below centre 16-18/60.
- The two-hand box is around 30/60 under every scripted tweak tried.

---

## 6. Mixed shapes at scale (2026-10-22)

- walk_grasp regression: the box carry had dropped to 1/12 placed with 7 falls. Batch IK was not the cause (Pink failed too). Defaults changed after round 4 (arms parked instead of raised, self-collision on, a carry pull-in) had caused it. With round-4 settings: Pink 12 robots placed 7, 0 falls; batch IK 128 robots placed 53/128, 9 fell, 3.6 min. Batch IK keeps the success rate and runs 128 robots per job.
- All shapes, 128 robots each: set down upright with the robot standing: box 33/128, cylinder 18/128, ball 6/128, lying cylinder 1/128, capsule 0/128. A slower put-down: ball 11/128, lying cylinder 3.
- **Two spawn bugs:** every env spawned env 0's object, so a run of 128 was one size 128 times; lying shapes were built standing on end. Fixed with one spawn config per env and shapes built along their axis. **All earlier walk_grasp numbers are one object size per shape.**
- Round and lying objects rolled during the walk because footsteps shake the table and PhysX has no rolling friction. Angular damping, 6 shapes x 16: damping 0 15/96, 5 18/96, 20 27/96. Default 20.
- **384 robots, 6 shapes x 64, each with its own size, one GPU, 5.2 min: set down upright with the robot standing 144/384 (38 %):** box 32/64, ball 24, lying cylinder 27, capsule 24, cylinder 21, egg 16. Damping 50: 146/384, the same. Eggs are approximated as capsules (Isaac Lab has no ellipsoid spawner).
- **Random layouts** (`walk_grasp.py --rand_layout`): each robot gets its own scene: table A 0.8-1.8 m ahead and up to 0.3 m sideways, table B at a 75-105 degree turn either side, 0.9-1.5 m from the turn spot. 6 shapes x 16, damping 20: 19/96 set down upright with the robot standing (box 5, cylinder 3, ball 3, lying cylinder 3, capsule 3, egg 2), 0 falls; the fixed layout re-checked at 27/96. Arriving (96) and turning (89) hold up; the loss is at the grab (held 37 to 25).

## 6.1 walk_carry as a fixed benchmark: scripted 92/192, v8 grab 119/192 on validation (2026-10-26 to 2026-10-28)

Goal (`scripts/goal_common.py`, `bench/README.md`): beat the 19/96 above on a benchmark fixed first, with the learned grab chosen on validation and test scored once. `isaac/bench_spec.py` holds the contract: disjoint seed splits (val 700000+, test 900000+, the historic walk runs at 960000+ stay test), and shared success tests: lifted = 9 cm up *and* not tipped (a box on its side reads ~6 cm up), upright by kind (standing: axis within 20 degrees of its start; lying: long axis from horizontal; balls always), released = no contact and a 4 cm gap. Every run prints one `BENCH_RESULT` record; `scripts/check_day1..3.py` read them on CPU.

- Fixes to the walk found by the audit and the first baselines: a sideways walking command toward the table's centre line with a heading hold (random layouts put the table up to 0.3 m to either side), table B 1.6 m wide in both layouts (random layouts had 1.2), the grasp goal read after the 1.5 s settle with the object's current heading (eggs still rolling at the arrival had been aimed at where they were), eggs spawned on their capsule stand-in's radius (wider-than-tall eggs sat inside the table and were kicked), and a levelled set-down: the hand turns about the pinch point until a standing object's axis is vertical, lowers to 2 mm above rest, holds 0.3 s, then opens (cylinders held 7-18 degrees tilted had landed on an edge and toppled: 5 of 9 placed). Every robot arrives and turns; 0 falls.
- **Noise floor.** The same scripted code on the same 96 validation layouts: cluster RTX 8000 40/96, RTX 3090 41/96, 9 layouts differ; a zero-adjustment learned-grab run vs scripted on one machine: 45 vs 41, 9 differ. GPU PhysX is not deterministic. Records carry the host; comparisons are paired per machine, and a margin under ~10 layouts at n = 96 is noise.
- **Validation, 6 shapes x 32 layouts, RTX 3090** (`bench/results/`): scripted 92/192 (48 %); the v8 grasp-parameter policy (3.5) applied to the scripted pinch from where each robot stops, with the 0.4 cm perception error it trained with: **119/192 (62 %)**; only-v8 41 layouts, only-scripted 14, McNemar p = 0.0004. By shape (scripted / v8 of 32): box 19 / 27, cylinder 20 / 29, lying cylinder 7 / 10, ball 16 / 19, capsule 13 / 16, egg 17 / 18; boxes and cylinders carry the gain, round shapes are a wash. v8 was trained at a fixed base 0.11 m from the table; the walk stops at 0.18 m, and it transferred anyway. `bench/selection.json` names v8.
- **Test, once (2026-10-29, cluster, 6 shapes x 16 layouts):** v8 **48/96 (50 %)**, scripted 41/96, on the same layouts; only-v8 18, only-scripted 11, McNemar p = 0.265. Above the development best 19/96, so all three goal days pass (`scripts/check_goal.py`); on test the v8 margin over scripted is within the noise floor.
- Scripted per-stage losses on validation (where the rest is): lying cylinders are missed by the pinch itself (hand on target, object untouched); balls fall out while backing off or walking; cylinders that tipped at set-down are fixed by the levelling (upright-on-table 8 to 15 of 16).

## 6.2 The closed-loop grasp (dex-v6b) on the walking base: the base gets kicked (2026-10-27 to 2026-10-28)

- `g1_dexsuite.py eval --probe` and `isaac/obs_compare.py`: the same checkpoint on the same 72 validation layouts, fixed base 65/72 lifted, walking base 0/72. The observations match at the start (object in the pelvis frame within 3 cm). Within 0.1 s of the first actions the pelvis pitches 0 to 22 to 26 degrees and sinks, the base drifts 29 cm in the first second, and the hand never leaves its start height in the pelvis frame (16 cm from the object throughout). On the fixed base the same actions move the hand 19 cm in 0.1 s: motions learned on a welded base knock AGILE's standing base over. Dynamics, not perception; why earlier fine-tunes (4.2) had no reward to learn from.
- Action-side knobs (`--freeze_waist`, `--act_clip`, `--ema`): waist frozen, ema 0.4 (A40, 72 layouts): pelvis kick 11 degrees median (p90 13.7), closest pinch-to-object 0.139 m, lifted 0/72. Halves the kick; the hand still stops 14 cm short.
- `isaac/auto_rl.py` (docs/auto_rl.md): the pipeline that tries the remaining option, fine-tuning on the walking base (warm start, random carry motions of the legs during training, plateau/divergence stop, selection on validation, the end-to-end task in the same env), inside a 10 GPU-hour budget. Tested on fake logs; on the cluster only its first stage has run (2026-10-29: the three knobs settings, all 0/72 lifted, closest pinch 13-14 cm, so none reached the 8 cm the stage asks for); training not run.

---

## 7. Negative results (index)

Kept because they save others the time. Details in the sections above.

- SmolVLA pick from cameras: 0/20; with an exact target hint at best 25/40 (2.1).
- Two-camera box agreement, 2 px mask trimming, newer single-image depth models, SAM 3: no gain (2.3).
- Forced answer schema for the planner: hurts, most with thinking (2.4).
- Whole-picture object listing with an LLM; SigLIP shortlist + Gemma pick; PAM-3B, Rex-Omni (2.5).
- Untrained Gemma as a decision model never says "done" (2.6).
- Unitree teleop "wrap" closing in the MuJoCo sampler: 0/545 (3.2).
- Residual REINFORCE on the grasp chooser; chooser v4 over v3 not significant (3.2).
- MuJoCo Warp for G1 grasping: no noslip solver, fidelity no-go (3.3).
- Copies per layout, more grasp parameters (v6 on test), retry variants, joint-limit-aware IK, width pinch for learned policies (v9) (3.5).
- Dexsuite dex-v1 to dex-v5 before smoothing; LCP smoothing (3.6).
- Fixed-base Dexsuite grasp on the walking base: 0/72 zero-shot and after fine-tuning (4.2).
- Dex3 walking from scratch; Unitree's released walking policy zero-shot (4.1).
- Navigation BC alone (0/20 strict), hybrid BC (0/20), place BC retrained on place-start demos (4.3).
- Mimic from wheel demos to five box and crate shapes: 0 successes (5.1).
- Box RL: never learned to set down (5.4).
- Size-aware box BC: 0 successes with either IK (5.3).
- Walk-with-box: settling before the pick, slower walking, wrists pulled in, centring by side-stepping (5.5).

## 8. Bugs that changed results

- IK stalled with a joint pinned at a limit (1).
- Unseeded policy noise in SmolVLA evaluation: +/-3 of 40 between runs (2.1).
- Jaw-down IK weight schedule pushed far cubes (2.2).
- SAM 3 loaded with a random mask decoder via plain `from_pretrained` (2.3).
- Masks over known fixtures accepted as objects after calibration (2.3).
- Hand-close rounding (about 1e-17, tested with `==`) and thumb-reading hold check hid good G1 grasps (3.2).
- Isaac: objects spawned at twice the env offset; self-collision off by default; hand camera-mount collider fighting the palm (3.4).
- Isaac: about 10 % of objects knocked off or through the table by the reach in every evaluation (3.5).
- Isaac: eggs simulated as spheres; a waist 20x stronger than the real G1 (3.5).
- Dexsuite: objects launched at reset; upstream point-cloud noise maximum equal to its minimum (-0.01); hand points at link origins 5.2 cm behind the finger tips; one validation run scored on test layouts (3.6).
- Dex3 whole-body USD: zero-mass sensor bodies, about 6.7 kg phantom mass (4.1).
- Episode time limit resetting robots mid-carry (4.2).
- Navigation evaluation: drops missed after auto-reset; in-hand wheel counted as placed (4.3).
- Box frame at its centre vs the wheel's at its bottom (5.1).
- Box flung at reset by the arm controller's leftover pose (5.3).
- `held` check measured the wrong distance; centring loop overwrote the grip half-width (5.5).
- walk_grasp: every env spawned env 0's object; lying shapes built on end; changed defaults caused a regression (6).
- walk_grasp: the grasp goal read before the settle; random-layout table B narrower than the fixed one; Kit exiting 0 after a crash so a failed job looked finished (6.1).
- g1_dexsuite on the walking base: a legs action term with 4 dims made the policy's action and observation sizes differ from its checkpoint (6.2).
