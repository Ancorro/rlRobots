# Stack v1 (SO-101, simulated tabletop)

Finalised 2026-10-09; robot: SO-101 only. Every layer was chosen and measured
on its own (see docs/results.md); v1 connects them into one loop driven by a free-text request and measures the
whole chain. Walking, houses and the G1 are v2.

## Layers (frozen for v1)

| Layer | Component | Measured alone |
|---|---|---|
| 3 Planner | Gemma 4 12B, 4-bit (Ollama `gemma4:12b`), thinking on | 99.4 % of 3,600 held-out requests (scene text from the simulator) |
| 2 Decision | JEV-9B untuned (`autotrust/JEV-9B`, its `vl/serve_decide.py`, POST /v1/decide), object options restricted to what the skill accepts | tabletop 49/50, sort 50/50 |
| 1 Perception | Grounding DINO tiny ("object.", see.py threshold) -> SAM 2.1 large mask -> DAM-3B names the region (no name list) -> depth -> 3D points | boxes 99.8 / 99.6 % found; names 92.4 / 78.0 % found, 26 / 38 % wrong (simple / household) |
| 0 Skills | scripted pick (grasp planned from the object's camera points) and place on a free spot (bench_general.py) | 386/400 simple, 350/400 household (task names given) |

## The loop (`stack_v1.py`)

1. Scene: 3-5 objects (simple shapes or household), the red tray, the blue tray, the shelf; a request from the planner
   benchmark's families (`so101/planner.py requests`) with its exact goal, or None when it cannot be done.
2. Look: the arm parks, the cameras list every object (perception above): id p1, p2, ..., DAM's short name, position
   (cm), where it is (table, red tray, blue tray, shelf). The simulator's state is never read except to score.
3. Plan: the planner reads the request and the perceived scene (planner.prompt, perceived names instead of true ones)
   and answers the goal as moves "p2 -> red_tray", or "cannot".
4. Decide: JEV-9B reads the goal, the perceived scene, what the gripper holds and the last skill's outcome, and picks
   a skill (pick, place, done) and its object (a perceived id, or a place). At most 2 x moves + 4 skills.
5. Act: pick uses the chosen object's points from the latest look; place puts it on a free spot of the place.
6. Look again after each skill; ids carry over to the nearest object seen before (within 4 cm), new ones get new ids.
7. Score (simulator truth): success = every goal object at its place and no other object moved more than 2 cm (an
   impossible request succeeds when the planner says cannot and nothing moves). Each failure gets one cause: perception
   (object missed or misnamed so the plan was wrong), planner (wrong goal on the perceived scene), decision (wrong
   order, gave up, or stopped early), skill (pick or place failed on a correct choice).

## Pre-registered test (fixed before the first run)

- Frozen 2026-10-09 after development on seed 400000 (12 episodes, 8 done): the loop adds, from docs/plan.md, skill
  options limited by the gripper (pick when empty, place when holding) and escalation: a skill choice below 0.7
  goes to the planner, which re-plans from the current perceived scene (at most twice per episode).

- Held-out scenes: seed 420000 (never run; seed 400000 became the development set after its first 6 episodes were
  used to debug the loop, 2026-10-09), 100 episodes: 50 simple shapes, 50 household, one
  request each, request families in turn (planner.requests order), 3-5 objects.
- Headline: episodes where the request was carried out, of 100, with the failure causes. Also per layer: perception
  found / wrong names, plan right on the perceived scene, plan right on the true scene (oracle names), skills done.
- Reference runs on the same episodes, to locate the losses: (a) true scene text to the planner (oracle perception),
  (b) the rule decider instead of JEV-9B. Neither is the headline.
- All objects start on the table (the planner benchmark's tray / shelf starts are not built physically), so the
  families that need them (clear, swap, already) do not occur: single, two, all-but, superlative, category,
  not-category, sort, conditional, impossible.
- Compute: OSU cluster only, two GPUs (dgxh): JEV-9B alone on one (vLLM, 0.85 of it); Gemma (~10 GiB), DAM (~7 GiB),
  DINO + SAM (~4 GiB) on the other. On one 48 GB card they ran out of memory.
