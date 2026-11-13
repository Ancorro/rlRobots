# Stack v2 (2026-10-09)

More architecture on top of v1, which fails on longer tasks (objects to move: 1 → 32/47 done,
2 → 19/38, 3+ → 1/5) and on linking the request's words to objects (21 of 38 failures: DAM names like "four-sided cube"
for "black block"; the planner was right on 100/100 with true names).

## Architecture

| Layer | v1 | v2 |
|---|---|---|
| Perception | DINO + SAM 2.1 + depth | same |
| Grounding | DAM-3B free-text names | request nouns matched to object crops (VLM), short colour + kind names as fallback |
| Scene memory | none (ids carried to the nearest object within 4 cm) | `so101/memory.py`: tracks with stable ids, appearance, position, place, status, history; action-predicted positions; Hungarian matching; missing / drop flags |
| Task planner | Gemma 4 12B: moves | same model; ordered subgoals for big requests |
| Executive | none (fixed loop, escalate below 0.7) | Gemma 4 12B (same server, own prompt): continue / retry / re-plan / done / cannot, from structured state; rules in code: at most 2 retries per object, "done" only when the goal check on memory passes |
| Decisions | JEV-9B | same |
| Verifier | none | code: goal predicates on scene memory after each step |
| Skills | scripted SO-101 | + slow first lift (default), re-grasp after a drop |

Later (not v2): a sim data engine (free ground truth) to fine-tune grounding and decisions on tagged failures.

## Re-identification check (before memory goes into the stack)

`reid_bench.py`: SO-101 scenes with look-alike twins; 6 events per episode (move, drop told as a move, knock, hide one
look, swap twins, remove), a look after each; perception cached, every method replays the same looks.
Methods: v1 (nearest within 4 cm), position (with action prediction), appearance (SigLIP 2 / DINOv2 crops), both.
Pre-registered: settings for "both" tuned on dev seeds 600000+ (40 episodes, grid); one test run on seeds 620000+
(100 episodes). Pass: "both" (dev pick) beats "position" on identity accuracy for twins and moved objects, paired
McNemar p < 0.05, and flags drops at least as often with no more false "missing".

### Dev result and pick (2026-10-10, 40 episodes, 1,338 detections)

| Method | Identity | Twins | Moved | Switches | Extra tracks | False missing | Removed found |
|---|---|---|---|---|---|---|---|
| v1 (4 cm) | 0.755 | 0.792 | 0.443 | 117 | 79 | 87 | 0.975 |
| position | 0.889 | 0.922 | 0.750 | 67 | 18 | 35 | 0.775 |
| appearance (SigLIP 2) | 0.857 | 0.807 | 0.733 | 95 | 2 | 17 | 0.875 |
| both, defaults (SigLIP 2) | 0.958 | 0.972 | 0.909 | 39 | 3 | 19 | 0.85 |
| both, defaults (DINOv2) | 0.920 | 0.956 | 0.824 | 50 | 21 | 33 | 0.925 |
| **both, dev pick** | **0.979** | **0.991** | **0.960** | 31 | 3 | 15 | 0.95 |

Pick (highest identity, then fewest switches, then first in grid order): SigLIP 2, pos_scale 3 cm, app_scale 0.1,
gate 3, re-link gate 2. Several settings tie; it is flat near the top. Calibration on dev pairs: at P(same) = 0.5,
4.6 % of different pairs accepted, 5.2 % of same pairs rejected. Test: seeds 620000+, 100 episodes, once.

### Test result (2026-10-10, seeds 620000+, 100 episodes, 3,441 detections): PASS

| Method | Identity | Twins | Moved | Switches | Extra tracks | Drops flagged | False missing |
|---|---|---|---|---|---|---|---|
| v1 (4 cm) | 0.776 | 0.780 | 0.501 | 263 | 181 | 0.91 | 205 |
| position | 0.912 | 0.906 | 0.810 | 128 | 42 | 0.91 | 77 |
| **both, dev pick** | **0.984** | **0.981** | **0.975** | 50 | 4 | 0.98 | 33 |

Paired McNemar vs position: twins 168 vs 4 (p = 1e-44), moved 74 vs 1 (p = 4e-21). Drops flagged more often, fewer
false "missing". Memory goes into the stack.


## Grounding check (before grounding goes into the stack)

`ground_bench.py`: stack v1's scenes and request families, its perception (DAM names, crops) and its Gemma planner, run
under five namings of the same perceived objects: dam (v1), true (ceiling), siglip@gate (request phrases matched to
crops by SigLIP 2 giant-384; gates 0.01 / 0.03 / 0.1 / 0.3), gemma (closed choice per crop: a request phrase or "none
of these"), vision (the planner sees each object's crop). Phrases: Gemma extracts kinds, colours, categories from the
request. Score: plans right (planner.check), paired against dam.
Pre-registered: dev seeds 640000+ (60 scenes) pick one naming (most plans right; ties go to the simpler, in the order
above); one test run on seeds 660000+ (100 scenes): the pick must beat dam, paired McNemar p < 0.05.

### Grounding test result (2026-10-13, seeds 660000+, 100 scenes, job 21813312): PASS

Plans right: true 99, **vision 91**, siglip@0.3 74, siglip@0.1 71, dam 67, gemma 58. Vision vs dam 24 vs 0, exact McNemar
p = 1e-7. Dev pick held (dev 55/60). Vision goes into the stack: `stack_v2.py --grounding=vision` (default) gives the
planner and its re-plans each tracked object's best crop (ground_bench.best_crop / vision_plan); JEV still sees DAM names.

## Stack v2 with grounding: dev check and final test (pre-registered 2026-10-14; larger sets)

- Dev (no pass rule; finds bugs before the freeze): seed 400000, episodes 0-59, v2 `--grounding=dam` and `--grounding=vision`,
  same episodes; report requests done, plans right, causes, paired. Code copy stack2d.
- Final test, once v2 is frozen (executive in, or dropped): seed 420000, episodes 0-199, one run.
  v1 (frozen stack1 copy) also runs episodes 100-199 to join its 0-99 test; those v1 results are not read before v2 freezes.
  Headline: requests done, v2 vs v1, paired over 200, exact McNemar, pass p < 0.05. Secondary: by family, causes, plans right.
