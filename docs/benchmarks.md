# Benchmarks and object sets

What we can score against, and with which objects (2026-10-27). Numbers and dates are in `docs/results.md`; this file
lists only what exists. Seeds are layout seed bases; "held-out" sets are scored once per selected method.

## Object sets

| Set | Where | Size | Used by |
|---|---|---|---|
| Primitives, G1 | `isaac/g1_grasp_rl.py` `SHAPES`, `isaac/walk_grasp.py` | 6 shapes: box, cylinder, lying cylinder, ball, capsule, egg (a capsule with an egg's length and width); random size, mass, friction per layout | G1 fixed-base grasp, walk and carry |
| Primitives, SO-101 | `so101/objects.py` | random shape, size, colour, mass and friction, sized for the SO-101 jaw | SO-101 pick-and-place, stack v1/v2 |
| Two-hand box | `isaac/box_cfgs.py`, `BOX_SIZE` | one box, 0.22 x 0.14 x 0.20 m by default (sized variants used for box BC) | `box_scripted.py`, `box_walk.py`, `box_bcplay.py`, `box_rl.py` |
| GSO, SO-101 | `so101/assets/objects/gso_index.json`, meshes in `gso/` | 50 Google Scanned Objects (CC-BY 4.0) with phrases for language requests | SO-101 household scenes, stack v1/v2, re-ID |
| GSO, G1 one-hand (household) | `so101/assets/objects/gso_g1.py`, `gso_g1_index.json`; hulls `gso_g1/hulls.npz` | 114 kept of 1,030 measured: narrowest width 1-5 cm, height 1.5-15 cm, longest <= 20 cm, mass <= 0.5 kg | G1 household benchmark (`g1_grasp_rl.py` / `g1_dexsuite.py --objects=household`) |
| Object pool, G1 | `so101/assets/objects/pool_g1.py`, `pool_cluster.sbatch` | GSO 114 + DexGraspNet 557 = 671 (train / val / test 522 / 77 / 72, split by object-name hash, not by source) | `g1_dexsuite.py --objects=pool` (dex-v7) |
| Random GSO draw | `so101/assets/objects/random_set.py`, `random_set.json` | 40 one-hand + 40 two-hand (longest 20-50 cm, width >= 8 cm, mass <= 3 kg), seed 0 | drawn, not yet used in a benchmark |
| Not ours yet | `docs/environments.md` | ShapeNetSem (access granted, on the cluster), ACRONYM, HSSD-200 (no access), PartNet-Mobility | none |

## G1 (Isaac Lab)

| Benchmark | Script | Objects | Size | Seeds | Scored as |
|---|---|---|---|---|---|
| Fixed-base lift | `isaac/g1_grasp_rl.py eval`, `isaac/g1_dexsuite.py eval` | primitives | 6 shapes x 60 = 360 | val 700000+, test 900000+ | lifted >= 9 cm, held, kept after a shake |
| Fixed-base household | same, `--objects=household` | GSO one-hand 114 x 3 poses = 342 | 342 | 950000+ | as above |
| Fixed-base pool | `g1_dexsuite.py --objects=pool` | pool val 77 / test 72 objects | e.g. 720 | train 810000, val 710000, test 970000 | as above |
| Walk, pick, carry, place | `isaac/walk_grasp.py --carry` | primitives, one hand | 6 shapes x N (16 = 96, 32 = 192, 64 = 384) | `isaac/bench_spec.py` splits: val 700000+, test 900000+ (the earlier 960000+ runs fall in test) | set down upright on table B, released, robot standing |
| Same, random layouts | `walk_grasp.py --carry --rand_layout` | primitives | 6 x N | as above | as above; table A 0.8-1.8 m, B at a 75-105 deg turn |
| Same, scanned or pooled objects | `walk_grasp.py --carry --rand_layout --objects=gso` or `--objects=pool --pool DIR` | GSO one-hand (val 6, test 11 objects) or the pool's one-hand objects (val 58, test 60) | objects x `--n` poses (val: 6 x 16 = 96; 58 x 4 = 232) | seeds BASE + 100 x object + pose | as above, judged at the object's pinch point and centre; not part of the goal checks |
| Grab alone, walking base | `isaac/grab_rl.py eval` | primitives | 6 x 64 = 384 | walk_carry val layouts, robot placed 13.1 cm from the edge | lifted 9 cm, within 5 cm of the pinch point, robot up; balls not tilt-tested |
| Grab alone, walking base, scanned or pooled objects | `isaac/grab_rl.py eval --objects pool --pool DIR --n 4` (or `--objects gso --n 16`) | object pool or household scans | 58 x 4 = 232; 6 x 16 = 96 | walk_grasp.py's layouts of the set, robot placed 11.5 cm from the edge | lifted 9 cm, the object's grasp point within 5 cm of the pinch point, robot up |
| Two-hand box carry | `isaac/box_walk.py` via `sdg_cluster.sbatch STEP=boxwalk` | two-hand box | 60 trials | `--seed` (walk15-th09: seeds 31-41) | picked, walked, set down upright, released |
| Two-hand box, standing | `isaac/box_scripted.py`, `box_bcplay.py`, `box_rl.py` | two-hand box (and sizes) | 20-80 rollouts | per script | lifted, handed off, placed |

## SO-101 (MuJoCo)

| Benchmark | Script | Objects | What it scores |
|---|---|---|---|
| General pick-and-place | `bench_general.py` | primitives + GSO 50; red tray, blue tray, 6 cm shelf | camera-only sensing + scripted grasp, "put the <colour> <noun> <place>", up to 3 attempts |
| Planner (no physics) | `plan_bench.py` | same scenes | LLM plans checked against each request's exact goal |
| Grounding | `ground_bench.py` | stack v1 scenes (dev seed 640000) | request words matched to object crops vs DAM names |
| Perception probe | `percept_probe.py` | `ground_bench.py` dev scenes, 60 | planner given vision / names hidden / two views / true names |
| Re-identification | `reid_bench.py` | 3-5 objects plus look-alike twins | scene memory re-ID across moves |
| Full stack | `stack_v1.py`, `stack_v2.py` | primitives + household, 50/50 | requests done end to end (test seed 420000, 100 or 200 episodes) |

## OpenHLM-data (real G1 teleoperation recordings, offline)

| Benchmark | Script | Data | Size | Split | Scored as |
|---|---|---|---|---|---|
| Open-loop action prediction on held-out episodes | `openhlm/pipeline.py` (`openhlm/run.sbatch`), contract `openhlm/spec.py`, check `scripts/check_openhlm.py` | `OpenHLM/OpenHLM-data`: 5 LeRobot sets of a real Unitree G1 in three recording modes (`full_teleop`, `stationary_teleop`, `humi`); by their names an `HLM-12` task set and a long five-fruit task | 20 test episodes (4 per set), the check's sample | by a salted hash of set/episode: per set 4 test, ~15 % val, rest train | per episode: 1 s action chunks from 1/3 s of state history, error normalised by the training std, success = error at most half the hold baseline's (skill >= 0.5); `docs/openhlm.md` |

## Checks

`scripts/check_day1.py`, `check_day2.py`, `check_day3.py` check the goal days (benchmarks fixed, learned grab chosen on
validation, held-out test; `check_goal.py` runs all three). The goal is walk_carry only; box_carry and household records are kept but not checked. Pair
learned and scripted runs by layout seed on one machine: GPU physics differs across machines (9 of 96 layouts on
2026-10-26).
