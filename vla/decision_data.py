"""Decision examples with known right answers for tuning a decision model (owner 2026-10-21: Gemma in decision mode +
a LoRA used only for decisions). No physics: stack v1's scene and request generator gives objects and a goal; a short
simulated run then walks through it with stack_v1.Decide, the exact state text and questions the stack shows its
decision model, and Decide.oracle_action (the rule decider, 50/50 on sort) labels each step.

Each step is one example: one question over the step's valid (skill, object) pairs, the way the Gemma deciders ask it.
Outcomes are drawn so the hard states appear: grasp fails, drops after a "done" pick (gripper empty, object back on the
table), places that land on the table, objects knocked off their place, objects missing from view. Names: the true name
or, half the time, a perception-style name drawn from --names (DAM-3B names seen on dev runs) so the model learns to
follow ids, not words. Option order is shuffled per example when --shuffle (training), kept as the stack shows it
otherwise.

  python vla/decision_data.py --seed=800000 --scenes=800 --out=runs/decide/train.jsonl --shuffle=1 [--names=names.json]
Seeds: training 800000+, held-out 810000+ (stack dev 400000, test 420000, never used here).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from so101 import planner  # noqa: E402
from so101.objects import OBJECTS_DIR, layout, random_objects, scanned_objects  # noqa: E402
from so101.deciders import joint_question  # noqa: E402
from stack_v1 import ARGS, Decide  # noqa: E402

SEED, N = int(ARGS.get("seed", 800000)), int(ARGS.get("scenes", 800))
OUT, SHUFFLE = Path(ARGS.get("out", "runs/decide/train.jsonl")), ARGS.get("shuffle", "0") == "1"
POOL = json.loads(Path(ARGS["names"]).read_text()) if "names" in ARGS else []
CENTRE = {p: np.array(xy) / 100 for p, (_, xy) in planner.PLACES.items()}


def scene(rng, index):
    scanned = bool(rng.integers(2))
    for _ in range(50):
        n = int(rng.integers(3, 5 if scanned else 6))
        specs = scanned_objects(rng, n, strict=False) if scanned else random_objects(rng, n)
        try:
            poses = layout(rng, specs, tries=200, max_r=0.23)
            break
        except RuntimeError:
            continue
    objs = planner.make_scene(specs, poses, index)
    reqs = planner.requests(objs, rng, scanned)
    fam, _, goal = reqs[int(rng.integers(len(reqs)))]
    ids = [f"p{k + 1}" for k in rng.permutation(len(objs))]
    names = {i: (o["name"] if not POOL or rng.random() < 0.5 else str(rng.choice(POOL))) for i, o in zip(ids, objs)}
    seen = {i: dict(centre=np.array([o["x"] / 100, o["y"] / 100, 0.02]), where=o["where"]) for i, o in zip(ids, objs)}
    to_id = {o["id"]: i for i, o in zip(ids, objs)}
    moves = [(to_id[k], p) for k, p in (goal or {}).items()]
    rng.shuffle(moves)
    return fam, names, seen, [tuple(m) for m in moves]


def table_spot(rng):
    return np.array([rng.uniform(0.08, 0.26), rng.uniform(-0.2, 0.2), 0.02])


def examples(d, rng, meta):
    """The step as one question over its valid (skill, object) pairs (so101/deciders.joint_question), as the Gemma
    deciders ask it: one forward pass per step."""
    opts = {q: list(s["criteria"]) for q, s in d.questions.items()}
    a = d.oracle_action()
    skill = opts["skill"][a[0]]
    target = opts["object"][a[1]] if skill != "done" else None
    spec, names, pairs = joint_question(d)
    answer = names[pairs.index((skill, target))]
    if SHUFFLE:
        names = [names[k] for k in rng.permutation(len(names))]
    out = [] if len(names) < 2 else [dict(meta, q="step", skill=skill, state=d.state_text(), instructions=spec["instructions"],
                                          criteria={n: spec["criteria"][n] for n in names}, names=names,
                                          answer=names.index(answer))]
    return out, skill, target


def run(k, index):
    rng = np.random.default_rng([SEED, k])
    fam, names, seen, moves = scene(rng, index)
    held, last, steps, cap, rows = None, "none yet", 0, 2 * len(moves) + 4, []
    while steps < cap:
        d = Decide(moves, names, {i: o for i, o in seen.items() if not o.get("missing")}, held, last, steps, cap)
        ex, skill, target = examples(d, rng, dict(scene=k, family=fam, step=steps))
        rows += ex
        if skill == "done":
            break
        steps += 1
        r = rng.random()
        if skill == "pick":
            if r < 0.15:
                last = f"pick {target}: failed"
                seen[target]["centre"] = seen[target]["centre"] + np.r_[rng.normal(0, 0.01, 2), 0]
            elif r < 0.25:  # a drop the stack notices only from the empty gripper
                last, seen[target]["centre"], seen[target]["where"] = f"pick {target}: done", table_spot(rng), "table"
            else:
                held, last = target, f"pick {target}: done"
        else:
            if r < 0.12:
                last, seen[held]["centre"], seen[held]["where"] = f"place {target}: failed", table_spot(rng), "table"
            else:
                last, seen[held]["where"] = f"place {target}: done", target
                seen[held]["centre"] = np.r_[CENTRE[target] + rng.normal(0, 0.015, 2), 0.02]
            held = None
        r = rng.random()
        others = [i for i in seen if i != held]
        if r < 0.06 and others:  # something knocked: back on the table elsewhere
            i = others[int(rng.integers(len(others)))]
            seen[i]["centre"], seen[i]["where"] = table_spot(rng), "table"
        elif r < 0.09 and others:  # missed by the look this time
            seen[others[int(rng.integers(len(others)))]]["missing"] = True
        else:
            for o in seen.values():
                o.pop("missing", None)
    return rows


def main():
    index = json.loads((OBJECTS_DIR / "gso_index.json").read_text())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    total, by = 0, {}
    with OUT.open("w") as f:
        for k in range(N):
            for r in run(k, index):
                f.write(json.dumps(r) + "\n")
                total += 1
                key = r["skill"]
                by[key] = by.get(key, 0) + 1
    print(f"wrote {OUT}: {total} examples from {N} scenes; {dict(sorted(by.items()))}", flush=True)


if __name__ == "__main__":
    main()
