"""Perception probe for stack v2 (2026-10-26): v2's perception failures find every object (no junk) but the planner
still picks the wrong one (DAM calls a red cube "orange"; two similar boxes swap). Does showing the planner the
objects differently fix it? Same scenes, perception and Gemma 4 12B planner as ground_bench.py; planner only.

Variants (all see crops, as stack v2's --grounding=vision):
  vision         v2 as it is: one best crop per object, DAM names in the scene text
  noname         one crop, DAM names hidden (every object "object pN"): the picture is the only evidence
  twoview        best crop from each camera (up to two per object), DAM names
  twoview-noname two crops, names hidden
  true           text only, true names (ceiling)
Score: plan right (planner.check on the true scene), paired McNemar against vision. No pass rule.

  python percept_probe.py --seed=640000 --first=0 --episodes=30 --tag=dev-a   (SCRIPT=percept_probe.py vla/stack.sbatch)
  python percept_probe.py --score --tag=dev                                     (joins runs/percept/dev*.jsonl)
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

from ground_bench import mcnemar, png
from stack_v1 import ARGS, FAMILIES, SAME, Namer, look, ollama_plan

OUT = Path("runs/percept")
TAG = ARGS.get("tag", "dev")
VARIANTS = ["vision", "noname", "twoview", "twoview-noname", "true"]


def crop(view):
    """A view's object, padded and upscaled to >= 224 px (ground_bench.best_crop for one view)."""
    from PIL import Image
    img, _, b = view
    x0, y0, x1, y1 = b
    pad = 0.15 * max(x1 - x0, y1 - y0) + 4
    x0, y0, x1, y1 = np.clip(np.round([x0 - pad, y0 - pad, x1 + pad, y1 + pad]).astype(int), 0, [img.shape[1], img.shape[0]] * 2)
    c = Image.fromarray(img[y0:y1, x0:x1])
    k = 224 / max(c.size)
    if k > 1:
        c = c.resize((round(c.size[0] * k), round(c.size[1] * k)), Image.BICUBIC)
    return np.asarray(c)


def area(v):
    b = v[2]
    return (b[2] - b[0]) * (b[3] - b[1])


def crops_of(o, two):
    """The largest view, or with two the largest view from each camera (views carry their image: match by identity)."""
    views = sorted(o["views"], key=area, reverse=True)
    if not two:
        return [crop(views[0])]
    out, seen = [], []
    for v in views:
        if not any(v[0] is s for s in seen):
            seen.append(v[0])
            out.append(crop(v))
    return out[:2]


def plan_with_pictures(pobjs, crops, request, named):
    from so101 import planner
    msgs = planner.prompt(pobjs, request)
    images, order = [], []
    for o, cs in zip(pobjs, crops):
        images += [png(c) for c in cs]
        order += [o["id"]] * len(cs)
    note = ("Names come from a camera and may be wrong; trust the pictures over the names." if named else
            "The objects are not named: tell them apart by the pictures.")
    multi = " (some objects have two pictures, from two cameras)" if len(order) > len(pobjs) else ""
    msgs[-1] = dict(msgs[-1], images=images, content=msgs[-1]["content"] +
                    f"\n\nThe pictures show the objects' close-ups{multi}, in this order: {', '.join(order)}. " + note)
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    host = host if host.startswith("http") else "http://" + host
    body = dict(model="gemma4:12b", messages=msgs, stream=False, think=True,
                options=dict(temperature=1.0, top_p=0.95, seed=0, num_ctx=16384, num_predict=8192))  # vision_plan's
    q = urllib.request.Request(host + "/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(q, timeout=3600) as f:
        o = json.loads(f.read())
    return None if o.get("done_reason") == "length" else planner.parse(o.get("message", {}).get("content", ""))


def scene(e, holder, namer):
    """ground_bench.scene's scene, request and look; then the planner under each variant."""
    import mujoco

    from so101 import planner
    from so101.objects import OBJECTS_DIR, body_name, build, layout, random_objects, scanned_objects
    from so101.robot import SO101, Guard, SimArm
    from so101.skills import Arm
    from stack_v1 import PARK

    index = holder.setdefault("index", json.loads((OBJECTS_DIR / "gso_index.json").read_text()))
    scanned = e % 2 == 1
    rng = np.random.default_rng([int(ARGS.get("seed", 640000)), e])
    for _ in range(50):
        n = int(rng.integers(3, 5 if scanned else 6))
        specs = scanned_objects(rng, n, strict=False) if scanned else random_objects(rng, n)
        try:
            poses = layout(rng, specs, tries=200, max_r=0.23)
            break
        except RuntimeError:
            continue
    true_objs = planner.make_scene(specs, poses, index)
    reqs = {f: (t, g) for f, t, g in planner.requests(true_objs, rng, scanned)}
    fam = next(f for f in FAMILIES[(e // 2) % len(FAMILIES):] + FAMILIES if f in reqs)
    request, goal = reqs[fam]
    model = build(specs, poses)
    data, scratch = mujoco.MjData(model), mujoco.MjData(model)
    if holder.get("eyes") is None:
        from so101.see import Eyes
        holder["eyes"] = Eyes(model, data, cams=("cam_left", "cam_right"))
    eyes = holder["eyes"]
    eyes.attach(model, data, seed=e)
    arm = Arm(model, data, scratch, robot=Guard(SimArm(model, data, seed=e, robot=SO101), model, mode="warn", robot=SO101))
    arm.goal = PARK.copy()
    for _ in range(67):
        arm.step(arm.target(arm.goal))

    def true_of(c):
        d = [np.linalg.norm(data.xpos[model.body(body_name(s)).id][:2] - c[:2]) for s in specs]
        return true_objs[int(np.argmin(d))]["id"] if min(d) < SAME else None
    objs = look(eyes, namer, keep_views=True)
    ids = [f"p{i + 1}" for i in range(len(objs))]
    to_true = {i: true_of(o["centre"]) for i, o in zip(ids, objs)}
    true_name = {t["id"]: t["name"] for t in true_objs}
    one, two = [crops_of(o, False) for o in objs], [crops_of(o, True) for o in objs]

    def pobjs(names):
        return [dict(id=i, name=nm, dims=o["dims"], x=round(float(o["centre"][0]) * 100, 1),
                     y=round(float(o["centre"][1]) * 100, 1), where=o["where"]) for i, nm, o in zip(ids, names, objs)]

    def ok(ans):
        mapped = None if ans is None else dict(cannot=ans.get("cannot", False), moves=[
            dict(object=to_true.get(m.get("object")) or "none", place=m.get("place")) for m in ans.get("moves") or []
            if isinstance(m, dict)])
        return planner.check(true_objs, mapped, goal)
    dam = [o["name"] for o in objs]
    hidden = ["object " + i for i in ids]
    plans = {"vision": plan_with_pictures(pobjs(dam), one, request, True),
             "noname": plan_with_pictures(pobjs(hidden), one, request, False),
             "twoview": plan_with_pictures(pobjs(dam), two, request, True),
             "twoview-noname": plan_with_pictures(pobjs(hidden), two, request, False),
             "true": ollama_plan(pobjs([true_name.get(to_true[i], "unknown object") for i in ids]), request, True)[0]}
    rec = dict(episode=e, set="scanned" if scanned else "primitives", family=fam, request=request, goal=goal,
               perceived=[dict(id=i, true=to_true[i], dam=d, views=len(c)) for i, d, c in zip(ids, dam, two)],
               plans=plans, plan_ok={k: ok(v) for k, v in plans.items()})
    print(f"SCENE {e} {fam}: {request} | " + " ".join(f"{k}={int(v)}" for k, v in rec["plan_ok"].items()), flush=True)
    return rec


def score():
    rows = [json.loads(l) for p in sorted(OUT.glob(f"{TAG}*.jsonl")) for l in p.read_text().splitlines()]
    out = {k: sum(r["plan_ok"][k] for r in rows) for k in VARIANTS}
    out["n"] = len(rows)
    out["vs_vision"] = {k: mcnemar([r["plan_ok"][k] for r in rows], [r["plan_ok"]["vision"] for r in rows])
                        for k in VARIANTS if k != "vision"}
    for s in ("primitives", "scanned"):
        out[s] = {k: sum(r["plan_ok"][k] for r in rows if r["set"] == s) for k in VARIANTS} | {
            "n": sum(r["set"] == s for r in rows)}
    (OUT / f"{TAG}-score.json").write_text(json.dumps(out, indent=1))
    print("SCORE " + json.dumps(out), flush=True)


def main():
    if "--score" in sys.argv:
        return score()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{TAG}.jsonl"
    done = {json.loads(l)["episode"] for l in path.read_text().splitlines()} if path.exists() else set()
    holder, namer = {}, Namer()
    n, first = int(ARGS.get("episodes", 60)), int(ARGS.get("first", 0))
    with path.open("a") as f:
        for e in range(first, first + n):
            if e in done:
                continue
            t0 = time.time()
            r = scene(e, holder, namer)
            r["seconds"] = round(time.time() - t0, 1)
            f.write(json.dumps(r, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
