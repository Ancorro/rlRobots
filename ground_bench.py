"""Grounding check for stack v2 (docs/stack_v2.md): does matching the request's words to object crops fix v1's naming
failures? v1 test: plans right 79/100 with DAM-3B's names, 100/100 with the true names (21 of 38 failures were naming).

Each scene: stack v1's generator and request family (seed, episode), stack_v1.look with DAM names and crops, then the
same Gemma 4 12B planner (stack_v1.ollama_plan) on the same perceived objects under each naming:
  dam        v1: DAM-3B short names
  true       reference ceiling: each perceived object's true name
  siglip@g   request phrases (Gemma extracts them, text only) scored against every crop by SigLIP 2 giant-384
             ("a photo of a <phrase>."); an object takes its best phrase when that sigmoid score >= g, else its DAM name
  gemma      Gemma picks, per object crop, one of the request's phrases or "none of these" (closed choice)
  vision     the planner itself sees the crops: one picture per object, in id order (scene text unchanged)
Score: plan right (planner.check on the true scene, as in the stack) per naming, paired McNemar against dam.
Pre-registered: dev seeds 640000+ (60 scenes) pick one naming (highest plans right; ties: the simpler, in the order
above); one run on test seeds 660000+ (100 scenes): the pick must beat dam, paired McNemar p < 0.05.

  python ground_bench.py --seed=640000 --episodes=60 --tag=dev   (GPU job: STACK servers; SCRIPT=ground_bench.py vla/stack.sbatch)
  python ground_bench.py --score --tag=dev                        -> runs/ground/<tag>-score.json
"""
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

from stack_v1 import ARGS, FAMILIES, SAME, Namer, look, ollama_plan

OUT = Path("runs/ground")
TAG = ARGS.get("tag", "dev")
GIANT = "google/siglip2-giant-opt-patch16-384"
GATES = [0.01, 0.03, 0.1, 0.3]
NONE = "none of these"


def gemma(messages, think=False, n=512):
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    host = host if host.startswith("http") else "http://" + host
    body = dict(model="gemma4:12b", messages=messages, stream=False, think=think,
                options=dict(temperature=0, seed=0, num_ctx=16384, num_predict=n))
    q = urllib.request.Request(host + "/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(q, timeout=3600) as f:
        return json.loads(f.read()).get("message", {}).get("content", "")


PHRASE_Q = """List the words in this robot request that say what an object looks like or what kind of thing it is:
names, kinds, colours, categories. Leave out positions, sizes and comparisons ("closest", "tallest", "leftmost"),
"object"/"thing" on its own, and places (red tray, blue tray, shelf). Singular form; a bare colour becomes
"<colour> object". Answer with one JSON list of strings and nothing else, [] if there are none.
Request: """


def phrases(request):
    text = gemma([{"role": "user", "content": PHRASE_Q + request}], n=256)
    m = re.search(r"\[.*?\]", text, re.S)
    try:
        out = [str(p).strip().lower() for p in json.loads(m.group(0))] if m else []
    except json.JSONDecodeError:
        out = []
    return [p for p in dict.fromkeys(out) if p]


def png(img):
    import base64
    import io

    from PIL import Image
    f = io.BytesIO()
    Image.fromarray(img).save(f, "PNG")
    return base64.b64encode(f.getvalue()).decode()


def best_crop(o):
    """The object's largest view, padded and upscaled to >= 224 px (vla/crop_list.crop_png), as a numpy image."""
    from PIL import Image
    img, _, b = max(o["views"], key=lambda v: (v[2][2] - v[2][0]) * (v[2][3] - v[2][1]))
    x0, y0, x1, y1 = b
    pad = 0.15 * max(x1 - x0, y1 - y0) + 4
    x0, y0, x1, y1 = np.clip(np.round([x0 - pad, y0 - pad, x1 + pad, y1 + pad]).astype(int), 0, [img.shape[1], img.shape[0]] * 2)
    c = Image.fromarray(img[y0:y1, x0:x1])
    k = 224 / max(c.size)
    if k > 1:
        c = c.resize((round(c.size[0] * k), round(c.size[1] * k)), Image.BICUBIC)
    return np.asarray(c)


class Giant:
    def __init__(self, dev):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch, self.dev = torch, dev
        self.proc = AutoProcessor.from_pretrained(GIANT)
        self.model = AutoModel.from_pretrained(GIANT, torch_dtype=torch.float16).to(dev).eval()

    def __call__(self, crops, texts):
        """(crops x texts) sigmoid scores."""
        with self.torch.inference_mode():
            x = self.proc(text=[f"a photo of a {t}." for t in texts], images=crops, padding="max_length", max_length=64,
                          return_tensors="pt").to(self.dev)
            x["pixel_values"] = x["pixel_values"].half()
            return self.torch.sigmoid(self.model(**x).logits_per_image.float()).cpu().numpy()


def gemma_choice(crop, ps):
    letters = "ABCDEFGHIJ"[:len(ps) + 1]
    opts = "\n".join(f"{c}) {p}" for c, p in zip(letters, ps + [NONE]))
    text = gemma([{"role": "user", "images": [png(crop)], "content":
                   f"A robot camera's close-up of one object on a table. Which description fits this object?\n{opts}\n"
                   "Answer with one letter and nothing else."}], n=8)
    m = re.search(r"\b([A-J])\b", text.strip().upper())
    i = letters.find(m.group(1)) if m else -1
    return ps[i] if 0 <= i < len(ps) else None


def vision_plan(pobjs, crops, request):
    from so101 import planner
    msgs = planner.prompt(pobjs, request)
    ids = ", ".join(o["id"] for o in pobjs)
    msgs[-1] = dict(msgs[-1], images=[png(c) for c in crops],
                    content=msgs[-1]["content"] + f"\n\nThe pictures show the objects' close-ups, in this order: {ids}. "
                    "Names come from a camera and may be wrong; trust the pictures over the names.")
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    host = host if host.startswith("http") else "http://" + host
    body = dict(model="gemma4:12b", messages=msgs, stream=False, think=True,
                options=dict(temperature=1.0, top_p=0.95, seed=0, num_ctx=16384, num_predict=8192))  # ollama_plan's
    q = urllib.request.Request(host + "/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(q, timeout=3600) as f:
        o = json.loads(f.read())
    return None if o.get("done_reason") == "length" else planner.parse(o.get("message", {}).get("content", ""))


def scene(e, holder, namer, giant_holder):
    import mujoco

    from so101 import planner
    from so101.objects import OBJECTS_DIR, body_name, build, layout, random_objects, scanned_objects
    from so101.robot import SO101, Guard, SimArm
    from so101.skills import Arm
    from stack_v1 import PARK

    index = holder.setdefault("index", json.loads((OBJECTS_DIR / "gso_index.json").read_text()))
    scanned = e % 2 == 1
    rng = np.random.default_rng([int(ARGS.get("seed", 640000)), e])  # stack_v1.episode's draws, in order
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
    if giant_holder.get("m") is None:
        giant_holder["m"] = Giant(eyes.dev)
    giant = giant_holder["m"]

    def true_of(c):
        d = [np.linalg.norm(data.xpos[model.body(body_name(s)).id][:2] - c[:2]) for s in specs]
        return true_objs[int(np.argmin(d))]["id"] if min(d) < SAME else None
    objs = look(eyes, namer, keep_views=True)
    ids = [f"p{i + 1}" for i in range(len(objs))]
    to_true = {i: true_of(o["centre"]) for i, o in zip(ids, objs)}
    crops = [best_crop(o) for o in objs]
    ps = phrases(request)
    rec = dict(episode=e, set="scanned" if scanned else "primitives", family=fam, request=request, goal=goal,
               phrases=ps, objects=[o["name"] for o in true_objs],
               perceived=[dict(id=i, true=to_true[i], dam=o["name"]) for i, o in zip(ids, objs)])
    S = giant(crops, ps) if ps else np.zeros((len(objs), 0))
    rec["siglip"] = S.round(4).tolist()
    rec["gemma_pick"] = [gemma_choice(c, ps) if ps else None for c in crops]
    true_name = {t["id"]: t["name"] for t in true_objs}
    namings = {"dam": [o["name"] for o in objs], "true": [true_name.get(to_true[i], "unknown object") for i in ids]}
    for g in GATES:
        namings[f"siglip@{g}"] = [ps[int(S[k].argmax())] if ps and S[k].max() >= g else o["name"]
                                  for k, o in enumerate(objs)]
    namings["gemma"] = [p or o["name"] for p, o in zip(rec["gemma_pick"], objs)]
    rec["names"], rec["plans"], rec["plan_ok"] = namings, {}, {}

    def pobjs(names):
        return [dict(id=i, name=nm, dims=o["dims"], x=round(float(o["centre"][0]) * 100, 1),
                     y=round(float(o["centre"][1]) * 100, 1), where=o["where"]) for i, nm, o in zip(ids, names, objs)]

    def ok(ans):
        mapped = None if ans is None else dict(cannot=ans.get("cannot", False), moves=[
            dict(object=to_true.get(m.get("object")) or "none", place=m.get("place")) for m in ans.get("moves") or []
            if isinstance(m, dict)])
        return planner.check(true_objs, mapped, goal)
    done = {}
    for k, names in namings.items():
        key = tuple(names)
        if key not in done:  # identical names give the identical prompt: one planner call
            done[key] = ollama_plan(pobjs(names), request, True)[0]
        rec["plans"][k], rec["plan_ok"][k] = done[key], ok(done[key])
    ans = vision_plan(pobjs(namings["dam"]), crops, request)
    rec["plans"]["vision"], rec["plan_ok"]["vision"] = ans, ok(ans)
    print(f"SCENE {e} {fam}: {request} | phrases {ps} | " + " ".join(f"{k}={int(v)}" for k, v in rec["plan_ok"].items()),
          flush=True)
    return rec


def mcnemar(a, b):
    from math import comb
    x, y = sum(i and not j for i, j in zip(a, b)), sum(j and not i for i, j in zip(a, b))
    n = x + y
    p = min(1.0, 2 * sum(comb(n, k) for k in range(min(x, y) + 1)) / 2 ** n) if n else 1.0
    return dict(only_a=x, only_b=y, p=p)


def score():
    rows = [json.loads(l) for l in (OUT / f"{TAG}.jsonl").read_text().splitlines()]
    keys = list(rows[0]["plan_ok"])
    out = {k: sum(r["plan_ok"][k] for r in rows) for k in keys}
    out["n"] = len(rows)
    out["vs_dam"] = {k: mcnemar([r["plan_ok"][k] for r in rows], [r["plan_ok"]["dam"] for r in rows]) for k in keys if k != "dam"}
    fams = {}
    for r in rows:
        f = fams.setdefault(r["family"], {k: 0 for k in keys} | {"n": 0})
        f["n"] += 1
        for k in keys:
            f[k] += r["plan_ok"][k]
    out["by_family"] = fams
    (OUT / f"{TAG}-score.json").write_text(json.dumps(out, indent=1))
    print("SCORE " + json.dumps({k: v for k, v in out.items() if k not in ("by_family",)}), flush=True)


def main():
    if "--score" in sys.argv:
        return score()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{TAG}.jsonl"
    done = {json.loads(l)["episode"] for l in path.read_text().splitlines()} if path.exists() else set()
    holder, giant = {}, {}
    namer = Namer()
    n, first = int(ARGS.get("episodes", 60)), int(ARGS.get("first", 0))
    with path.open("a") as f:
        for e in range(first, first + n):
            if e in done:
                continue
            t0 = time.time()
            r = scene(e, holder, namer, giant)
            r["seconds"] = round(time.time() - t0, 1)
            f.write(json.dumps(r, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)) + "\n")
            f.flush()
    score()


if __name__ == "__main__":
    main()
