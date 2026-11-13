"""Smoke test for stack v1 (docs/stack_v1.md): every layer's model loads and answers on one GPU, next to the others.
  JEV_URL=... OLLAMA_HOST=... MUJOCO_GL=egl python vla/stack_smoke.py      (vla/stack.sbatch MODE=smoke)
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import httpx  # noqa: E402

ok = {}


def step(name, fn):
    t0 = time.time()
    try:
        out = fn()
        ok[name] = True
        print(f"SMOKE {name}: ok in {time.time() - t0:.1f} s: {out}", flush=True)
    except Exception as e:  # noqa: BLE001
        ok[name] = False
        print(f"SMOKE {name}: FAILED in {time.time() - t0:.1f} s: {type(e).__name__}: {e}", flush=True)


def jev():
    r = httpx.post(os.environ["JEV_URL"] + "/v1/decide", timeout=120, json=dict(
        kind="choice", state="Task: put the red cube in the red tray. Gripper: empty. red cube: on the table.",
        question="Which skill should the robot use next?",
        options=["pick: pick up an object", "place: put the held object on a place", "done: the task is finished"]))
    r.raise_for_status()
    d = r.json()
    return f"{d['choice']} {np.round(d['probabilities'], 3).tolist()}"


def gemma():
    r = httpx.post(f"http://{os.environ['OLLAMA_HOST']}/api/chat", timeout=300, json=dict(
        model="gemma4:12b", stream=False, think=False,
        messages=[{"role": "user", "content": "Reply with the single word OK."}]))
    r.raise_for_status()
    return r.json()["message"]["content"].strip()[:40]


def scene():
    """A bench_general scene (seed 400000, household objects) seen by the front camera; DINO boxes with "object."."""
    import mujoco
    from so101.objects import build, layout, scanned_objects
    from so101.see import Eyes
    rng = np.random.default_rng(400000)
    for _ in range(50):
        specs = scanned_objects(rng, 3, strict=False)
        try:
            poses = layout(rng, specs, tries=200)
            break
        except RuntimeError:
            continue
    model = build(specs, poses)
    data = mujoco.MjData(model)
    for _ in range(200):
        mujoco.mj_step(model, data)
    eyes = Eyes(model, data, cams=("cam_left", "cam_right"))
    views = eyes.snap()
    boxes, _ = eyes.boxes(views[0][0], "object")
    globals()["SCENE"] = (views[0][0], boxes, eyes, [s["name"] for s in specs])
    return f"{len(boxes)} boxes for {[s['name'] for s in specs]}"


def dam():
    import stack_v1
    img, boxes, eyes, names = SCENE
    keep = stack_v1.keep_boxes(boxes, np.ones(len(boxes)), eyes.known_px[0])
    namer = stack_v1.Namer()
    out = [stack_v1.short_name(namer(img, eyes.mask(img, boxes[i]))) for i in keep]
    return f"kept {len(keep)} of {len(boxes)} boxes, names {out}; true names {names}"


def listing():
    import stack_v1
    img, boxes, eyes, names = SCENE
    objs = stack_v1.look(eyes, stack_v1.Namer())
    return [(o["name"], o["where"], (o["centre"][:2] * 100).round(1).tolist()) for o in objs] + [f"true: {names}"]


for n, f in [("jev", jev), ("gemma", gemma), ("scene", scene), ("dam", dam), ("listing", listing)]:
    step(n, f)
print("SMOKE-DONE " + " ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in ok.items()), flush=True)
