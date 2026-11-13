"""Stack v1 (docs/stack_v1.md): the whole SO-101 stack on one free-text request, camera only, scored by the simulator.

  request -> perception lists the scene (Grounding DINO "object." boxes, SAM 2.1 masks, DAM-3B names, depth points)
          -> planner (Gemma 4 12B 4-bit, Ollama) turns request + perceived scene into moves "p2 -> red_tray" or cannot
          -> decision (JEV-9B, /v1/decide) picks skill + object each step from the goal and the perceived scene
          -> skills (bench_general.py's scripted pick from the object's camera points, place on a free spot)
          -> look again after every skill; ids carry over to the nearest object seen before
  The simulator's state is read only to score (and for the oracle reference runs below).

  python stack_v1.py [--episodes=100] [--seed=400000] [--first=0] [--part=i/n] [--tag=v1]
         [--perception=dam|oracle] [--decider=jev|rule] [--think=on|off] [--only=3,17,...] [--video=on|off]
  Needs JEV_URL and OLLAMA_HOST (vla/stack_serve.sh); cluster: vla/stack.sbatch. -> runs/stack_v1/<tag>.jsonl
  Replays (owner 2026-10-09: a few per version, not every episode; filming slows each episode): after a test, rerun 4
  chosen episodes with --only=... --video=on -> runs/stack_v1/video/<tag>/ep<NNN>.mp4, a view of the scene beside both
  cameras (so101/camview.py's Recorder), status line = request, step, result. Display only.
"""
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

import mujoco
import numpy as np

from so101 import planner
from so101.objects import OBJECTS_DIR, body_name, build, layout, random_objects, scanned_objects

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
OUT = Path("runs/stack_v1")
TAG = ARGS.get("tag", "v1")
PERCEPTION, DECIDER = ARGS.get("perception", "dam"), ARGS.get("decider", "jev")
FAMILIES = ["single", "two", "all-but", "superlative", "category", "not-category", "sort", "conditional", "impossible"]
PLACE_HALF = {"red_tray": (0.045, 0.045), "blue_tray": (0.045, 0.045), "shelf": (0.04, 0.06)}  # bench_general.PLACES
PARK = np.array([0.10, 0.0, 0.20])  # where the hand waits while the cameras look (bench_general.py)
ESCALATE = 0.7  # docs/plan.md: below this the decision model's skill choice goes to the planner instead
SAME = 0.04  # m: a perceived object within this of one seen before keeps its id (and maps to a true object for scoring)
MOVED = 0.02  # m: a non-goal object that moved more than this spoils the request
VIDEO = ARGS.get("video", "off") == "on"

# --- perception: list every object (vla/crop_list.py's detector-first listing, DAM-3B namer) ------------------------
FIXTURE = 0.3  # vla/crop_list.py: drop a box more than 30 % robot or fixture pixels
JUNK = re.compile(r"\b(reflection|reflective|shadow|glare|robot|arm|tray|shelf|table|floor|surface|background)s?\b", re.I)
DAM_Q = "<image>\nDescribe the masked region in a few words."


def short_name(text):
    """vla/crop_list.py short_name: first sentence, at most 12 words, no article; None when it names only junk."""
    s = re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0].strip().rstrip(".")
    s = re.sub(r"^(the|a|an|this is|it is|label:|category:)\s+", "", s, flags=re.I).strip()
    s = " ".join(s.split()[:12])
    rest = JUNK.sub("", s)
    return None if JUNK.search(s) and not re.search(r"[a-z]{3,}", re.sub(r"\b(shiny|dark|light|grey|gray|white|black|"
                                                                        r"yellow|blue|red|wooden|flat|part|of|on|the|a|an|piece|edge|corner|its|"
                                                                        r"with|and|small|large|metal)\b", "", rest,
                                                                        flags=re.I), re.I) else (s or None)


def _iou(a, b):
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    i = w * h
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def keep_boxes(boxes, scores, known, nms=0.5):
    """vla/crop_list.py keep_boxes after see.py's threshold and size filter: NMS, then boxes mostly robot/fixture out."""
    keep = []
    for i in np.argsort(-np.asarray(scores), kind="stable"):
        b = boxes[i]
        if any(_iou(b, boxes[j]) > nms for j in keep):
            continue
        x0, y0, x1, y1 = np.clip(np.round(b).astype(int), 0, 640)
        if x1 > x0 and y1 > y0 and known[y0:y1, x0:x1].mean() > FIXTURE:
            continue
        keep.append(int(i))
    return keep


class Namer:
    """DAM-3B (NVIDIA Describe Anything) through vla/dam_serve.py ($DAM_URL; its own environment): region -> a few words."""

    def __init__(self):
        self.url = os.environ.get("DAM_URL", "http://127.0.0.1:17000").rstrip("/") + "/describe"

    def __call__(self, img, mask):
        import base64
        import io
        from PIL import Image

        def png(a):
            f = io.BytesIO()
            Image.fromarray(a).save(f, "PNG")
            return base64.b64encode(f.getvalue()).decode()
        body = json.dumps(dict(image=png(img), mask=png(mask.astype(np.uint8) * 255))).encode()
        q = urllib.request.Request(self.url, body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(q, timeout=300) as f:
            return json.loads(f.read())["text"]


def place_of(c):
    """Where a perceived object is, from its position (the places are fixed and calibrated)."""
    for p, (_, (x, y)) in planner.PLACES.items():
        hx, hy = PLACE_HALF[p]
        if abs(c[0] - x / 100) < hx and abs(c[1] - y / 100) < hy and (p != "shelf" or c[2] > 0.055):
            return p
    return "table"


def look(eyes, namer, keep_views=False):
    """Every object the two cameras see: [{name, centre, pts, dims (cm), where}], one entry per physical object."""
    views = eyes.snap()
    found = []
    for k, (img, _) in enumerate(views):
        boxes, scores = eyes.boxes(img, "object")
        for i in keep_boxes(boxes, scores, eyes.known_px[k]):
            m = eyes.mask(img, boxes[i])
            pts = eyes.points_from_mask(k, m)
            if pts is None or pts[:, 2].max() <= 0.009 or np.ptp(pts[:, :2], 0).max() >= 0.18:
                continue  # flat (a tray), or too big for one object
            found.append(dict(cam=k, score=float(scores[i]), img=img, mask=m, pts=pts, centre=np.median(pts, 0),
                              box=boxes[i]))
    def inside(a, b, share=0.7):
        """Most of the smaller of two boxes lies inside the other."""
        w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
        small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
        return w > 0 and h > 0 and w * h > share * small

    def touch(a, b, r=0.01, share=0.2):
        """Same object: at least `share` of one cloud's points lie within r of the other's. Footprint overlap merged
        neighbours (2026-10-09 dev run: a pill bottle into the box beside it); layout() keeps objects >= 3.5 cm apart."""
        rng = np.random.default_rng(0)
        a = a[rng.choice(len(a), min(len(a), 400), replace=False)]
        b = b[rng.choice(len(b), min(len(b), 400), replace=False)]
        d = np.linalg.norm(a[:, None] - b[None], axis=2)
        return bool(max((d.min(1) < r).mean(), (d.min(0) < r).mean()) > share)
    objs = []
    for f in sorted(found, key=lambda f: -f["score"]):  # the same object in both cameras (or boxed twice): merge
        # (2026-10-09 dev run: centres within 3 cm missed one box seen from both sides: two objects)
        # one camera: a box mostly inside another is a part of it (a picture printed on a box); side-by-side objects on
        # a place may touch, so touching alone does not merge them. Two cameras: clouds that touch are one object.
        o = next((o for o in objs if any(inside(b, f["box"]) for b in o["boxes"].get(f["cam"], []))
                  or (f["cam"] not in o["boxes"] and touch(o["pts"], f["pts"]))), None)
        if o is None:
            objs.append(dict(best=f, pts=f["pts"], centre=f["centre"], boxes={f["cam"]: [f["box"]]}, fs=[f]))
        else:
            o["pts"] = np.concatenate([o["pts"], f["pts"]])
            o["centre"] = np.median(o["pts"], 0)
            o["boxes"].setdefault(f["cam"], []).append(f["box"])
            o["fs"].append(f)
    out = []
    for o in objs:
        name = short_name(namer(o["best"]["img"], o["best"]["mask"])) if namer else None
        if namer and not name:
            continue  # the namer saw junk (a reflection, part of the robot)
        p = o["pts"]
        lw = sorted(np.ptp(p[:, :2], 0) * 100, reverse=True)
        c = np.median(p, 0)
        out.append(dict(name=name, centre=c, pts=p, dims=(round(lw[0], 1), round(lw[1], 1), round(float(p[:, 2].max()) * 100, 1)),
                        where=place_of(c), **({"views": [(f["img"], f["mask"], f["box"]) for f in o["fs"]]} if keep_views else {})))
    return out


# --- the planner and the decision model -------------------------------------------------------------------------------
def ollama_plan(objs, request, think=True):
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    host = host if host.startswith("http") else "http://" + host
    body = dict(model="gemma4:12b", messages=planner.prompt(objs, request), stream=False, think=think,
                options=dict(temperature=1.0, top_p=0.95, seed=0, num_ctx=16384, num_predict=8192))  # plan_bench's settings
    q = urllib.request.Request(host + "/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(q, timeout=3600) as f:
        o = json.loads(f.read())
    text = o.get("message", {}).get("content", "")
    ans = None if o.get("done_reason") == "length" else planner.parse(text)
    return ans, dict(seconds=round(time.time() - t0, 1), tokens=o.get("eval_count"), text=text[-400:])


SKILL_Q = {"type": "choice",
           "instructions": "Which skill should the robot run next? Pick grasps an object and lifts it; "
                           "place puts the held object on a tray or the shelf; done ends the task.",
           "criteria": {"pick": "grasp an object and lift it", "place": "put the held object on a place",
                        "done": "the task is finished, stop"}}  # so101/tabletop.py's wording (JEV-9B: 49/50 there)


class Decide:
    """The decision model's view of one step (Jev / Laya schema, so101/deciders.py): goal, perceived scene, gripper."""

    def __init__(self, goal, names, seen, held, last, steps, cap):
        self.goal, self.names, self.seen, self.held, self.last, self.steps, self.cap = goal, names, seen, held, last, steps, cap
        targets = {i: f"the {names[i]}" for i in seen} | {p: planner.PLACES[p][0] for p in planner.PLACES}
        ok = ("place", "done") if held else ("pick", "done")  # pick needs an empty gripper, place a held object
        skill_q = dict(SKILL_Q, criteria={k: v for k, v in SKILL_Q["criteria"].items() if k in ok})
        self.questions = {"skill": skill_q, "object": {
            "type": "choice", "criteria": targets,
            "instructions": "Which object or place should the skill act on? Pick needs an object; place needs a tray or the shelf."}}
        self.valid_targets = {"pick": [i for i in seen if i != held], "place": list(planner.PLACES)}

    def goal_text(self):
        if not self.goal:
            return "Nothing to do."
        parts = [f"the {self.names[i]} ({i}) {planner.WHERE[p]}" for i, p in self.goal]
        return "Put " + " and ".join(parts) + "."

    def state_text(self):
        lines = [f"Task: {self.goal_text()}",
                 f"Gripper: {'holding the ' + self.names[self.held] + f' ({self.held})' if self.held else 'empty'}."]
        for i, o in self.seen.items():
            at = ("in the gripper" if i == self.held else "on the table" if o["where"] == "table"
                  else planner.WHERE[o["where"]])
            lines.append(f"{self.names[i]} ({i}): x={o['centre'][0] * 1000:.0f} y={o['centre'][1] * 1000:.0f} mm, {at}.")
        lines.append(f"Last skill: {self.last}. Skills used: {self.steps} of {self.cap}.")
        return " ".join(lines)

    def oracle_action(self):  # the rule decider (reference run): next unmet goal move
        opts = {q: list(s["criteria"]) for q, s in self.questions.items()}
        if self.held:
            p = dict(self.goal).get(self.held, "shelf")
            return [opts["skill"].index("place"), opts["object"].index(p)]
        for i, p in self.goal:
            if i in self.seen and self.seen[i]["where"] != p:
                return [opts["skill"].index("pick"), opts["object"].index(i)]
        return [opts["skill"].index("done"), 0]


# --- one episode -------------------------------------------------------------------------------------------------------
def episode(e, holder, namer, decider, index):
    from so101.grasp import HOVER_MIN, pick_at, place_on, plan, support_height
    from so101.robot import SO101, Guard, SimArm
    from so101.skills import Arm, move_to, run as drive

    scanned = e % 2 == 1
    rng = np.random.default_rng([int(ARGS.get("seed", 400000)), e])
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
    rec = dict(episode=e, set="scanned" if scanned else "primitives", family=fam, request=request, goal=goal,
               objects=[o["name"] for o in true_objs])
    print(f"EPISODE {e} {rec['set']} {fam}: {request}", flush=True)

    model = build(specs, poses)
    g = model.actuator("gripper").id
    model.actuator_forcerange[g] = [-1.47, 1.47]  # bench_general's gripper torque (LeRobot's 50 % limit)
    data, scratch = mujoco.MjData(model), mujoco.MjData(model)
    if holder.get("eyes") is None:  # perception models load once, on the first scene (see.py needs a model)
        from so101.see import Eyes
        holder["eyes"] = Eyes(model, data, cams=("cam_left", "cam_right"))
    eyes = holder["eyes"]
    eyes.attach(model, data, seed=e)
    sim_arm = SimArm(model, data, seed=e, robot=SO101)
    guard = Guard(sim_arm, model, mode="warn", robot=SO101)
    arm = Arm(model, data, scratch, robot=guard)
    cam = vid = None
    if VIDEO:  # every arm tick redraws the cameras and adds a frame; nothing feeds back into the run
        from so101.camview import CamView, Recorder
        (OUT / "video" / TAG).mkdir(parents=True, exist_ok=True)
        cam = CamView(eyes, every=4, window=False)
        cam.status = f"{e}: {request}"
        vid = Recorder(OUT / "video" / TAG / f"ep{e:03d}.mp4", model, data, cam, fps=15)
        raw_step = arm.step

        def step_and_film(q):
            raw_step(q)
            noise = getattr(eyes.src, "image_noise", 0)  # filmed without image noise: draws nothing from the cameras'
            eyes.src.image_noise = 0  # noise stream (perception sees what it would unfilmed) and skips the slow part
            cam.show()
            eyes.src.image_noise = noise
            vid.add()
        arm.step = step_and_film
    arm.blocked_margin = 0.06
    arm.goal = PARK.copy()
    for _ in range(67):
        arm.step(arm.target(arm.goal))
    true_pos = lambda s: data.xpos[model.body(body_name(s)).id].copy()  # noqa: E731
    start = [true_pos(s) for s in specs]

    def true_of(c):
        """The true object nearest a perceived centre (scoring only), or None."""
        d = [np.linalg.norm(true_pos(s)[:2] - c[:2]) for s in specs]
        return true_objs[int(np.argmin(d))]["id"] if min(d) < SAME else None

    def settle(n=60):
        for _ in range(n):
            arm.step(arm.target(arm.goal))

    # 1. look
    t0 = time.time()
    if PERCEPTION == "oracle":  # reference run: true names and positions, objects still pointed at by the cameras
        seen0 = look(eyes, None)
        for o in seen0:
            tid = true_of(o["centre"])
            o["name"] = next((t["name"] for t in true_objs if t["id"] == tid), "unknown object")
    else:
        seen0 = look(eyes, namer)
    names, seen, nid = {}, {}, 0
    for o in seen0:
        nid += 1
        pid = f"p{nid}"
        names[pid], seen[pid] = o["name"], o
    rec["look_s"] = round(time.time() - t0, 1)
    rec["perceived"] = [dict(id=i, name=names[i], true=true_of(o["centre"]), at_cm=(o["centre"][:2] * 100).round(1).tolist(),
                             where=o["where"]) for i, o in seen.items()]
    tids = [p["true"] for p in rec["perceived"] if p["true"]]
    rec["found"], rec["junk"] = len(set(tids)), sum(p["true"] is None for p in rec["perceived"])

    # 2. plan, on the perceived scene; the same planner on the true scene is the reference (oracle perception)
    pobjs = [dict(id=i, name=names[i], dims=o["dims"], x=round(float(o["centre"][0]) * 100, 1),
                  y=round(float(o["centre"][1]) * 100, 1), where=o["where"]) for i, o in seen.items()]
    think = ARGS.get("think", "on") == "on"
    ans, info = ollama_plan(pobjs, request, think)
    oracle_ans, oinfo = ollama_plan(true_objs, request, think)
    to_true = {p["id"]: p["true"] for p in rec["perceived"]}
    mapped = None if ans is None else dict(cannot=ans.get("cannot", False), moves=[
        dict(object=to_true.get(m.get("object")) or "none", place=m.get("place")) for m in ans.get("moves") or []
        if isinstance(m, dict)])
    rec.update(plan=ans, plan_info=info, plan_ok=planner.check(true_objs, mapped, goal),
               oracle_plan_ok=planner.check(true_objs, oracle_ans, goal), oracle_plan_s=oinfo["seconds"])

    # 3-5. decide and act until done
    moves = [] if ans is None or ans.get("cannot") else [(m["object"], m["place"]) for m in ans.get("moves") or []
                                                         if isinstance(m, dict) and m.get("object") in seen
                                                         and m.get("place") in planner.PLACES]
    cap = 2 * len(moves) + 4
    held, last, steps, log, off_plan, skill_fail = None, "none yet", 0, [], False, False
    escalations = 0
    failed_yaws = {}
    while moves and steps < cap:
        ok_place_id = None
        d = Decide(moves, names, seen, held, last, steps, cap)
        ch = decider.decide(d)
        skill, target = ch["skill"]["choice"], ch["object"]["choice"]
        step = dict(step=steps + 1, skill=skill, target=target, p_skill=round(ch["skill"]["probabilities"].get(skill, 0), 3),
                    p_target=round(ch["object"]["probabilities"].get(target, 0), 3))
        steps += 1
        if cam:
            cam.status = f"{e} step {steps}: {skill} {names.get(target, target)} (p {step['p_skill']:.2f})"
        if ch["skill"]["probabilities"] and step["p_skill"] < ESCALATE and escalations < 2:
            # docs/plan.md: low confidence -> the planner re-plans (the rule decider gives no probabilities)
            escalations += 1
            now = [dict(id=i, name=names[i], dims=o["dims"], x=round(float(o["centre"][0]) * 100, 1),
                        y=round(float(o["centre"][1]) * 100, 1), where="gripper" if i == held else o["where"])
                   for i, o in seen.items()]
            re_ans, re_info = ollama_plan([o for o in now if o["where"] != "gripper"] if not held else
                                          [dict(o, where="table") if o["where"] == "gripper" else o for o in now],
                                          request, think)
            new_moves = [] if re_ans is None or re_ans.get("cannot") else [
                (m["object"], m["place"]) for m in re_ans.get("moves") or []
                if isinstance(m, dict) and m.get("object") in seen and m.get("place") in planner.PLACES]
            step.update(escalated=True, replan=re_ans, replan_s=re_info["seconds"])
            if held and held not in dict(new_moves):  # the planner forgot the held object: keep its planned place
                new_moves = [(held, dict(moves).get(held, "shelf"))] + new_moves
            moves = new_moves
            cap = max(cap, steps + 2 * len(moves) + 2)
            log.append(step)
            last = "asked the planner again"
            if not moves:
                break
            continue
        if skill == "done":
            left = [i for i, p in moves if i not in seen or seen[i]["where"] != p]
            off_plan |= bool(held or left)
            step["left"] = left
            log.append(step)
            break
        want = dict(moves)
        if (skill == "pick" and (target not in want or held)) or (skill == "place" and (not held or want.get(held) != target)):
            off_plan = True
        ok, why = False, ""
        if skill == "pick" and target in seen and not held:
            drive(arm, move_to(arm, PARK))
            settle(30)
            eyes.snap()
            everything = eyes.scene_points(min_z=-0.01)
            scene = everything[everything[:, 2] > 0.004]
            pts = seen[target]["pts"]
            sup = support_height(pts, everything)
            gr = plan(pts, scene, sup, avoid=failed_yaws.get(target, []), hand=arm.hand, check=arm.can_grasp)
            if gr is None:
                why = "no grasp fits"
            else:
                hover = max(HOVER_MIN, 0.06 + gr.z - gr.support + 0.025, gr.top + 0.02)
                ok, why = drive(arm, pick_at(arm, gr, hover))
                if ok:
                    held, seen[target]["grasp"] = target, gr
                else:
                    failed_yaws.setdefault(target, []).append(gr.yaw)
        elif skill == "place" and held and target in planner.PLACES:
            ok_place_id = held
            gr = seen[held]["grasp"]
            hang = gr.z - gr.support
            hover = max(HOVER_MIN, 0.06 + hang + 0.025, gr.top + 0.02)
            eyes.snap()
            everything = eyes.scene_points(min_z=-0.01)
            scene = everything[everything[:, 2] > 0.004]
            centre = data.xpos[model.body(target).id][:2]  # fixed, calibrated place
            half = np.array(PLACE_HALF[target])
            radius = max(gr.radius + 0.005, 0.015)
            cands = [centre + np.array([dx, dy]) * (half - radius).clip(0) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
            surf_pts = everything[np.linalg.norm(everything[:, :2] - centre, axis=1) < 0.03]
            surface = float(np.median(surf_pts[:, 2])) if len(surf_pts) else 0.0
            busy = [np.sum((np.linalg.norm(scene[:, :2] - c, axis=1) < radius + 0.01) & (scene[:, 2] > surface + 0.004))
                    for c in cands]
            spot = cands[int(np.argmin([b + 0.1 * np.linalg.norm(c - centre) * 100 for b, c in zip(busy, cands)]))]
            ok, why = drive(arm, place_on(arm, spot, surface, hang, max(hover, surface + hang + 0.03)))
            settle(80)
            held = None
        else:
            why = "not possible now"
        skill_fail |= not ok
        anchor = {}  # id -> (where to look for it, how far): an object the skill just moved keeps its id
        if skill == "place" and ok_place_id:
            anchor[ok_place_id] = (np.r_[spot, 0.0], 0.06)
        elif skill == "pick" and not ok and target in seen:
            anchor[target] = (seen[target]["centre"], 0.10)
        last = f"{skill} {target}: {'done' if ok else 'failed'}"
        step.update(ok=bool(ok), why=why)
        log.append(step)
        # look again; ids carry over to the nearest object seen before
        drive(arm, move_to(arm, PARK))
        settle(30)
        new = look(eyes, namer) if PERCEPTION == "dam" else look(eyes, None)
        tip = arm.tip_pos()
        nxt = {}
        if held:
            new = [o for o in new if np.linalg.norm(o["centre"] - tip) >= 0.06]  # the held object, seen in the jaw
        for pid, (c, r) in anchor.items():
            cand = [o for o in new if np.linalg.norm(o["centre"][:2] - c[:2]) < r]
            if cand:
                o = min(cand, key=lambda o: np.linalg.norm(o["centre"][:2] - c[:2]))
                o.update({k: v for k, v in seen.get(pid, {}).items() if k == "grasp"})
                nxt[pid] = o
                new = [x for x in new if x is not o]
        for o in new:
            prev = min((i for i in seen if i not in nxt), key=lambda i: np.linalg.norm(seen[i]["centre"][:2] - o["centre"][:2]),
                       default=None)
            if prev and prev not in nxt and np.linalg.norm(seen[prev]["centre"][:2] - o["centre"][:2]) < SAME:
                o.update({k: v for k, v in seen[prev].items() if k == "grasp"})
                nxt[prev] = o
            else:
                nid += 1
                pid = f"p{nid}"
                names[pid] = o["name"] if PERCEPTION == "dam" else (
                    next((t["name"] for t in true_objs if t["id"] == true_of(o["centre"])), "unknown object"))
                nxt[pid] = o
        if held:
            nxt[held] = seen[held]
        seen = nxt
        if held and not arm.holding(None):  # the jaw closed on nothing: dropped
            held = None

    # score (truth)
    def at_place(s, p):
        c, q = true_pos(s), data.xpos[model.body(p).id]
        top = 0.06 if p == "shelf" else 0.0
        return bool(np.all(np.abs(c[:2] - q[:2]) < PLACE_HALF[p]) and top - 0.005 < c[2] < top + 0.1)
    ids = [o["id"] for o in true_objs]
    goal_d = goal or {}
    placed = {i: at_place(specs[ids.index(i)], p) for i, p in goal_d.items()}
    disturbed = [i for i in ids if i not in goal_d and np.linalg.norm(true_pos(specs[ids.index(i)])[:2] - start[ids.index(i)][:2]) > MOVED]
    if goal is None:
        success = bool(ans and ans.get("cannot")) and not any(
            np.linalg.norm(true_pos(s)[:2] - p0[:2]) > MOVED for s, p0 in zip(specs, start))
    else:
        success = all(placed.values()) and not disturbed
    cause = None
    if not success:
        if not rec["plan_ok"]:
            cause = "planner" if not rec["oracle_plan_ok"] else "perception"
        elif off_plan:
            cause = "decision"
        elif skill_fail or not all(placed.values()):
            cause = "skill"
        else:
            cause = "disturbed" if disturbed else "unknown"
    rec.update(steps=log, placed=placed, disturbed=disturbed, success=success, cause=cause,
               seconds=round(time.time() - t0, 1))
    if vid:  # a second of the end state under the result (after scoring)
        cam.status = f"{e}: {'DONE' if success else 'FAILED (' + str(cause) + ')'}  -  {request}"
        settle(60)
        vid.close()
    print(f"RESULT {e}: {'OK' if success else 'FAILED (' + str(cause) + ')'} plan_ok={rec['plan_ok']} "
          f"oracle_plan_ok={rec['oracle_plan_ok']} steps={len(log)} found={rec['found']}/{len(specs)} junk={rec['junk']} "
          f"{rec['seconds']} s", flush=True)
    return rec


def main():
    from so101 import deciders
    index = json.loads((OBJECTS_DIR / "gso_index.json").read_text())
    n, first = int(ARGS.get("episodes", 100)), int(ARGS.get("first", 0))
    eps = [int(x) for x in ARGS["only"].split(",")] if "only" in ARGS else list(range(first, first + n))
    if "part" in ARGS:
        i, k = map(int, ARGS["part"].split("/"))
        eps = eps[i::k]
    holder = {}
    namer = Namer() if PERCEPTION == "dam" else None
    decider = deciders.make("jev-local" if DECIDER == "jev" else "rule")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{TAG}{'-part' + ARGS['part'].replace('/', 'of') if 'part' in ARGS else ''}.jsonl"
    done = {json.loads(l)["episode"] for l in path.read_text().splitlines()} if path.exists() else set()
    with path.open("a") as f:
        for e in eps:
            if e in done:
                continue
            r = episode(e, holder, namer, decider, index)
            f.write(json.dumps(r, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)) + "\n")
            f.flush()
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    causes = {}
    for r in rows:
        causes[r["cause"]] = causes.get(r["cause"], 0) + 1
    print(f"SUMMARY {path}: {sum(r['success'] for r in rows)}/{len(rows)} requests done; causes {causes}; plan right "
          f"{sum(r['plan_ok'] for r in rows)}, oracle-perception plan right {sum(r['oracle_plan_ok'] for r in rows)}", flush=True)


if __name__ == "__main__":
    main()
