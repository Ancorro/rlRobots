"""Stack v2 (docs/stack_v2.md), built one layer at a time on stack v1's frozen perception, planner and decider.

  request -> perception (stack_v1.look: DINO + SAM 2.1 boxes and masks, DAM-3B names, depth points)
          -> scene memory (so101/memory.py, re-ID test passed 2026-10-10: SigLIP 2 crops + action-predicted positions,
             Hungarian matching): stable ids across looks, drops flagged, lost objects re-linked by appearance
          -> planner (Gemma 4 12B) -> decisions (JEV-9B) -> scripted skills, a look after every skill
          -> verifier: "done" is accepted only when memory shows every goal object at its place and the gripper empty;
             otherwise the planner re-plans from memory (at most VERIFY_TRIES times, then "done" stands)
          -> grounding (--grounding=vision, default; grounding test passed 2026-10-13): the planner and its re-plans see
             each tracked object's best crop beside the DAM names (ground_bench.vision_plan); --grounding=dam is v1's text
  Layer still to come (docs/stack_v2.md): executive (Gemma).
  The simulator's state is read only to score.

  python stack_v2.py [--episodes=100] [--seed=400000] [--first=0] [--part=i/n] [--tag=v2-mem]
         [--perception=dam|oracle] [--decider=jev|subq|gemma|gemma-vllm|gemma-lora|rule] [--verify=on|off] [--grounding=vision|dam] [--think=on|off]
         [--only=3,17,...]
  Needs JEV_URL, OLLAMA_HOST, DAM_URL (vla/stack_serve.sh); cluster: STACK=v2 vla/stack.sbatch. -> runs/stack_v2/<tag>.jsonl
"""
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from so101 import planner
from so101.memory import SceneMemory
from so101.objects import OBJECTS_DIR, body_name, build, layout, random_objects, scanned_objects
from ground_bench import best_crop, vision_plan
from stack_v1 import ARGS, ESCALATE, FAMILIES, MOVED, PARK, PLACE_HALF, SAME, Decide, Namer, look, ollama_plan

OUT = Path("runs/stack_v2")
TAG = ARGS.get("tag", "v2-mem")
PERCEPTION, DECIDER = ARGS.get("perception", "dam"), ARGS.get("decider", "jev")
VERIFY = ARGS.get("verify", "on") == "on"
GROUNDING = ARGS.get("grounding", "vision")
VERIFY_TRIES = 2  # a rejected "done" re-plans; after this many the decision model's "done" stands
MEMORY = dict(method="both", pos_scale=0.03, app_scale=0.1, gate=3.0, relink_gate=2.0)  # re-ID dev pick (docs/stack_v2.md)


def crop(img, mask, box):
    """The object's masked crop, background grey (reid_bench.Embedder.crop: the crops the re-ID test used)."""
    x0, y0, x1, y1 = np.round(box).astype(int)
    pad = int(0.1 * max(x1 - x0, y1 - y0))
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(img.shape[1], x1 + pad), min(img.shape[0], y1 + pad)
    c = img[y0:y1, x0:x1].copy()
    c[~mask[y0:y1, x0:x1]] = 128
    return c


def embed(eyes, views):
    """SigLIP 2 (the Eyes' own model) on each view's crop; unit vectors, mean over views."""
    import torch
    with torch.inference_mode():
        s = eyes.sig.get_image_features(**eyes.sig_proc(images=[crop(*v) for v in views], return_tensors="pt").to(eyes.dev))
        s = getattr(s, "pooler_output", s)
    return (s / s.norm(dim=-1, keepdim=True)).mean(0).float().cpu().numpy()


class Clock:
    """Wall seconds per stage within one episode (rec["timing"]); nested stages count in both."""

    def __init__(self):
        self.s = {}

    def __call__(self, key, fn, *a, **k):
        t = time.time()
        try:
            return fn(*a, **k)
        finally:
            self.s[key] = self.s.get(key, 0.0) + time.time() - t

    def out(self):
        return {k: round(v, 1) for k, v in sorted(self.s.items())}


# --- one episode (stack_v1.episode with scene memory and the verifier) -------------------------------------------------
def episode(e, holder, namer, decider, index):
    from so101.grasp import HOVER_MIN, pick_at, place_on, plan as grasp_plan, support_height
    from so101.robot import SO101, Guard, SimArm
    from so101.skills import Arm, move_to, run as run_skill

    clock = Clock()
    drive = lambda *a: clock("arm", run_skill, *a)  # noqa: E731  (simulated arm motion)
    plan = lambda *a, **k: clock("grasp_plan", grasp_plan, *a, **k)  # noqa: E731
    if namer:
        dam = namer
        namer = lambda img, m: clock("dam_names", dam, img, m)  # noqa: E731  (inside look)
    _look = look
    look_ = lambda *a, **k: clock("look", _look, *a, **k)  # noqa: E731  (detect, mask, depth points, DAM names)

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
    rec = dict(grounding=GROUNDING, decider=DECIDER, episode=e, set="scanned" if scanned else "primitives", family=fam, request=request, goal=goal,
               objects=[o["name"] for o in true_objs])
    print(f"EPISODE {e} {rec['set']} {fam}: {request}", flush=True)

    model = build(specs, poses)
    g = model.actuator("gripper").id
    model.actuator_forcerange[g] = [-1.47, 1.47]  # bench_general's gripper torque (LeRobot's 50 % limit)
    data, scratch = mujoco.MjData(model), mujoco.MjData(model)
    if holder.get("eyes") is None:
        from so101.see import Eyes
        holder["eyes"] = Eyes(model, data, cams=("cam_left", "cam_right"))
    eyes = holder["eyes"]
    eyes.attach(model, data, seed=e)
    sim_arm = SimArm(model, data, seed=e, robot=SO101)
    guard = Guard(sim_arm, model, mode="warn", robot=SO101)
    arm = Arm(model, data, scratch, robot=guard)
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
        t = time.time()
        for _ in range(n):
            arm.step(arm.target(arm.goal))
        clock.s["arm"] = clock.s.get("arm", 0.0) + time.time() - t

    mem = SceneMemory(**MEMORY)
    latest, mem_events, grasps = {}, [], {}  # track id -> its latest detection (points, place, dims); held grasp

    def observe(objs, held=None):
        """One look into memory. The held object is left out: any detection with points within 3 cm of the jaw (v1's
        6 cm centre rule missed big boxes: dev1 episode 3 matched a held box as a new sighting)."""
        if PERCEPTION == "oracle":
            for o in objs:
                tid = true_of(o["centre"])
                o["name"] = next((t["name"] for t in true_objs if t["id"] == tid), "unknown object")
        if held:
            tip = arm.tip_pos()
            objs = [o for o in objs if np.linalg.norm(o["centre"] - tip) >= 0.06
                    and np.linalg.norm(o["pts"] - tip, axis=1).min() >= 0.03]
        dets = [dict(centre=o["centre"], dims=o["dims"], where=o["where"], name=o["name"], emb=clock("embed", embed, eyes, o["views"]))
                for o in objs]
        assign, events = mem.update(dets)
        for i, o in enumerate(objs):
            if GROUNDING == "vision":
                o["crop"] = best_crop(o)
            o.pop("views", None)
            latest[assign[i]] = o
        if held and mem.tracks[held].status != "held":  # matched anyway: it is still in the jaw
            t = mem.tracks[held]
            t.status, t.where = "held", "gripper"
        mem_events.append([list(ev) for ev in events])
        return events

    def scene(held):
        """The memory's current view for the decision model: every track not missing, plus the held one."""
        seen = {t.id: latest[t.id] for t in mem.tracks.values() if t.status == "seen"}
        if held:
            seen[held] = latest[held]
        return {t.id: t.name for t in mem.tracks.values()}, seen

    # 1. look
    t0 = time.time()
    observe(look_(eyes, namer if PERCEPTION == "dam" else None, keep_views=True))
    names, seen = scene(None)
    rec["look_s"] = round(time.time() - t0, 1)
    rec["perceived"] = [dict(id=i, name=names[i], true=true_of(o["centre"]), at_cm=(o["centre"][:2] * 100).round(1).tolist(),
                             where=o["where"]) for i, o in seen.items()]
    tids = [p["true"] for p in rec["perceived"] if p["true"]]
    rec["found"], rec["junk"] = len(set(tids)), sum(p["true"] is None for p in rec["perceived"])

    # 2. plan, on the perceived scene; the same planner on the true scene is the reference (oracle perception)
    def objs_now(held):
        return [dict(id=i, name=names[i], dims=o["dims"], x=round(float(o["centre"][0]) * 100, 1),
                     y=round(float(o["centre"][1]) * 100, 1), where="table" if i == held else o["where"])
                for i, o in seen.items()]
    think = ARGS.get("think", "on") == "on"

    def plan_now(held):
        """The planner on memory's current view; with vision grounding it also sees each object's crop."""
        objs = objs_now(held)
        if GROUNDING != "vision":
            return ollama_plan(objs, request, think)
        t1 = time.time()
        a = vision_plan(objs, [latest[o["id"]]["crop"] for o in objs], request)
        return a, dict(seconds=round(time.time() - t1, 1))
    ans, info = clock("plan", plan_now, None)
    oracle_ans, oinfo = clock("oracle_plan_scoring_only", ollama_plan, true_objs, request, think)
    to_true = {p["id"]: p["true"] for p in rec["perceived"]}
    mapped = None if ans is None else dict(cannot=ans.get("cannot", False), moves=[
        dict(object=to_true.get(m.get("object")) or "none", place=m.get("place")) for m in ans.get("moves") or []
        if isinstance(m, dict)])
    rec.update(plan=ans, plan_info=info, plan_ok=planner.check(true_objs, mapped, goal),
               oracle_plan_ok=planner.check(true_objs, oracle_ans, goal), oracle_plan_s=oinfo["seconds"])

    def valid(a):
        return [] if a is None or a.get("cannot") else [(m["object"], m["place"]) for m in a.get("moves") or []
                                                         if isinstance(m, dict) and m.get("object") in seen
                                                         and m.get("place") in planner.PLACES]

    # 3-5. decide and act until done
    moves = valid(ans)
    cap = 2 * len(moves) + 4
    held, last, steps, log, off_plan, skill_fail = None, "none yet", 0, [], False, False
    escalations = rejected = 0
    failed_yaws = {}

    def replan(step, why):
        """docs/plan.md escalation, and the verifier's re-plan: the planner again, on memory's current view."""
        nonlocal moves, cap, last
        re_ans, re_info = clock("replan", plan_now, held)
        new_moves = valid(re_ans)
        step.update(escalated=why, replan=re_ans, replan_s=re_info["seconds"])
        if held and held not in dict(new_moves):  # the planner forgot the held object: keep its planned place
            new_moves = [(held, dict(moves).get(held, "shelf"))] + new_moves
        moves = new_moves
        cap = max(cap, steps + 2 * len(moves) + 2)
        log.append(step)
        last = "asked the planner again"

    while moves and steps < cap:
        ok_place_id = None
        d = Decide(moves, names, seen, held, last, steps, cap)
        ch = clock("decide", decider.decide, d)
        skill, target = ch["skill"]["choice"], ch["object"]["choice"]
        step = dict(step=steps + 1, skill=skill, target=target, p_skill=round(ch["skill"]["probabilities"].get(skill, 0), 3),
                    p_target=round(ch["object"]["probabilities"].get(target, 0), 3))
        steps += 1
        if ch["skill"]["probabilities"] and step["p_skill"] < ESCALATE and escalations < 2:
            escalations += 1
            replan(step, "low confidence")
            if not moves:
                break
            continue
        if skill == "done":
            left = mem.goal_left(dict(moves))
            step["left"] = left
            if VERIFY and (left or held) and rejected < VERIFY_TRIES:
                rejected += 1  # the verifier: memory says the goal is not met yet
                replan(step, "verifier: " + ", ".join(left or [f"holding {held}"]))
                if not moves:
                    break
                continue
            off_plan |= bool(held or left)
            log.append(step)
            break
        want = dict(moves)
        if (skill == "pick" and (target not in want or held)) or (skill == "place" and (not held or want.get(held) != target)):
            off_plan = True
        ok, why = False, ""
        if skill == "pick" and target in seen and not held:
            drive(arm, move_to(arm, PARK))
            settle(30)
            clock("skill_look", eyes.snap)
            everything = clock("skill_look", eyes.scene_points, min_z=-0.01)
            pts_scene = everything[everything[:, 2] > 0.004]
            pts = seen[target]["pts"]
            sup = support_height(pts, everything)
            gr = plan(pts, pts_scene, sup, avoid=failed_yaws.get(target, []), hand=arm.hand, check=arm.can_grasp)
            if gr is None:
                why = "no grasp fits"
            else:
                hover = max(HOVER_MIN, 0.06 + gr.z - gr.support + 0.025, gr.top + 0.02)
                ok, why = drive(arm, pick_at(arm, gr, hover))
                if ok:
                    held, grasps[target] = target, gr
                    mem.picked(target, arm.tip_pos()[:2])
                else:
                    failed_yaws.setdefault(target, []).append(gr.yaw)
        elif skill == "place" and held and target in planner.PLACES:
            ok_place_id = held
            gr = grasps[held]
            hang = gr.z - gr.support
            hover = max(HOVER_MIN, 0.06 + hang + 0.025, gr.top + 0.02)
            clock("skill_look", eyes.snap)
            everything = clock("skill_look", eyes.scene_points, min_z=-0.01)
            pts_scene = everything[everything[:, 2] > 0.004]
            centre = data.xpos[model.body(target).id][:2]  # fixed, calibrated place
            half = np.array(PLACE_HALF[target])
            radius = max(gr.radius + 0.005, 0.015)
            cands = [centre + np.array([dx, dy]) * (half - radius).clip(0) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
            surf_pts = everything[np.linalg.norm(everything[:, :2] - centre, axis=1) < 0.03]
            surface = float(np.median(surf_pts[:, 2])) if len(surf_pts) else 0.0
            busy = [np.sum((np.linalg.norm(pts_scene[:, :2] - c, axis=1) < radius + 0.01) & (pts_scene[:, 2] > surface + 0.004))
                    for c in cands]
            spot = cands[int(np.argmin([b + 0.1 * np.linalg.norm(c - centre) * 100 for b, c in zip(busy, cands)]))]
            ok, why = drive(arm, place_on(arm, spot, surface, hang, max(hover, surface + hang + 0.03)))
            settle(80)
            mem.placed(ok_place_id, spot, target)  # told; a landing elsewhere shows as "drop?" at the next look
            held = None
        else:
            why = "not possible now"
        skill_fail |= not ok
        last = f"{skill} {target}: {'done' if ok else 'failed'}"
        step.update(ok=bool(ok), why=why)
        if held and not arm.holding(None):  # the jaw closed on nothing: dropped; memory finds it again by appearance
            t = mem.tracks[held]
            t.status, t.expect, t.misses = "seen", None, 0
            t.history.append((mem.look, "dropped"))
            held = None
        # look again; memory keeps the ids
        drive(arm, move_to(arm, PARK))
        settle(30)
        events = observe(look_(eyes, namer if PERCEPTION == "dam" else None, keep_views=True), held)
        step["memory"] = [list(ev) for ev in events]
        log.append(step)
        names, seen = scene(held)

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
    rec.update(timing=clock.out(), steps=log, placed=placed, disturbed=disturbed, success=success, cause=cause, rejected_done=rejected,
               memory=mem.text(), memory_events=mem_events, seconds=round(time.time() - t0, 1))
    print(f"RESULT {e}: {'OK' if success else 'FAILED (' + str(cause) + ')'} plan_ok={rec['plan_ok']} "
          f"oracle_plan_ok={rec['oracle_plan_ok']} steps={len(log)} found={rec['found']}/{len(specs)} junk={rec['junk']} "
          f"rejected_done={rejected} {rec['seconds']} s", flush=True)
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
    decider = deciders.make({"jev": "jev-local", "subq": "subq", "gemma": "gemma", "gemma-vllm": "gemma-vllm", "gemma-lora": "gemma-lora"}.get(DECIDER, "rule"))
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
          f"{sum(r['plan_ok'] for r in rows)}, oracle-perception plan right {sum(r['oracle_plan_ok'] for r in rows)}"
          + (f"; gemma letter misses {decider.misses}" if hasattr(decider, "misses") else ""), flush=True)
    keys = sorted({k for r in rows for k in r.get("timing", {})})
    print("TIMING mean s per task: total %.1f; " % np.mean([r["seconds"] for r in rows])
          + ", ".join(f"{k} {np.mean([r.get('timing', {}).get(k, 0) for r in rows]):.1f}" for k in keys), flush=True)


if __name__ == "__main__":
    main()
