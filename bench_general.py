"""General pick-and-place benchmark: random objects, camera-only sensing, scripted grasping.

Each scene: N random objects (so101/objects.py) and the fixed places (red tray, blue tray, 6 cm shelf).
Two tasks per scene, "put the <colour> <noun> <place>". Per task, up to 3 attempts of:
park the arm out of view -> find the object by name with both RGB-D cameras (so101/see.py) ->
plan a grasp from its points (so101/grasp.py) -> pick; then place it on a free spot of the place.
The simulator's true state is read only to score the result.

Usage:  python bench_general.py [--scenes=20] [--objects=5] [--seed=0] [--set=primitives|scanned] [--shapes=egg,ball] [--maxr=0.23] [--finder=dino|sam3] [--gui] [--camview]
        [--robot=g1] (the Unitree G1 humanoid at the same table, right arm and Dex3 hand, so101/g1.py; default SO-101)
        [--masker=sam3|sam2-tiny] (outline model for the chosen box; default SAM 2.1 large since 2026-10-05)
        [--workers=3]   (scenes split across processes, each with its own models, ~1.5 GB GPU each)
        Real-camera tests: [--cams=cam_front] (one camera) [--depth-noise] [--pose-err=CM,DEG] (calibration error)
        [--light] (random lighting, image noise) [--fit-table] (heights from the fitted table plane)
        [--mono=small|base|large|metric-small] (no depth camera: Depth Anything depth from the colour image)
        Real-arm tests (so101/robot.py): [--joint-err=DEG] (servo calibration error, std per joint) [--play=DEG]
        (servo dead band) [--quantise] (encoder steps) [--guard=warn|stop] (safety rules count or freeze; default warn)
        [--arm-calib] (calibrate joints + camera from a marker on the gripper first, so101/armcal.py; one camera;
        [--tape=MM,DEG] how crooked the marker is taped, default 2,3)
        [--video=DIR] (one mp4 per scene: whole-scene view + camera view with what perception found; works headless)
        [--g1-grasp=chooser] (G1 picks with the learned grasp chooser on the camera points, so101/g1_campick.py;
        [--chooser-model=PATH] default runs/grasps/chooser-v3.pt; placing stays scripted; default: scripted grasp)
        [--g1-hold=touch|thumb] (G1 grasp detection: fingertip contacts, a tactile stand-in (default), or the thumb joint (old))
        [--dry-run] (rehearse each pick and place on a ghost arm first; with --gui it is shown, --ask waits for Enter)
        In the viewer, Space stops the arm (freezes it) and Space again lets it carry on.
Results: runs/general/<set>-s<seed>-n<scenes>.json (per task + summary; other robots: <robot>-<set>-...).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
SCENES, NOBJ, SEED = int(ARGS.get("scenes", 20)), int(ARGS.get("objects", 5)), int(ARGS.get("seed", 0))
SET, WORKERS = ARGS.get("set", "primitives"), int(ARGS.get("workers", 3))
GUI = "--gui" in sys.argv
DRY = "--dry-run" in sys.argv
CAMVIEW = "--camview" in sys.argv  # live camera window with detections drawn on (so101/camview.py)
VIDEO = ARGS.get("video")  # folder: one mp4 per scene, scene view + camera view with detections (so101/camview.py)
SHAPES = ARGS["shapes"].split(",") if "shapes" in ARGS else None  # e.g. --shapes=egg,ball (primitives only)
PART = ARGS.get("part")  # "i/n": this process runs scenes i, i+n, ...
OUT = Path("runs/general")
PLACES = {"red_tray": ("in the red tray", (0.045, 0.045)), "blue_tray": ("in the blue tray", (0.045, 0.045)),
          "shelf": ("on the shelf", (0.04, 0.06))}
ROBOT = ARGS.get("robot", "so101")
# where the hand waits while the cameras look: SO-101 above its base; G1 to its right, beside the blue tray
PARK = np.array({"so101": [0.10, 0.0, 0.20], "g1": [0.05, -0.30, 0.15]}[ROBOT])
G1GRASP, CHOOSER_MODEL = ARGS.get("g1-grasp", "scripted"), ARGS.get("chooser-model", "runs/grasps/chooser-v3.pt")
assert G1GRASP == "scripted" or (G1GRASP == "chooser" and ROBOT == "g1"), "--g1-grasp=chooser needs --robot=g1"
G1HOLD = ARGS.get("g1-hold", "touch" if ROBOT == "g1" else "thumb")
assert G1HOLD == "thumb" or (G1HOLD == "touch" and ROBOT == "g1"), "--g1-hold=touch needs --robot=g1"
ATTEMPTS = 4  # looks + grasps per task, including re-picks after a drop


def run_scene(scene_seed, eyes_holder):
    import so101.grasp
    so101.grasp.SMOOTH = "--smooth" in sys.argv
    so101.grasp.THIN_TALL = "--no-thin-tall" not in sys.argv
    if "fixed-clear" in ARGS:  # fixed jaw's clearance from the object's near side, metres (default 0.006)
        so101.grasp.FIXED_CLEAR = float(ARGS["fixed-clear"])
    if "zfrac" in ARGS:  # grasp-height experiments
        so101.grasp.Z_FRAC, so101.grasp.Z_BELOW_TOP = float(ARGS["zfrac"]), float(ARGS.get("zbelow", 0.02))
    from so101.grasp import HOVER_MIN, pick_at, place_on, plan, support_height
    from so101.objects import body_name, build, layout, random_objects, scanned_objects
    from so101.robot import Guard, SimArm, Stopped
    from so101.see import Eyes
    from so101.skills import Arm, move_to, run as drive

    rng = np.random.default_rng(scene_seed)
    for k in range(40):  # some draws do not fit on the table together: draw again (after 20 tries, one object fewer)
        specs = random_objects(rng, NOBJ - (k >= 20), SHAPES) if SET == "primitives" else scanned_objects(rng, NOBJ - (k >= 20))
        try:
            poses = layout(rng, specs, tries=200, max_r=float(ARGS.get("maxr", 0.23)))
            break
        except RuntimeError:
            continue
    if ROBOT == "g1":
        from so101 import g1
        ArmClass, desc, root = g1.G1Arm, g1.G1, "pelvis"
        model = build(specs, poses, g1.SCENE, g1.prepare)
    else:
        from so101.robot import SO101 as desc
        ArmClass, root = Arm, "base"
        model = build(specs, poses)
    if "noslip" in ARGS:  # MuJoCo's no-slip solver passes: removes the slow creep of soft-contact friction
        model.opt.noslip_iterations = int(ARGS["noslip"])
    if ROBOT == "so101":
        # gripper torque limit, N m. Default 1.47: LeRobot writes Max_Torque_Limit 50 % to the real gripper (the model's
        # 2.94 is the 12 V servo's full stall torque). Seed 8: 386 / 350 placed vs 372 / 345 at full torque (2026-10-03).
        g, grip = model.actuator("gripper").id, float(ARGS.get("gripforce", 1.47))
        model.actuator_forcerange[g] = [-grip, grip]
    data, scratch = mujoco.MjData(model), mujoco.MjData(model)
    if ROBOT == "g1":
        g1.start(model, data)
    if eyes_holder.get("eyes") is None:
        sim = dict(depth_noise="--depth-noise" in sys.argv, light="--light" in sys.argv,
                   pose_err=tuple(map(float, ARGS["pose-err"].split(","))) if "pose-err" in ARGS else None,
                   mono=ARGS.get("mono"))
        cams = tuple(ARGS["cams"].split(",")) if "cams" in ARGS else ("cam_left", "cam_right")
        masker = {"sam3": "facebook/sam3", "sam2-tiny": "facebook/sam2.1-hiera-tiny"}.get(ARGS.get("masker"), "facebook/sam2.1-hiera-large")
        eyes_holder["eyes"] = Eyes(model, data, cams=cams, finder=ARGS.get("finder", "dino"), sim=sim,
                                   fit_table="--fit-table" in sys.argv, masker=masker, root=root)
    eyes = eyes_holder["eyes"]
    if "known-reject" in ARGS:  # share of robot/fixture pixels that rules a mask out (0 = off)
        eyes.known_reject = float(ARGS["known-reject"])
    if "erode" in ARGS or "gate" in ARGS:  # mask-edge trimming and depth gating experiments
        eyes.erode, eyes.depth_gate = int(ARGS.get("erode", eyes.erode)), float(ARGS.get("gate", eyes.depth_gate))
    eyes.attach(model, data, seed=scene_seed)
    robot_ids = [b for b in range(model.nbody) if model.body_rootid[b] == model.body(root).id]
    sim_arm = SimArm(model, data, joint_err=float(ARGS.get("joint-err", 0)), play=float(ARGS.get("play", 0)),
                     quantise="--quantise" in sys.argv, seed=scene_seed, robot=desc)
    plug = sim_arm
    if "--arm-calib" in sys.argv and ROBOT != "so101":
        raise SystemExit("--arm-calib: SO-101 only (its marker sits on the SO-101 jaw)")
    if "--arm-calib" in sys.argv:  # camera-based arm calibration (so101/armcal.py), one camera, before the tasks
        from so101.armcal import Offset, sim_calibrate
        src = getattr(eyes.src, "inner", eyes.src)  # (MonoDepth wraps the sim cameras)
        pe = tuple(map(float, ARGS["pose-err"].split(","))) if "pose-err" in ARGS else (0.2, 0.2)
        tape = tuple(map(float, ARGS.get("tape", "2,3").split(",")))
        try:
            offsets, cal = sim_calibrate(sim_arm, src, 0, seed=scene_seed, tape=tape, cam_prior=pe)
            plug = Offset(sim_arm, offsets)
        except Exception as e:  # counted in the results (calib.failed); the scene runs uncalibrated
            cal = {"failed": f"{type(e).__name__}: {e}"}
    guard = Guard(plug, model, mode=ARGS.get("guard", "warn"), robot=desc)
    viewer = None
    if GUI:
        def key(k):
            if k == 32:  # Space: stop (or, when stopped, carry on)
                guard.pressed = True
        viewer = mujoco.viewer.launch_passive(model, data, key_callback=key)
    cam = rec = None
    if CAMVIEW:
        from so101.camview import CamView
        cam = CamView(eyes)
    elif VIDEO:
        from so101.camview import CamView, Recorder
        Path(VIDEO).mkdir(parents=True, exist_ok=True)
        cam = CamView(eyes, every=2, window=False)
        name = f"{SET}-{scene_seed}.mp4" if ROBOT == "so101" else f"{ROBOT}-{SET}-{scene_seed}.mp4"
        rec = Recorder(Path(VIDEO) / name, model, data, cam, wide=ROBOT == "g1")
    arm = ArmClass(model, data, scratch, robot=guard)
    arm.blocked_margin = 0.06
    if G1HOLD == "touch":
        from so101.g1_campick import touch_sensing
        touch_sensing(arm)
    arm.goal = PARK.copy()
    q = arm.target(arm.goal)
    for _ in range(67):  # objects settle, arm parks
        arm.step(q)

    def redraw():
        if viewer:
            viewer.sync()
        if cam:
            cam.show()
        if rec:
            rec.add()

    def run(make):
        """make(arm) -> skill. With --dry-run it is rehearsed on a ghost arm first; refused if a safety rule breaks."""
        if DRY:
            from so101.dryrun import rehearse
            broken = rehearse(arm, make, guard, viewer, ask="--ask" in sys.argv)
            if broken:
                return False, f"dry run refused: {broken[0]}"
        return drive(arm, make(arm), each=redraw)

    def settle(n=60):
        for _ in range(n):
            arm.step(arm.target(arm.goal))
            redraw()

    def stopped(e):
        """Stop pressed (or a rule broke with --guard=stop): wait for Space, then carry on from a fresh look."""
        print(f"STOPPED: {e}. Space to carry on.", flush=True)
        if cam:
            cam.status = f"STOPPED ({e}) - Space to carry on"
        guard.pressed = False
        while viewer and viewer.is_running() and not guard.pressed:
            time.sleep(0.05)
            redraw()
        guard.pressed = False
        guard.resume()

    def true_pos(spec):
        return data.xpos[model.body(body_name(spec)).id].copy()

    def at_place(spec, place):
        c, p = true_pos(spec), data.xpos[model.body(place).id]
        half = PLACES[place][1]
        top = 0.06 if place == "shelf" else 0.0  # the trays are flat pads (no collision): objects rest on the floor
        held = arm.holding(None) and np.linalg.norm(c - arm.tip_pos()) < 0.05
        # body origin: the centre for primitives, the bottom for scanned objects
        return bool(np.all(np.abs(c[:2] - p[:2]) < half) and top - 0.005 < c[2] < top + 0.1 and not held)

    order = rng.permutation(len(specs))[:2]
    places = rng.choice(list(PLACES), 2, replace=False)
    results = []
    for oi, place in zip(order, places):
        spec = specs[oi]
        r = dict(scene=scene_seed, object=spec["name"], shape=spec["shape"], place=place, attempts=0,
                 picked=False, placed=False, wrong_object=0, why=[])
        if plug is not sim_arm:  # joint error left after the fit (true minus fitted offsets), and the fit's pixel error
            r["calib"] = dict(px=round(cal["rms_px"], 2), joint_deg=np.round(np.degrees(sim_arm.err[:5] - offsets), 2).tolist())
        elif "--arm-calib" in sys.argv:
            r["calib"] = {"failed": cal["failed"]}
        t0 = time.time()
        starts = [true_pos(s_) for s_ in specs]  # truth, to count neighbours knocked (> 2 cm) during the task
        failed_yaws = []
        counts0 = dict(guard.counts)
        guard.lowest = np.inf
        for attempt in range(ATTEMPTS):
            try:
                r["attempts"] = attempt + 1
                task = f"put the {spec['name']} {PLACES[place][0]}"
                if G1GRASP == "chooser":
                    from so101 import g1_campick
                    g1_campick.reset(arm)
                if cam:
                    cam.clear()
                    cam.status = f"{task}  -  looking (try {attempt + 1})"
                run(lambda a: move_to(a, PARK))
                if arm.jaw_closed() or arm.grip != arm.grip_open:
                    arm.grip = arm.open
                settle(30)
                pts = eyes.find(spec["name"])
                if cam:
                    cam.found()
                if pts is None:
                    r["why"].append("not found")
                    continue
                near = min(specs, key=lambda s: np.linalg.norm(true_pos(s)[:2] - np.median(pts, 0)[:2]))
                if near is not spec:
                    r["wrong_object"] += 1
                src = getattr(eyes.src, "inner", eyes.src)
                if hasattr(src, "true_bodies") and eyes.last and eyes.seen.get("masks"):  # what the mask really covers
                    k, m = eyes.seen["masks"][0]
                    ids = src.true_bodies(k)[m]
                    tid = model.body(body_name(spec)).id
                    objs = {model.body(body_name(s)).id for s in specs}
                    on = ids[np.isin(ids, list(objs))]
                    top = int(np.bincount(on).argmax()) if len(on) else -1
                    r.setdefault("looks", []).append(dict(
                        near_ok=near is spec, mask_ok=top == tid, target_frac=round(float(np.mean(ids == tid)), 2),
                        other_frac=round(float(np.mean(np.isin(ids, list(objs - {tid})))), 2),
                        robot_frac=round(float(np.mean(np.isin(ids, robot_ids))), 2),
                        known_frac=round(float(np.mean(eyes.known_px[k][m])), 2),
                        what=model.body(int(np.bincount(ids[ids >= 0]).argmax())).name if (ids >= 0).any() else "none",
                        off_cm=round(float(np.linalg.norm(np.median(pts, 0)[:2] - true_pos(spec)[:2])) * 100, 2)))
                everything = eyes.scene_points(min_z=-0.01)
                scene = everything[everything[:, 2] > 0.004]
                sup = support_height(pts, everything)
                g = plan(pts, scene, sup, avoid=failed_yaws, hand=arm.hand, check=arm.can_grasp)
                if g is None:
                    r["why"].append("no grasp fits")
                    r.setdefault("no_fit", []).append(dict(  # what the camera saw, to tell bad points from big objects
                        n=len(pts), xy_extent_cm=(np.ptp(pts[:, :2], 0) * 100).round(1).tolist(),
                        z_cm=[round(float(pts[:, 2].min()) * 100, 1), round(float(pts[:, 2].max()) * 100, 1)], support_cm=round(sup * 100, 1)))
                    continue
                if cam:
                    cam.found(g)
                    cam.status = f"{task}  -  picking"
                hang = g.z - g.support
                chosen = None
                if G1GRASP == "chooser":  # the learned grasp on the same points; None: the scripted pick below
                    chosen = g1_campick.choose(arm, pts, sup, g, (scene_seed, int(oi), attempt), CHOOSER_MODEL)
                    r.setdefault("chooser", []).append(chosen["info"] if chosen else None)
                    if chosen:
                        hang = chosen["grasp"][2] - g.support
                hover = max(HOVER_MIN, 0.06 + hang + 0.025, g.top + 0.02)
                z0 = true_pos(spec)[2]
                ok, why = run(lambda a: g1_campick.pick(a, chosen, hover) if chosen else pick_at(a, g, hover))
                r.setdefault("rise_cm", []).append(round(float(true_pos(spec)[2] - z0) * 100, 1))  # truth, diagnostic
                if chosen:  # outcome with the chooser's record (so101/g1_camdata.py makes training rows of these)
                    chosen["info"].update(held=bool(ok), why=why, rise_cm=r["rise_cm"][-1], attempt=attempt)
                r["why"].append(why)
                # where the arm believes its tip is vs where it is (servo calibration error shows here)
                r.setdefault("tip_err_mm", []).append(round(float(np.linalg.norm(arm.tip_pos() - data.site_xpos[arm.tip])) * 1000, 1))
                tp = true_pos(spec)
                r.setdefault("aim_err_cm", []).append(round(float(np.linalg.norm(g.xy - tp[:2])) * 100, 2))
                r.setdefault("z_vs_centre_cm", []).append(round((g.z - tp[2]) * 100, 2))
                if not ok:
                    failed_yaws.append(g.yaw)
                    continue
                r["picked"] = True
                r["grasp"] = dict(width=g.width, gap=g.gap, z=g.z, top=g.top, blocked=g.blocked)
                # a free spot on the place: the candidate with the fewest points above its surface nearby
                centre = data.xpos[model.body(place).id][:2]  # fixed, calibrated place (like the trays' positions)
                half = np.array(PLACES[place][1])
                radius = max(g.radius + 0.005, 0.015)  # whole footprint inside the place
                cands = [centre + np.array([dx, dy]) * (half - radius).clip(0) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
                surf_pts = everything[np.linalg.norm(everything[:, :2] - centre, axis=1) < 0.03]
                surface = float(np.median(surf_pts[:, 2])) if len(surf_pts) else 0.0
                busy = [np.sum((np.linalg.norm(scene[:, :2] - c, axis=1) < radius + 0.01) & (scene[:, 2] > surface + 0.004))
                        for c in cands]
                spot = cands[int(np.argmin([b + 0.1 * np.linalg.norm(c - centre) * 100 for b, c in zip(busy, cands)]))]
                hover = max(hover, surface + hang + 0.03)
                if cam:
                    cam.clear()
                    cam.status = f"{task}  -  placing"
                ok, why = run(lambda a: g1_campick.place(a, spot, surface, hang, hover) if chosen
                              else place_on(a, spot, surface, hang, hover))
                settle(80)
                r["placed"] = at_place(spec, place)
                if "--snap" in sys.argv and not r["placed"]:  # debugging: what a failed place looks like
                    from PIL import Image
                    (OUT / "snaps").mkdir(exist_ok=True)
                    views = eyes.snap()
                    Image.fromarray(np.concatenate([v[0] for v in views], 1)).save(
                        OUT / "snaps" / f"{scene_seed}-{spec['name'].replace(' ', '_')}-{place}.png")
                    r["true_pos_cm"] = (true_pos(spec) * 100).round(1).tolist()
                r["why"].append(why if r["placed"] else f"{why}; not at place")
                if ok or "--no-recover" in sys.argv:
                    break
                # dropped on the way (the jaw reports it): look again and pick it up from wherever it fell
            except Stopped as e:
                r["why"].append(f"stopped: {e}")
                stopped(e)
        if cam:  # the outcome, held on screen for a moment
            cam.clear()
            cam.status = f"put the {spec['name']} {PLACES[place][0]}  -  {'PLACED' if r['placed'] else 'NOT placed'}"
            settle(40)
        r["disturbed"] = sum(1 for k, s_ in enumerate(specs) if k != oi and np.linalg.norm(true_pos(s_)[:2] - starts[k][:2]) > 0.02)
        r["seconds"] = round(time.time() - t0, 1)
        r["guard"] = {k: v - counts0[k] for k, v in guard.counts.items() if v - counts0[k]}
        r["lowest_jaw_mm"] = round(guard.lowest * 1000, 1)
        results.append(r)
        print(json.dumps(r), flush=True)
    if viewer:
        viewer.close()
    if cam:
        cam.close()
    if rec:
        rec.close()
    return results


def summarise(results):
    n = len(results)
    first = sum(r["picked"] and r["attempts"] == 1 for r in results)
    by_shape = {}
    for r in results:
        s = by_shape.setdefault(r["shape"], [0, 0])
        s[0] += r["placed"]
        s[1] += 1
    return {"tasks": n, "placed": sum(r["placed"] for r in results), "picked": sum(r["picked"] for r in results),
            "picked_first_try": first, "disturbed_tasks": sum(bool(r.get("disturbed")) for r in results),
            "drops": sum(w.startswith("dropped") for r in results for w in r["why"]), "wrong_object_looks": sum(r["wrong_object"] for r in results),
            "mean_seconds": float(np.mean([r["seconds"] for r in results])),
            "placed_by_shape": {k: f"{a}/{b}" for k, (a, b) in sorted(by_shape.items())}}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = [SEED * 10000 + i for i in range(SCENES)]
    if "only" in ARGS:  # chosen scene seeds, e.g. failures from an earlier run: --only=80012,80066
        seeds = [int(s) for s in ARGS["only"].split(",")]
    if PART:
        i, n = map(int, PART.split("/"))
        holder = {}
        for s in seeds[i::n]:
            run_scene(s, holder)
        sys.exit(0)
    t = time.time()
    if WORKERS > 1 and not (GUI or CAMVIEW):
        args = [a for a in sys.argv[1:] if not a.startswith("--workers")]
        procs = [subprocess.Popen([sys.executable, __file__, *args, f"--part={i}/{WORKERS}"], stdout=subprocess.PIPE,
                                  text=True) for i in range(WORKERS)]
        results = []
        for p in procs:
            for line in p.stdout:
                if line.startswith("{"):
                    results.append(json.loads(line))
                    print(line.strip(), flush=True)
            p.wait()
    else:
        holder, results = {}, []
        for s in seeds:
            results += run_scene(s, holder)
    summary = summarise(results) | {"args": sys.argv[1:], "set": SET, "scenes": SCENES, "objects": NOBJ, "seed": SEED,
                                    "wall_minutes": round((time.time() - t) / 60, 1)}
    path = OUT / f"{'' if ROBOT == 'so101' else ROBOT + '-'}{SET}-s{SEED}-n{SCENES}{'-' + ARGS['tag'] if 'tag' in ARGS else ''}.json"
    path.write_text(json.dumps({"summary": summary, "results": results}, indent=1))
    print(json.dumps(summary, indent=1))
