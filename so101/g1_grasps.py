"""Grasp sampler for the Unitree G1's right hand (so101/g1.py): many rough grasps per primitive object, executed on the
simulated robot and labelled by what happened. The successes warm-start a learned grasp (LOG.md 2026-10-06), which must
beat today's scripted grasp on the held-out lift test (`test`).

A grasp is a hand pose relative to the object (Grasp below):
  tilt     approach angle from straight down (0 = from above, pi/2 = from the side)
  azimuth  the side the hand leans in from (world angle about z): the wrist is on that side of the object
  heading  the closing direction (index pad -> thumb) about the vertical before tilting (world angle)
  height   pinch height as a share of the object's height above the table
  clear    the index pad's gap to the object's near side before closing (the thumb closes the rest), metres
  margin   hand opening beyond the object's width along the closing direction, metres
  back     pre-grasp distance back along the approach, metres
  hand     0: the thumb-index pinch; 1: the real hand's closing habit (g1.SYNERGY, from Unitree's recordings)
Executor (`run`): one object settles on the table; the arm goes from PARK over the pre-grasp, to the pre-grasp,
approaches, closes the thumb-index pinch, lifts 10 cm and holds 1 s, at a fixed hand orientation. Success from sim truth
only: object at least 9 cm above its start after the hold and within 5 cm of the pinch site. Poses the arm cannot reach
(IK (inverse kinematics) over 3 mm or 5 degrees off) or reaches only touching itself, the table or the shelf are not run.
The object's sizes come from its true geometry (`extent`); perception noise is added when training on the data.

Commands (cluster, CPU; output JSON lines, one per grasp):
  python -m so101.g1_grasps sample --shape=cylinder --n=200 --seed=1 --procs=16      random grasps -> runs/grasps/
  python -m so101.g1_grasps test --policy=scripted|chooser --n=60 --procs=16 [--noise=0.4]   held-out lift test
                                                                                (seeds 900000+; noise: perception error, cm)
  python -m so101.g1_grasps video --from=FILE.jsonl --rows=3,17 --out=x.mp4     replay grasps to mp4 (GPU job)
  python -m so101.g1_grasps stats|paired|figure ...                             summaries of sample and test files
"""
import json
import sys
import time
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

from so101 import g1

SHAPES = ("box", "cylinder", "lying cylinder", "ball", "capsule", "egg")  # so101/objects.py's primitives
TEST_SEED = 900000  # held-out layouts start here; sampling and training use seeds below
OUT = Path("runs/grasps")
LIFT, HOLD = 0.10, 33  # lift height (m) and hold (ticks, 1 s)
PARK = np.array([0.05, -0.30, 0.15])
HANDS = 1  # hand closings the sampler draws from: 1 the pinch only; 2 also the real-hand synergy (2026-10-06: it lifted
# 0 of 545 sampled grasps vs the pinch's 276 of 810, so it is off)  # where the hand waits, beside the blue tray (bench_general.py PARK for the G1)


@dataclass
class Grasp:
    tilt: float
    azimuth: float
    heading: float
    height: float
    clear: float
    margin: float
    back: float
    hand: int = 0


def random_grasp(rng, narrow):
    """The sampler's prior: wide on purpose (top-down to near side-on, any side, low to high); the heading half the time
    near the object's narrow side (`narrow`, which perception gives for long things), else anywhere."""
    heading = narrow + rng.normal(0, 0.35) + np.pi * rng.integers(2) if rng.random() < 0.5 else rng.uniform(-np.pi, np.pi)
    return Grasp(tilt=rng.uniform(0, 1.4), azimuth=rng.uniform(-np.pi, np.pi), heading=float(heading),
                 height=rng.uniform(0.2, 0.8), clear=rng.uniform(0.002, 0.012), margin=rng.uniform(0.006, 0.03),
                 back=rng.uniform(0.04, 0.1), hand=int(rng.integers(2)) if HANDS == 2 else 0)


def near(rng, g0):
    """A grasp around g0 (today's scripted one): a warm start for the sampler, small changes in every parameter."""
    return Grasp(tilt=abs(rng.normal(0, 0.25)), azimuth=rng.uniform(-np.pi, np.pi), heading=g0.heading + rng.normal(0, 0.25),
                 height=float(np.clip(g0.height + rng.normal(0, 0.12), 0.15, 0.9)), clear=rng.uniform(0.002, 0.012),
                 margin=rng.uniform(0.006, 0.03), back=float(np.clip(g0.back + rng.normal(0, 0.015), 0.03, 0.1)))


def frame(g):
    """World hand orientation (columns for DOWN, CLOSE, their cross) of grasp g: top-down at `heading`, then tilted by
    `tilt` toward `azimuth` (Rodrigues rotation about the horizontal axis that keeps the wrist on that side)."""
    a0, c0 = np.array([0.0, 0.0, -1.0]), np.array([np.cos(g.heading), np.sin(g.heading), 0.0])
    k = np.array([-np.sin(g.azimuth), np.cos(g.azimuth), 0.0])

    def rot(v):
        return v * np.cos(g.tilt) + np.cross(k, v) * np.sin(g.tilt) + k * (k @ v) * (1 - np.cos(g.tilt))
    a, c = rot(a0), rot(c0)
    return np.column_stack([a, c, np.cross(a, c)])


def extent(model, data, geom, u):
    """How far the object reaches from its centre along unit world vector u (its support function; primitives)."""
    s, v = model.geom_size[geom], data.geom_xmat[geom].reshape(3, 3).T @ u
    t = model.geom_type[geom]
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        return float(np.abs(v) @ s)
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return float(s[0])
    if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return float(s[0] * np.hypot(v[0], v[1]) + s[1] * abs(v[2]))
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return float(s[0] + s[1] * abs(v[2]))
    return float(np.linalg.norm(s * v))  # ellipsoid


def scene(seed, shape):
    """One primitive of `shape` at a benchmark spot (objects.layout), settled, with the G1's hand at PARK (put there
    before any physics step: at READY the hand is over the table and knocked objects off it). Returns the arm and the
    object's description (true centre, half sizes along x, y, z, heading of its narrow side)."""
    from so101.objects import body_name, build, layout, random_objects
    from so101.robot import Guard, SimArm
    rng = np.random.default_rng(seed)
    specs = random_objects(rng, 1, [shape])
    model = build(specs, layout(rng, specs), g1.SCENE, g1.prepare)
    data, scratch = mujoco.MjData(model), mujoco.MjData(model)
    g1.start(model, data)
    guard = Guard(SimArm(model, data, robot=g1.G1), model, mode="warn", robot=g1.G1)
    arm = g1.G1Arm(model, data, scratch, robot=guard)
    arm.goal = PARK.copy()
    q = arm.target(arm.goal)
    data.qpos[arm.qadr] = q
    data.ctrl[[model.actuator(j).id for j in g1.G1.joints]] = q
    mujoco.mj_forward(model, data)
    arm.sync()
    for _ in range(20):  # the object settles
        arm.step(q)
    geom = model.geom(body_name(specs[0])).id
    c = data.geom_xpos[geom].copy()
    half = [extent(model, data, geom, e) for e in np.eye(3)]
    yaws = np.radians(np.arange(0, 180, 5))
    narrow = float(yaws[np.argmin([extent(model, data, geom, np.array([np.cos(y), np.sin(y), 0])) for y in yaws])])
    obj = dict(shape=shape, seed=seed, centre=c.round(4).tolist(), half=np.round(half, 4).tolist(), narrow=narrow,
               mass=float(model.body_mass[model.geom_bodyid[geom]]), friction=float(model.geom_friction[geom][0]))
    return arm, guard, geom, obj


def poses(arm, geom, g, noise=None):
    """Pinch-site goals (pre-grasp, grasp), hand orientation and opening for grasp g on the object. noise: perception
    error as (centre offset, size scale), e.g. from `perceived`."""
    m, d = arm.model, arm.data
    F = frame(g)
    a, c = F[:, 0], F[:, 1]
    dc, scale = noise if noise is not None else (np.zeros(3), 1.0)
    centre = d.geom_xpos[geom] + dc
    top = centre[2] + scale * extent(m, d, geom, np.array([0.0, 0.0, 1.0]))
    width = 2 * scale * extent(m, d, geom, c)
    target = np.array([centre[0], centre[1], g.height * top])
    grasp = target + c * (g1.Hand.FIXED_OFFSET - width / 2 - g.clear)  # the index pad `clear` from the object
    from so101.grasp import gap_to_q
    opening = 1.0 if g.hand else float(np.clip(gap_to_q(width + g.clear + g.margin, g1.Hand), 0, 1))
    return grasp - a * g.back, grasp, F, opening, width


def perceived(rng, cm=0.4):
    """A perception error like the detector's (LOG.md 2026-10-03: centre within a few mm, sizes a little off): centre
    offset with `cm` cm standard deviation per axis, sizes scaled by 1 +- 5 %."""
    return rng.normal(0, cm / 100, 3), float(1 + rng.normal(0, 0.05))


def check(arm, goals, F):
    """Posture (from g1.POSTURES) that reaches every goal within 3 mm and 5 degrees with the right arm and hand
    touching nothing but the object (tilted hands can pass through the table: only grazing finger tips allowed);
    else None, why and what (the closest miss in mm / degrees, or the contact pairs)."""
    why, what = "unreachable", None
    for posture in g1.POSTURES:
        ok = True
        for p in goals:
            q, err, ang = g1.ik(arm, p, frame=F, posture=posture, report=True)
            if err > 0.003 or ang > np.radians(5):
                if why == "unreachable" and (what is None or err * 1000 + np.degrees(ang) < sum(what)):
                    what = (round(err * 1000, 1), round(float(np.degrees(ang)), 1))
                ok = False
                break
            c = arm.clashes(q)
            if c or touches_fixed(arm):
                ok, why, what = False, "contact", sorted(set(c))[:3] or "hand-table"
                break
        if ok:
            return posture, "ok", None
    return None, why, what


def touches_fixed(arm, depth=0.002):
    """Any right-hand body more than `depth` into the table or shelf in the pose clashes() just left on scratch (finger
    tips grazing the table at the lowest top-down grasps are allowed, as in the benchmark)."""
    m, s = arm.model, arm.scratch
    b1, b2 = m.geom_bodyid[s.contact.geom1], m.geom_bodyid[s.contact.geom2]
    mv, fixed = arm.moving, arm.fixed
    return bool(np.any(((mv[b1] & fixed[b2]) | (mv[b2] & fixed[b1])) & (s.contact.dist < -depth)))


def run(arm, guard, geom, g, record=False, checked=None, noise=None):
    """Execute grasp g (checked: check()'s answer if already known; noise: aim with a perception error, see poses);
    returns (outcome dict, command sequence or None)."""
    from so101.skills import SPEED, move_to, run as drive, set_grip, settle
    pre, grasp, F, opening, width = poses(arm, geom, g, noise)
    posture, why, what = checked or check(arm, [pre, grasp], F)
    out = dict(width=round(width, 4), opening=round(opening, 3))
    if posture is None:
        return out | dict(result=why, why=what, success=False), None
    d, b = arm.data, arm.model.geom_bodyid[geom]
    z0, p0 = d.xpos[b][2], d.xpos[b].copy()
    cmds = []
    if record:
        step = arm.step
        arm.step = lambda q: (cmds.append(np.round(q, 4).tolist()), step(q))
    arm.frame, arm.posture, arm.synergy = F, posture, g1.SYNERGY if g.hand else None
    above = np.array([*pre[:2], max(pre[2], 0.12)])
    drive(arm, move_to(arm, above))
    drive(arm, move_to(arm, pre))
    drive(arm, set_grip(arm, opening))
    drive(arm, move_to(arm, grasp, speed=SPEED / 4))
    pushed = float(np.linalg.norm(d.xpos[b][:2] - p0[:2]))  # how far the approach shoved it before closing
    drive(arm, set_grip(arm, 0.0))
    drive(arm, move_to(arm, grasp + [0, 0, LIFT]))
    drive(arm, settle(arm, arm.target(arm.goal), HOLD, HOLD))
    rise = float(d.xpos[b][2] - z0)
    near = float(np.linalg.norm(d.xpos[b] - d.site_xpos[arm.tip]))
    ok = rise > LIFT - 0.01 and near < 0.05
    out |= dict(result="lifted" if ok else "dropped" if rise > 0.02 else "missed", success=bool(ok), rise=round(rise, 4),
                pushed_cm=round(pushed * 100, 2), guard={k: v for k, v in guard.counts.items() if v})
    return out, (cmds if record else None)


def scripted(arm, geom, obj):
    """Today's planner grasp (grasp.plan on the true object's top-down points: heading, height, opening) as a Grasp,
    the thumb on whichever side the arm reaches cleanly (G1Arm.choose)."""
    from so101.grasp import plan
    m, d = arm.model, arm.data
    b, (x, y, _) = m.geom_bodyid[geom], obj["centre"]
    pts, gid = [], np.array([-1], np.int32)
    for dx in np.arange(-0.06, 0.06, 0.002):
        for dy in np.arange(-0.06, 0.06, 0.002):
            p0 = np.array([x + dx, y + dy, 0.3])
            r = mujoco.mj_ray(m, d, p0, np.array([0, 0, -1.0]), None, 1, -1, gid)
            if r > 0 and m.geom_bodyid[gid[0]] == b:
                pts.append(p0 - [0, 0, r])
    p = plan(np.array(pts).reshape(-1, 3), np.zeros((0, 3)), 0.0, hand=g1.Hand, check=arm.can_grasp)
    if p is None:
        return None
    _, heading, _ = arm.choose(np.array([*p.xy, p.z]), [p.yaw, p.yaw + np.pi])
    top = obj["centre"][2] + obj["half"][2]
    return Grasp(tilt=0.0, azimuth=0.0, heading=float(heading), height=float(p.z / top), clear=g1.Hand.FIXED_CLEAR,
                 margin=g1.Hand.MOVING_CLEAR, back=0.04)


CHOOSER = None  # g1_chooser.Chooser, loaded once per process


def one(job):
    """One grasp: job = (seed, shape, policy, record, retest, noise). A sampled grasp that lifts is run `retest` more
    times on the same layout, aimed with a perception error (`perceived`): its robustness. noise (cm, test only): aim
    every policy with a perception error of that size (the chooser also sees the object that way)."""
    global CHOOSER
    seed, shape, policy, record, retest, noise_cm = job
    t = time.time()
    arm, guard, geom, obj = scene(seed, shape)
    noise = perceived(np.random.default_rng([seed, 13]), noise_cm) if noise_cm else None
    checked, rejects, score = None, {}, None
    if policy in ("chooser", "residual"):
        from so101.g1_chooser import Chooser, features
        CHOOSER = CHOOSER or Chooser()
        seen = obj if noise is None else dict(obj, centre=list(np.add(obj["centre"], noise[0])),
                                              half=list(np.multiply(obj["half"], noise[1])))
        g, checked, score = CHOOSER.choose(arm, geom, seen, np.random.default_rng([seed, 17]))
        if g is None:
            return dict(obj=obj, grasp=None, policy=policy, result="nothing reachable", success=False)
        if policy == "residual":  # the learned correction's mean (so101/g1_residual.py), then a fresh reach check
            from so101.g1_residual import PATH, Residual, correct
            phi = features(seen, asdict(g))
            g2 = correct(g, Residual(len(phi), PATH).mean(phi))
            pre, grasp, F, _, _ = poses(arm, geom, g2)
            c2 = check(arm, [pre, grasp], F)
            if c2[0] is not None:  # else keep the chooser's grasp
                g, checked = g2, c2
    elif policy in ("random", "near"):  # draw until the arm can do one (up to 80 draws; rejected ones are counted)
        rng = np.random.default_rng([seed, 7])
        g0 = scripted(arm, geom, obj) if policy == "near" else None
        if policy == "near" and g0 is None:
            return dict(obj=obj, grasp=None, policy=policy, result="no grasp fits", success=False)
        for _ in range(80):
            g = random_grasp(rng, obj["narrow"]) if g0 is None else near(rng, g0)
            pre, grasp, F, _, _ = poses(arm, geom, g)
            checked = check(arm, [pre, grasp], F)
            if checked[0] is not None:
                break
            rejects[checked[1]] = rejects.get(checked[1], 0) + 1
    else:
        g = scripted(arm, geom, obj)
        if g is None:
            return dict(obj=obj, grasp=None, policy=policy, result="no grasp fits", success=False)
    out, cmds = run(arm, guard, geom, g, record, None if noise is not None else checked, noise)
    row = dict(obj=obj, grasp={k: round(v, 4) for k, v in asdict(g).items()}, policy=policy, **out, rejects=rejects,
               seconds=round(time.time() - t, 1))
    if score is not None:
        row["predicted"] = round(score, 3)
    if cmds is not None and out["success"]:
        row["commands"] = cmds  # per tick: ARM then HAND joint targets (g1.G1.joints)
    if out["success"] and retest:
        rng, ok = np.random.default_rng([seed, 11]), 0
        for _ in range(retest):
            arm, guard, geom, _ = scene(seed, shape)
            ok += run(arm, guard, geom, g, noise=perceived(rng))[0]["success"]
        row["robust"] = round(ok / retest, 2)
    return row


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def video(rows, path):
    """Replay grasps (rows of a sampler or test file) on the simulator into one mp4: two views, close-up from the
    front-left and the whole robot, 15 frames/s (needs a GPU job for headless rendering: MUJOCO_GL=egl)."""
    import av
    out, stream = av.open(str(path), "w"), None
    for row in rows:
        arm, guard, geom, obj = scene(row["obj"]["seed"], row["obj"]["shape"])
        g = Grasp(**row["grasp"])
        m, d = arm.model, arm.data
        r = mujoco.Renderer(m, 360, 480)
        views = []
        for look, dist, az, el in ((obj["centre"], 0.45, 150, -20), ((0.05, 0, 0.15), 1.4, 160, -20)):
            v = mujoco.MjvCamera()
            v.lookat[:], v.distance, v.azimuth, v.elevation = look, dist, az, el
            views.append(v)
        step, tick = arm.step, [0]

        def filmed(q):
            nonlocal stream
            step(q)
            tick[0] += 1
            if tick[0] % 2:
                return
            frames = []
            for v in views:
                r.update_scene(d, v)
                frames.append(r.render())
            frame = np.concatenate(frames, 1)
            if stream is None:
                stream = out.add_stream("libx264", rate=15)
                stream.width, stream.height, stream.pix_fmt = frame.shape[1], frame.shape[0], "yuv420p"
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                out.mux(packet)
        arm.step = filmed
        res, _ = run(arm, guard, geom, g)
        print(row["obj"]["shape"], row["obj"]["seed"], res["result"], flush=True)
    for packet in stream.encode():
        out.mux(packet)
    out.close()


def tests(name, n=60):
    """Held-out lift test rows of a policy, 'policy[:tag]' -> runs/grasps/test-<policy>-<shape>-n<n>[-<tag>].jsonl,
    keyed by (shape, layout seed)."""
    import glob
    policy, _, tag = name.partition(":")
    files = glob.glob(str(OUT / f"test-{policy}-*-n{n}{'-' + tag if tag else ''}.jsonl"))
    return {(r["obj"]["shape"], r["obj"]["seed"]): r for f in files for r in map(json.loads, open(f))}


def paired(a, b):
    """Two policies on the same held-out layouts: lifted by each, layouts only one lifted, exact McNemar p."""
    from math import comb
    A, B = tests(a), tests(b)
    for s in (*SHAPES, None):
        keys = [k for k in A if k in B and (s is None or k[0] == s)]
        ka = sum(A[k]["success"] and not B[k]["success"] for k in keys)
        kb = sum(B[k]["success"] and not A[k]["success"] for k in keys)
        p = min(1.0, 2 * sum(comb(ka + kb, i) for i in range(min(ka, kb) + 1)) / 2 ** (ka + kb)) if ka + kb else 1.0
        print(f"{s or 'all':15s} layouts {len(keys)}: {a} {sum(A[k]['success'] for k in keys)}, {b}"
              f" {sum(B[k]['success'] for k in keys)}; only {a} {ka}, only {b} {kb}; McNemar p {p:.3f}")


def figure(names, path):
    """Lifted share per shape class with Wilson 95 % ranges for each policy (bars side by side) -> png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.2))
    w = 0.8 / len(names)
    for j, name in enumerate(names):
        rows = tests(name)
        for i, s in enumerate(SHAPES):
            r = [x["success"] for k, x in rows.items() if k[0] == s]
            k, n = sum(r), max(len(r), 1)
            lo, hi = wilson(k, len(r))
            x = i + (j - (len(names) - 1) / 2) * w
            ax.bar(x, k / n, w, color=f"C{j}", label=name if i == 0 else None)
            ax.errorbar(x, k / n, [[k / n - lo], [hi - k / n]], color="k", capsize=3, lw=1)
    ax.set_xticks(range(len(SHAPES)), SHAPES)
    ax.set_ylabel("lifted 10 cm and held 1 s")
    ax.set_ylim(0, 1)
    ax.set_title("G1 + Dex3, held-out lift test (60 layouts per class, Wilson 95 % ranges)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(path)


def stats(paths):
    """Lifted share of executed grasps per shape, by hand closing and by approach tilt; mean robustness of the lifts."""
    import glob
    rows = [json.loads(l) for a in paths for f in glob.glob(a) for l in open(f)]
    rows = [r for r in rows if r.get("grasp") and r["result"] in ("lifted", "missed", "dropped")]
    for s in SHAPES:
        r = [x for x in rows if x["obj"]["shape"] == s]
        if not r:
            continue
        cell = lambda xs: f"{sum(x['success'] for x in xs)}/{len(xs)}"
        hands = {h: cell([x for x in r if x["grasp"].get("hand", 0) == h]) for h in (0, 1)}
        tilts = {f"{lo}-{lo + 20}": cell([x for x in r if lo <= np.degrees(x["grasp"]["tilt"]) < lo + 20]) for lo in range(0, 81, 20)}
        rob = [x["robust"] for x in r if "robust" in x]
        print(f"{s:15s} lifted {cell(r):8s} pinch {hands[0]:8s} real-hand synergy {hands[1]:8s} tilt (deg) {tilts}"
              f" robust {np.mean(rob) if rob else float('nan'):.2f}")


def main():
    args = dict(a[2:].split("=", 1) for a in sys.argv[2:] if a.startswith("--") and "=" in a)
    cmd, procs, n = sys.argv[1], int(args.get("procs", 1)), int(args.get("n", 20))
    if cmd == "stats":  # python -m so101.g1_grasps stats 'runs/grasps/sample-*.jsonl'
        return stats(sys.argv[2:])
    if cmd == "paired":  # python -m so101.g1_grasps paired scripted chooser:v2
        return paired(*sys.argv[2:4])
    if cmd == "figure":  # python -m so101.g1_grasps figure scripted chooser chooser:v2 --out=runs/grasps/lift_test.png
        return figure([a for a in sys.argv[2:] if not a.startswith("--")], args.get("out", OUT / "lift_test.png"))
    if cmd == "video":  # --from=FILE.jsonl --rows=3,17 (line numbers) --out=x.mp4
        lines = open(args["from"]).read().splitlines()
        rows = [json.loads(lines[int(i)]) for i in args["rows"].split(",")]
        return video(rows, Path(args.get("out", OUT / "video.mp4")))
    shapes = [s.replace("-", " ") for s in args["shape"].split(",")] if "shape" in args else SHAPES
    if cmd == "sample":
        seed = int(args.get("seed", 0))
        assert (seed + 1) * 100000 <= TEST_SEED, "sampling seeds must stay below the held-out ones"
        # random: the wide prior; near: around today's scripted grasp; chooser: the learned chooser's picks (on-policy
        # data for retraining it, DAgger-style)
        prior = args.get("prior", "random")
        jobs = [(seed * 100000 + i, s, prior, True, int(args.get("retest", 0)), 0) for s in shapes for i in range(n)]
        name = f"sample-{'_'.join(s.replace(' ', '-') for s in shapes)}-s{seed}-n{n}{'' if prior == 'random' else '-' + prior}"
    else:  # test: held-out layouts, the same ones for every policy
        policy = args.get("policy", "scripted")
        noise = float(args.get("noise", 0))
        jobs = [(TEST_SEED + i, s, policy, False, 0, noise) for s in shapes for i in range(n)]
        name = f"test-{policy}-{'_'.join(s.replace(' ', '-') for s in shapes)}-n{n}{f'-noise{noise:g}' if noise else ''}"
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}{'-' + args['tag'] if 'tag' in args else ''}.jsonl"
    t, rows = time.time(), []
    with Pool(procs) as pool, open(path, "w") as f:
        for row in pool.imap_unordered(one, jobs):
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
    print(f"{path}: {len(rows)} grasps, {(time.time() - t) / 60:.1f} min on {procs} processes")
    for s in shapes:
        r = [x for x in rows if x["obj"]["shape"] == s]
        k = sum(x["success"] for x in r)
        lo, hi = wilson(k, len(r))
        why = {w: sum(x["result"] == w for x in r) for w in sorted({x["result"] for x in r})}
        print(f"  {s:15s} lifted {k}/{len(r)} ({lo:.0%}-{hi:.0%})  {why}")


if __name__ == "__main__":
    main()
