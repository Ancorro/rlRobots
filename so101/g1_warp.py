"""Feasibility study: the G1 + Dex3 lift trial (so101/g1_grasps.py) on the GPU with MuJoCo Warp (MJWarp), many worlds in
parallel, as a first step toward closed-loop RL on the G1 (LOG.md 2026-10-06: on CPU MuJoCo the G1 runs ~1.8x slower
than real time).

Findings that shaped this file (MJWarp 3.14.0, warp-lang 1.17.0):
  - MJWarp does step the original scene's convex mesh colliders, but slowly (16 ms per step for 4 worlds on a 4060 Ti,
    ~22 contacts per world at rest: feet on the ground, legs and arm against the torso), and it refuses
    `noslip_iterations` (objects.build sets 3). So the GPU scene has simplified collisions and no noslip passes.
  - Simplified scene (`simplify`, written to so101/assets/scene_g1_simple.xml by `make-scene`): with the pelvis welded
    and the left arm parked, the legs, feet, left arm and head cannot touch anything that matters: their collisions are
    off. Kept: pelvis and torso (the right arm can hit them), the right arm and hand, the table, the shelf, the object.
    Mesh colliders that stay are reduced to small convex hulls (`maxhullvert`), or, at level "prim", replaced with
    primitives fitted to the meshes (the finger links keep small hulls: their pads decide grasps).
  - The grasp executor is not purely open-loop (skills.settle waits for the joints, the IK warm-starts from the
    measured pose), so the GPU does not re-run its logic: the CPU run on the original scene records every tick's
    actuator targets (a "tape", failures included), and every column of the comparison replays the same tape from the
    same start state. That isolates the physics (scene, noslip, engine) from the planner.
  - MJWarp batches worlds of one model structure: geom types cannot differ per world, so one batch holds one object
    class; sizes, masses, friction, inertias and the start pose differ per world (`batch_model`).

Commands (cluster; .venv-warp; see warp/*.sbatch):
  python -m so101.g1_warp make-scene                     write so101/assets/scene_g1_simple.xml (local, once)
  python -m so101.g1_warp tapes --n=20 --procs=16        CPU: scripted grasp on training layouts (seeds 800000+),
                                                         live outcome + tape -> runs/warp/tapes-n<n>.pkl
  python -m so101.g1_warp cpu --tapes=FILE --scene=orig|simple --noslip=3|0 --procs=16    CPU replay column
  python -m so101.g1_warp gpu --tapes=FILE                MJWarp replay column (simplified scene, noslip 0)
  python -m so101.g1_warp speed --tapes=FILE --worlds=1,64,512,4096     GPU throughput; --cpu: CPU baseline
  python -m so101.g1_warp report                          fidelity and speed tables, throughput figure
"""
import json
import pickle
import re
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from so101 import g1

ASSETS = Path(__file__).parent / "assets"
SIMPLE = ASSETS / "scene_g1_simple.xml"
OUT = Path("runs/warp")
SEED0 = 800000  # fidelity layouts: training seeds (below g1_grasps.TEST_SEED = 900000), unused by the sampler (<= 600000)
TICK_STEPS = g1.G1.steps  # physics steps per 30 ms control tick (15 x 2 ms)
LIFT = 0.10
IMPRATIO = 10  # with elliptic cones (fixed before the comparison, not tuned on it)

# bodies whose collisions stay on: the right arm and hand can touch the pelvis and torso, the table, the shelf, objects
KEEP = ("pelvis", "torso_link")
HULL = 24  # max convex-hull vertices of the mesh colliders kept


def simplify(spec, level="hull"):
    """Edit a scene_g1.xml spec (before g1.prepare) for MJWarp. level "cull": collisions off for everything the welded
    robot cannot use (legs, feet, left arm, head, the ground plane), other colliders unchanged; "hull": also every
    remaining mesh collider reduced to a HULL-vertex convex hull; "prim": "hull" plus the pelvis, torso and right
    arm links (not the hand's fingers) as primitives fitted to their meshes (boxes; MuJoCo fits them at compile time)."""
    right = set()

    def walk(b, inside):
        inside = inside or b.name == "right_shoulder_pitch_link"
        if inside:
            right.add(b.name)
        for c in b.bodies:
            walk(c, inside)
    walk(spec.worldbody, False)
    fingers = {n for n in right if n.startswith("right_hand_") and n != "right_hand_palm_link"}
    for gm in spec.geoms:
        if not (gm.contype or gm.conaffinity):
            continue
        body = gm.parent.name if gm.parent is not None else "world"
        if gm.name == "ground" or (body not in right and body not in KEEP and body != "world" and body != "shelf"):
            gm.contype = gm.conaffinity = 0
            continue
        if gm.type != mujoco.mjtGeom.mjGEOM_MESH or level == "cull":
            continue
        if level == "prim" and body not in fingers:
            gm.type = mujoco.mjtGeom.mjGEOM_BOX  # MuJoCo fits it to the mesh at compile time (equivalent-inertia box)
        else:
            spec.mesh(gm.meshname).maxhullvert = int(level[4:] or HULL) if level.startswith("hull") else HULL
    return spec


def make_scene(level="hull", path=SIMPLE):
    spec = mujoco.MjSpec.from_file(str(ASSETS / "scene_g1.xml"))
    simplify(spec, level)
    xml = spec.to_xml()
    for ms in spec.meshes:  # to_xml (MuJoCo 3.14) drops mesh maxhullvert: put it back
        if ms.maxhullvert > 0:
            xml, k = re.subn(f'<mesh name="{ms.name}"', f'<mesh name="{ms.name}" maxhullvert="{ms.maxhullvert}"', xml)
            assert k == 1, ms.name
    xml = xml.replace(' nkey="1"', "")  # to_xml sizes the keyframe list; g1.prepare deletes the keyframe anyway
    header = (f"<!-- Generated by `python -m so101.g1_warp make-scene` (level {level}) from scene_g1.xml + Menagerie "
              f"unitree_g1/g1_with_hands.xml (BSD-3): collision geometry simplified for MuJoCo Warp (so101/g1_warp.py "
              f"simplify). Visual meshes unchanged. Do not edit by hand. -->\n")
    path.write_text(header + xml)
    print(path)


# --- tapes: the scripted grasp on CPU, every tick's actuator targets recorded ----------------------------------------
def model_for(seed, shape, scene=None, noslip=None, friction=None):
    """The layout g1_grasps.scene(seed, shape) builds (same random draws), on `scene` (default scene_g1.xml);
    noslip: override objects.build's noslip_iterations (3); friction "elliptic": elliptic friction cones with
    impratio IMPRATIO (the usual stand-in for noslip in MJX / MJWarp manipulation scenes) instead of pyramidal ones."""
    from so101.objects import build, layout, random_objects
    rng = np.random.default_rng(seed)
    specs = random_objects(rng, 1, [shape])
    model = build(specs, layout(rng, specs), scene or g1.SCENE, g1.prepare)
    if noslip is not None:
        model.opt.noslip_iterations = noslip
    if friction and friction.startswith("elliptic"):  # "elliptic" (impratio IMPRATIO) or "ellipticN" (impratio N, exploratory)
        model.opt.cone, model.opt.impratio = mujoco.mjtCone.mjCONE_ELLIPTIC, float(friction[8:] or IMPRATIO)
    return model


def tape(job):
    """job = (seed, shape): the scripted grasp of g1_grasps, live on CPU on the original scene, with the actuator
    targets of every tick recorded (from the first settle tick of g1_grasps.scene). Returns the live outcome, the start
    state and the tape; t0 = ticks before the grasp starts (the object's start height is read there, as run() does)."""
    from dataclasses import asdict
    from so101 import g1_grasps as G
    from so101.robot import SimArm
    seed, shape = job
    rec, tick0 = dict(ctrl=[], qpos=None), SimArm.tick

    def tick(self):
        if rec["qpos"] is None:
            rec["qpos"] = self.data.qpos.copy()
            assert not self.data.qvel.any()
        rec["ctrl"].append(self.data.ctrl.copy())
        tick0(self)
    SimArm.tick = tick
    t = time.time()
    try:
        arm, guard, geom, obj = G.scene(seed, shape)
        t0 = len(rec["ctrl"])
        g = G.scripted(arm, geom, obj)
        if g is None:
            return dict(obj=obj, live=dict(result="no grasp fits", success=False))
        out, _ = G.run(arm, guard, geom, g)
    finally:
        SimArm.tick = tick0
    if "rise" not in out:  # not executed (unreachable, or only with contact): nothing to replay
        return dict(obj=obj, grasp=asdict(g), live=out)
    body = arm.model.body(arm.model.geom_bodyid[geom]).name
    return dict(obj=obj, body=body, grasp=asdict(g), live=out, t0=t0, qpos=rec["qpos"], ctrl=np.array(rec["ctrl"]),
                seconds=round(time.time() - t, 1))


def outcome(model, data, body, z0):
    """The success rule of g1_grasps.run: the object 9 cm above its start, within 5 cm of the pinch site."""
    b, s = model.body(body).id, model.site("pinch").id
    rise = float(data.xpos[b][2] - z0)
    near = float(np.linalg.norm(data.xpos[b] - data.site_xpos[s]))
    return dict(success=bool(rise > LIFT - 0.01 and near < 0.05), rise=round(rise, 4), near=round(near, 4))


def replay(rec, model):
    """Replay a tape on a CPU model from the recorded start state; outcome as run() reads it. Also the peak contacts
    and constraint rows (to size MJWarp's buffers) and the physics time."""
    d = mujoco.MjData(model)
    d.qpos[:] = rec["qpos"]
    d.ctrl[:] = rec["ctrl"][0]
    mujoco.mj_forward(model, d)
    b, ncon, nefc, z0, dt = model.body(rec["body"]).id, 0, 0, None, 0.0
    for i, c in enumerate(rec["ctrl"]):
        if i == rec["t0"]:
            z0 = d.xpos[b][2]
        d.ctrl[:] = c
        t = time.perf_counter()
        for _ in range(TICK_STEPS):
            mujoco.mj_step(model, d)
            ncon, nefc = max(ncon, d.ncon), max(nefc, d.nefc)
        dt += time.perf_counter() - t
    return outcome(model, d, rec["body"], z0) | dict(ncon=ncon, nefc=nefc, physics_s=round(dt, 3),
                                                     steps=len(rec["ctrl"]) * TICK_STEPS)


def key(rec):
    return rec["obj"]["shape"], rec["obj"]["seed"]


def perturbed(rec, how):
    """The sensitivity floor: the same tape with a perturbation far below anything physical. "f32": targets rounded to
    float32 (what the GPU replays); "jitter": the object's start position moved by ~1 micrometre (seeded per layout)."""
    rec = dict(rec)
    if how == "f32":
        rec["ctrl"] = rec["ctrl"].astype(np.float32).astype(np.float64)
    elif how == "jitter":
        q = rec["qpos"].copy()
        q[-7:-4] += np.random.default_rng(rec["obj"]["seed"]).normal(0, 1e-6, 3)  # the object's free joint is last
        rec["qpos"] = q
    return rec


def cpu_job(job):
    rec, scene, noslip, friction, perturb = job
    if "ctrl" not in rec:
        return dict(key=key(rec), success=None)
    if perturb:
        rec = perturbed(rec, perturb)
    model = model_for(rec["obj"]["seed"], rec["obj"]["shape"], scene, noslip, friction)
    return dict(key=key(rec), **replay(rec, model))


# --- MJWarp: one object class per batch, per-world model fields ------------------------------------------------------
def batch_model(models):
    """mjw.Model for worlds that share a structure (one object class) but differ in numbers: every batchable ("*")
    Model field that differs between the CPU models gets one row per model (world w uses row w % len(models): MJWarp
    indexes batched fields that way). Fails on a difference MJWarp cannot batch, or on a field put_model transforms."""
    import dataclasses
    import mujoco_warp as mjw
    import warp as wp
    from mujoco_warp._src import types
    spec = {f.name: getattr(f.type, "shape", ()) for f in dataclasses.fields(types.Model)}
    diff, unbatchable = {}, []
    for name in dir(models[0]):
        v0 = getattr(models[0], name)
        if name.startswith("_") or not isinstance(v0, np.ndarray) or name not in spec:
            continue
        vals = [np.asarray(getattr(m, name)) for m in models]
        if all(v.shape == v0.shape and np.array_equal(v, v0) for v in vals[1:]) or name in UNUSED:
            continue
        if spec[name] and spec[name][0] == "*":
            diff[name] = np.stack(vals)
        else:
            unbatchable.append(name)
    assert not unbatchable, f"fields differ between layouts but MJWarp cannot batch them: {unbatchable}"
    mm = mjw.put_model(models[0], batch_sizes={k: len(models) for k in diff})
    for name, v in diff.items():
        arr = getattr(mm, name)
        a = arr.numpy()
        assert np.allclose(a[0].ravel(), v[0].ravel(), atol=1e-6), f"put_model transforms {name}: cannot batch it raw"
        setattr(mm, name, wp.array(v.reshape((len(models),) + a.shape[1:]).astype(a.dtype), dtype=arr.dtype))
    return mm, sorted(diff)


_KERNELS = None
UNUSED = ("dof_length", "key_qpos")  # differ per layout, read only by MJWarp's sleeping and keyframe reset (unused here)


def kernels():
    global _KERNELS
    if _KERNELS:
        return _KERNELS
    import warp as wp

    @wp.kernel
    def set_ctrl(tick: int, tapes: wp.array3d(dtype=float), tlen: wp.array(dtype=int), ctrl: wp.array2d(dtype=float)):
        w, a = wp.tid()
        k = w % tapes.shape[0]
        ctrl[w, a] = tapes[k, wp.min(tick, tlen[k] - 1), a]

    @wp.kernel
    def snap(done: int, t0: wp.array(dtype=int), tlen: wp.array(dtype=int), body: int, site: int,
             xpos: wp.array2d(dtype=wp.vec3), site_xpos: wp.array2d(dtype=wp.vec3), z0: wp.array(dtype=float),
             end_obj: wp.array(dtype=wp.vec3), end_site: wp.array(dtype=wp.vec3)):
        w = wp.tid()
        k = w % tlen.shape[0]
        if done == t0[k]:
            z0[w] = xpos[w, body][2]
        if done == tlen[k]:
            end_obj[w] = xpos[w, body]
            end_site[w] = site_xpos[w, site]

    @wp.kernel
    def track(nacon: wp.array(dtype=int), nefc: wp.array(dtype=int), peak: wp.array(dtype=int)):
        w = wp.tid()
        wp.atomic_max(peak, 0, nacon[0])
        wp.atomic_max(peak, 1, nefc[w])
    _KERNELS = set_ctrl, snap, track
    return _KERNELS


class Batch:
    """Replay the tapes of one object class in `nworld` MJWarp worlds (world w runs tape w % len(recs)) on `scene`
    (noslip off: MJWarp has none). nconmax: contacts per world (shared pool), njmax: constraint rows per world."""

    def __init__(self, recs, nworld, scene=SIMPLE, nconmax=48, njmax=320, friction=None):
        import mujoco_warp as mjw
        import warp as wp
        self.recs, self.n = recs, nworld
        models = [model_for(r["obj"]["seed"], r["obj"]["shape"], scene, 0, friction) for r in recs]
        self.mm, self.batched = batch_model(models)
        m0 = models[0]
        self.body, self.site = m0.body(recs[0]["body"]).id, m0.site("pinch").id
        assert all(m.body(r["body"]).id == self.body for m, r in zip(models, recs))
        self.d = mjw.put_data(m0, mujoco.MjData(m0), nworld=nworld, nconmax=nconmax, njmax=njmax)
        k = np.arange(nworld) % len(recs)
        self.qpos0 = np.stack([r["qpos"] for r in recs])[k]
        T = max(len(r["ctrl"]) for r in recs)
        tapes = np.stack([np.vstack([r["ctrl"], np.repeat(r["ctrl"][-1:], T - len(r["ctrl"]), 0)]) for r in recs])
        self.T, self.tapes = T, wp.array(tapes.astype(np.float32), dtype=float)
        self.lens = np.array([len(r["ctrl"]) for r in recs])
        self.tlen = wp.array(self.lens.astype(np.int32), dtype=int)
        self.t0 = wp.array(np.array([r["t0"] for r in recs], np.int32), dtype=int)
        self.z0 = wp.zeros(nworld, dtype=float)
        self.end_obj, self.end_site = wp.zeros(nworld, dtype=wp.vec3), wp.zeros(nworld, dtype=wp.vec3)
        self.peak = wp.zeros(2, dtype=int)
        self.set_ctrl, self.snap, self.track = kernels()
        self.reset()
        with wp.ScopedCapture() as cap:  # one control tick of physics (15 steps) as a CUDA graph
            for _ in range(TICK_STEPS):
                mjw.step(self.mm, self.d)
        self.graph = cap.graph

    def reset(self):
        """Every world at its tape's start state (qpos, zero velocity, first targets), as replay() starts on CPU."""
        import mujoco_warp as mjw
        import warp as wp
        self.d.qpos.assign(self.qpos0)
        self.d.qvel.zero_()
        self.d.qacc_warmstart.zero_()
        self.d.time.zero_()
        wp.launch(self.set_ctrl, self.d.ctrl.shape, inputs=[0, self.tapes, self.tlen, self.d.ctrl])
        mjw.forward(self.mm, self.d)

    def run(self):
        """All tapes to their end (shorter ones hold their last targets meanwhile; each world's outcome is read at its
        own tape end); returns per-world outcomes and the wall time (control upload included, set-up excluded)."""
        import warp as wp
        self.reset()
        self.peak.zero_()
        self.d.overflow.zero_()
        wp.synchronize()
        t = time.perf_counter()
        for tick in range(self.T):
            wp.launch(self.set_ctrl, self.d.ctrl.shape, inputs=[tick, self.tapes, self.tlen, self.d.ctrl])
            wp.capture_launch(self.graph)
            wp.launch(self.snap, self.n, inputs=[tick + 1, self.t0, self.tlen, self.body, self.site, self.d.xpos,
                                                 self.d.site_xpos, self.z0, self.end_obj, self.end_site])
            if tick % 5 == 0:
                wp.launch(self.track, self.n, inputs=[self.d.nacon, self.d.nefc, self.peak])
        wp.synchronize()
        wall = time.perf_counter() - t
        z0, eo, es = self.z0.numpy(), self.end_obj.numpy(), self.end_site.numpy()
        rise, near = eo[:, 2] - z0, np.linalg.norm(eo - es, axis=1)
        return dict(success=(rise > LIFT - 0.01) & (near < 0.05), rise=rise, near=near, wall=wall,
                    overflow=self.d.overflow.numpy(), peak=self.peak.numpy().tolist())


# --- commands ---------------------------------------------------------------------------------------------------------
SHAPES = ("box", "cylinder", "lying cylinder", "ball", "capsule", "egg")
SCENES = {"orig": None, "simple": SIMPLE}  # plus exploratory levels written by make-scene --level=X: scene_g1_X.xml
for _lv in ("cull", "hull64", "prim"):
    SCENES[_lv] = ASSETS / f"scene_g1_{_lv}.xml"
# Pre-registered go/no-go (2026-10-06, before the cluster runs), for gpu-simple (pyramidal cones) and gpu-simple-elliptic
# (elliptic cones, impratio 10: the one noslip stand-in tried, fixed in advance): GO for closed-loop RL on MJWarp if
#   speed:    MJWarp lift trials/s at its best batch >= 20x one CPU core replaying the same tapes on today's scene
#             (original colliders, noslip 3), on the same node type; and
#   fidelity: agreement of MJWarp (simplified scene) with CPU original (noslip 3) >= 90 % of executed layouts, and no
#             class whose MJWarp lift rate lies outside the CPU original's Wilson 95 % interval.
GO_SPEEDUP, GO_AGREE = 20, 0.90


def load(path):
    return pickle.load(open(path, "rb"))


def wilson(k, n, z=1.96):
    from so101.g1_grasps import wilson as w
    return w(k, n, z)


def device_name():
    import warp as wp
    return wp.get_device().name


def cmd_tapes(args):
    from multiprocessing import Pool
    n, procs = int(args.get("n", 20)), int(args.get("procs", 1))
    heldout = "heldout" in args  # the held-out lift test's layouts (g1_grasps.TEST_SEED 900000+): the Isaac Lab baseline
    seed0 = 900000 if heldout else SEED0
    jobs = [(seed0 + i, s) for s in SHAPES for i in range(n)]
    assert heldout or SEED0 + n <= 900000, "fidelity layouts must stay below the held-out seeds"
    t = time.time()
    with Pool(procs) as pool:
        recs = pool.map(tape, jobs, chunksize=1)
    path = OUT / f"tapes{'-heldout' if heldout else ''}-n{n}.pkl"
    pickle.dump(recs, open(path, "wb"))
    ran = [r for r in recs if "ctrl" in r]
    print(f"{path}: {len(recs)} layouts, {len(ran)} executed, {(time.time() - t) / 60:.1f} min on {procs} processes;"
          f" live executor {np.mean([r['seconds'] for r in ran]):.1f} s per trial (planning + physics, one core)")


def cmd_cpu(args):
    from multiprocessing import Pool
    recs = [r for r in load(args["tapes"]) if "ctrl" in r]
    scene, noslip, procs = args.get("scene", "orig"), int(args.get("noslip", 3)), int(args.get("procs", 1))
    friction, perturb = args.get("friction"), args.get("perturb")
    t = time.time()
    with Pool(procs) as pool:
        rows = pool.map(cpu_job, [(r, SCENES[scene], noslip, friction, perturb) for r in recs], chunksize=1)
    wall = time.time() - t
    path = OUT / f"cpu-{scene}-ns{noslip}{'-' + friction if friction else ''}{'-' + perturb if perturb else ''}.json"
    json.dump(dict(scene=scene, noslip=noslip, procs=procs, wall=wall, rows=rows), open(path, "w"))
    steps, phys = sum(r["steps"] for r in rows), sum(r["physics_s"] for r in rows)
    print(f"{path}: {len(rows)} replays, lifted {sum(r['success'] for r in rows)}; one core {steps / phys:.0f} steps/s,"
          f" {len(rows) / phys:.2f} trials/s; {procs} processes {len(rows) / wall:.2f} trials/s wall;"
          f" peak contacts {max(r['ncon'] for r in rows)}, rows {max(r['nefc'] for r in rows)}")


def cmd_gpu(args):
    """Fidelity column: every executed tape once, one batch per class (worlds = that class's layouts)."""
    recs = [r for r in load(args["tapes"]) if "ctrl" in r]
    scene = args.get("scene", "simple")
    rows, info = [], {}
    for s in SHAPES:
        rs = [r for r in recs if r["obj"]["shape"] == s]
        t = time.time()
        b = Batch(rs, len(rs), SCENES[scene] or g1.SCENE, int(args.get("nconmax", 48)), int(args.get("njmax", 320)),
                  args.get("friction"))
        out = b.run()
        info[s] = dict(worlds=len(rs), ticks=b.T, wall=out["wall"], setup=time.time() - t - out["wall"],
                       peak=out["peak"], overflow=int((out["overflow"] != 0).sum()), batched=b.batched)
        print(s, info[s], flush=True)
        for i, r in enumerate(rs):
            rows.append(dict(key=key(r), success=bool(out["success"][i]), rise=round(float(out["rise"][i]), 4),
                             near=round(float(out["near"][i]), 4), overflow=int(out["overflow"][i])))
    path = OUT / f"gpu-{scene}{'-' + args['friction'] if 'friction' in args else ''}.json"
    json.dump(dict(scene=scene, device=device_name(), info=info, rows=rows), open(path, "w"))
    print(f"{path}: {len(rows)} replays, lifted {sum(r['success'] for r in rows)}")


def cmd_speed(args):
    """Throughput vs batch size: every class's tapes tiled over `worlds` worlds, two timed runs each (the first run
    after set-up is a warm-up). --cpu: the CPU baseline on this node instead (one core, the first 3 tapes per class,
    original noslip 3, original noslip 0, simplified noslip 0)."""
    recs = [r for r in load(args["tapes"]) if "ctrl" in r]
    if "cpu" in args:
        sub = [r for s in SHAPES for r in [x for x in recs if x["obj"]["shape"] == s][:3]]
        res = {}
        for scene, ns in (("orig", 3), ("orig", 0), ("simple", 0)):
            rows = [cpu_job((r, SCENES[scene], ns, None, None)) for r in sub]
            steps, phys = sum(r["steps"] for r in rows), sum(r["physics_s"] for r in rows)
            res[f"{scene}-ns{ns}"] = dict(steps_s=steps / phys, trials_s=len(rows) / phys, steps_per_trial=steps / len(rows))
            print(scene, ns, res[f"{scene}-ns{ns}"], flush=True)
        import platform
        json.dump(dict(node=platform.node(), cpu=res), open(OUT / "speed-cpu.json", "w"))
        return
    scene = args.get("scene", "simple")
    rows = []
    for n in map(int, args.get("worlds", "1,64,512,4096").split(",")):
        for s in args.get("shape", ",".join(SHAPES)).split(","):
            s = s.replace("-", " ")
            rs = [r for r in recs if r["obj"]["shape"] == s]
            b = Batch(rs, n, SCENES[scene] or g1.SCENE, int(args.get("nconmax", 48)), int(args.get("njmax", 320)),
                      args.get("friction"))
            b.run()  # warm-up
            out = b.run()
            steps = n * b.T * TICK_STEPS
            row = dict(scene=scene + ("-" + args["friction"] if "friction" in args else ""), shape=s, worlds=n, ticks=b.T, wall=out["wall"], steps_s=steps / out["wall"],
                       trials_s=n / out["wall"], overflow=int((out["overflow"] != 0).sum()), peak=out["peak"],
                       agree_tiles=float(np.mean(out["success"] == out["success"][np.arange(n) % len(rs)])))
            rows.append(row)
            print(row, flush=True)
            del b
    path = OUT / f"speed-gpu-{scene}{'-' + args['friction'] if 'friction' in args else ''}-{device_name().replace(' ', '_')}.json"
    json.dump(dict(device=device_name(), rows=rows), open(path, "w"))
    print(path)


def table(cols):
    """Fidelity: lift rate per class (Wilson 95 %) per column; agreement with CPU original (noslip 3) and with CPU
    simplified (noslip 0)."""
    ref = cols["cpu-orig-ns3"]
    keys = [k for k in ref if all(k in c for c in cols.values())]
    lines = ["| column | " + " | ".join(SHAPES) + " | all | agree vs CPU orig ns3 | agree vs CPU simple ns0 |",
             "|---" * (len(SHAPES) + 4) + "|"]
    for name, c in cols.items():
        cells = []
        for s in (*SHAPES, None):
            ks = [k for k in keys if s is None or k[0] == s]
            k_ = sum(c[k] for k in ks)
            lo, hi = wilson(k_, len(ks))
            cells.append(f"{k_}/{len(ks)} ({lo:.0%}-{hi:.0%})")
        agree = lambda o: f"{np.mean([c[k] == o[k] for k in keys]):.1%}" if o is not None else "-"
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {agree(ref)} | {agree(cols.get('cpu-simple-ns0'))} |")
    per = ["| agreement with CPU orig ns3, per class | " + " | ".join(SHAPES) + " |", "|---" * (len(SHAPES) + 1) + "|"]
    for name, c in cols.items():
        per.append(f"| {name} | " + " | ".join(
            f"{np.mean([c[k] == ref[k] for k in keys if k[0] == s]):.0%}" for s in SHAPES) + " |")
    return "\n".join(lines + [""] + per), keys


def cmd_report(args):
    """Tables from runs/warp/*.json, the go/no-go check, and the throughput figure."""
    import glob
    recs = load(args.get("tapes", OUT / "tapes-n100.pkl"))
    ran = [r for r in recs if "ctrl" in r]
    cols = {"live (CPU orig ns3, executor)": {key(r): bool(r["live"]["success"]) for r in ran}}
    for f in sorted(glob.glob(str(OUT / "cpu-*.json"))) + sorted(glob.glob(str(OUT / "gpu-*.json"))):
        d = json.load(open(f))
        cols[Path(f).stem] = {tuple(r["key"]): bool(r["success"]) for r in d["rows"] if r["success"] is not None}
    print(f"{len(recs)} layouts (seeds {SEED0}+, {len(recs) // len(SHAPES)} per class), {len(ran)} executed by the "
          f"scripted grasp")
    text, keys = table(cols)
    print(text)
    live, ref = cols["live (CPU orig ns3, executor)"], cols["cpu-orig-ns3"]
    print(f"\ntape replay reproduces the live executor on {np.mean([live[k] == ref[k] for k in keys]):.1%} of layouts")
    for gpu_col, cpu_col in (("gpu-orig", "cpu-orig-ns0"), ("gpu-simple", "cpu-simple-ns0"),
                             ("gpu-simple-elliptic", "cpu-simple-ns0-elliptic"),
                             ("cpu-orig-ns3-jitter", "cpu-orig-ns3"), ("cpu-orig-ns3-f32", "cpu-orig-ns3"),
                             ("cpu-simple-ns0-elliptic-jitter", "cpu-simple-ns0-elliptic")):
        if gpu_col in cols and cpu_col in cols:  # the engine alone (same scene and physics settings), and the floor
            a, b = cols[gpu_col], cols[cpu_col]
            print(f"same physics: {gpu_col} vs {cpu_col}: agreement {np.mean([a[k] == b[k] for k in keys]):.1%}, lifted"
                  f" {sum(a[k] for k in keys)} vs {sum(b[k] for k in keys)}; per class " + ", ".join(
                      f"{s} {np.mean([a[k] == b[k] for k in keys if k[0] == s]):.0%}" for s in SHAPES))
    cpu = json.load(open(OUT / "speed-cpu.json"))
    base = cpu["cpu"]["orig-ns3"]
    print(f"\nCPU one core ({cpu['node']}): " + "; ".join(f"{k} {v['steps_s']:.0f} steps/s {v['trials_s']:.2f} trials/s"
                                                       for k, v in cpu["cpu"].items()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    best = {}  # scene variant -> best lift trials/s over batch sizes and GPUs
    for f in sorted(glob.glob(str(OUT / "speed-gpu-*.json"))):
        d = json.load(open(f))
        rows = d["rows"]
        ns = sorted({r["worlds"] for r in rows})
        tag = f"{d['device']} ({rows[0]['scene']})"
        agg = []
        for n in ns:
            rs = [r for r in rows if r["worlds"] == n]
            wall = sum(r["wall"] for r in rs)
            agg.append((n, sum(r["worlds"] * r["ticks"] * TICK_STEPS for r in rs) / wall, sum(r["worlds"] for r in rs) / wall))
        print(f"\n{tag}: | worlds | steps/s | lift trials/s | x one CPU core (orig ns3, trials) |")
        for n, st, tr in agg:
            print(f"  | {n} | {st:,.0f} | {tr:.1f} | {tr / base['trials_s']:.0f}x |")
            best[rows[0]["scene"]] = max(best.get(rows[0]["scene"], 0), tr)
        axes[0].plot([a[0] for a in agg], [a[1] for a in agg], "o-", label=tag)
        axes[1].plot([a[0] for a in agg], [a[2] for a in agg], "o-", label=tag)
    for k, v in cpu["cpu"].items():
        axes[0].axhline(v["steps_s"], ls="--", lw=1, color="gray")
        axes[0].text(1, v["steps_s"], f"CPU core {k}", fontsize=7, va="bottom")
        axes[1].axhline(v["trials_s"], ls="--", lw=1, color="gray")
        axes[1].text(1, v["trials_s"], f"CPU core {k}", fontsize=7, va="bottom")
    for ax, lab in zip(axes, ("simulated steps / s (all worlds)", "lift trials / s")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("parallel worlds")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3, which="both")
    axes[1].legend(fontsize=7)
    fig.suptitle("G1 + Dex3 lift trial on MuJoCo Warp vs one CPU core (tape replay, 2 ms steps, 15 per tick)")
    fig.tight_layout()
    fig.savefig(OUT / "throughput.png", dpi=130)
    print(OUT / "throughput.png")
    for name in ("gpu-simple", "gpu-simple-elliptic"):  # primary, and the one pre-specified noslip stand-in
        gpu = cols.get(name)
        if not gpu or name[4:] not in best:
            continue
        agree = np.mean([gpu[k] == ref[k] for k in keys])
        outside = []
        for s in SHAPES:
            ks = [k for k in keys if k[0] == s]
            lo, hi = wilson(sum(ref[k] for k in ks), len(ks))
            rate = np.mean([gpu[k] for k in ks])
            if not lo <= rate <= hi:
                outside.append(s)
        speed = best[name[4:]] / base["trials_s"]
        go = speed >= GO_SPEEDUP and agree >= GO_AGREE and not outside
        print(f"\n{name} pre-registered go/no-go: speed-up {speed:.0f}x (need {GO_SPEEDUP}x), agreement {agree:.1%} (need"
              f" {GO_AGREE:.0%}), classes outside the CPU interval {outside or 'none'} -> {'GO' if go else 'NO-GO'}")


# --- export for Isaac Lab (isaac/lift_replay.py): the tapes plus everything the Isaac scene must match ----------------
GEOM = {int(mujoco.mjtGeom.mjGEOM_BOX): "box", int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
        int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere", int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
        int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid"}  # keyed by plain int: m.geom_type holds numpy ints


def cmd_isaac(args):
    """--tapes=FILE -> runs/isaac/<stem>.npz (ctrl tapes, object arrays) + .json (robot, actuators, table, physics, objects).
    MuJoCo world frame throughout (table top z = 0, pelvis welded at g1.PELVIS). Per actuator: its joint, kp, kv, force
    limit, so Isaac's implicit PD can match. Objects: the primitive's MuJoCo geom (type, size, pose), mass, friction."""
    recs = [r for r in load(args["tapes"]) if "ctrl" in r]
    m0 = model_for(recs[0]["obj"]["seed"], recs[0]["obj"]["shape"])
    acts = []
    for a in range(m0.nu):
        j = m0.joint(m0.actuator_trnid[a, 0])
        fr = m0.actuator_forcerange[a] if m0.actuator_forcelimited[a] else (
            m0.jnt_actfrcrange[j.id] if m0.jnt_actfrclimited[j.id] else [-1e6, 1e6])
        acts.append(dict(joint=j.name, kp=float(m0.actuator_gainprm[a, 0]), kv=float(-m0.actuator_biasprm[a, 2]),
                         force=float(max(abs(fr[0]), abs(fr[1]))), lo=float(m0.actuator_ctrlrange[a, 0]),
                         hi=float(m0.actuator_ctrlrange[a, 1])))
    hinge = [m0.joint(i) for i in range(m0.njnt) if m0.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    pelvis = m0.body("pelvis")
    geoms = {n: dict(pos=m0.geom(n).pos.tolist(), half=m0.geom(n).size.tolist()) for n in ("floor", "shelf")}
    objs, ctrl = [], []
    for r in recs:
        m = model_for(r["obj"]["seed"], r["obj"]["shape"])
        b = m.body(r["body"])
        g = [i for i in range(m.ngeom) if m.geom_bodyid[i] == b.id][0]
        q = r["qpos"]
        adr = m.jnt_qposadr[b.jntadr[0]]
        objs.append(dict(shape=r["obj"]["shape"], seed=r["obj"]["seed"], geom=GEOM[int(m.geom_type[g])],
                         size=m.geom_size[g].tolist(), geom_pos=m.geom_pos[g].tolist(), geom_quat=m.geom_quat[g].tolist(),
                         pos=q[adr:adr + 3].tolist(), quat=q[adr + 3:adr + 7].tolist(), mass=float(m.body_mass[b.id]),
                         friction=float(m.geom_friction[g][0]), t0=int(r["t0"]), ticks=len(r["ctrl"]),
                         robot_q={j.name: float(q[m.jnt_qposadr[j.id]]) for j in hinge},
                         live=dict(success=bool(r["live"].get("success")), rise=r["live"].get("rise"))))
        ctrl.append(r["ctrl"].astype(np.float32))
    T = max(len(c) for c in ctrl)
    tapes = np.stack([np.vstack([c, np.repeat(c[-1:], T - len(c), 0)]) for c in ctrl])
    out = Path("runs/isaac")
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args["tapes"]).stem
    np.savez_compressed(out / f"{stem}.npz", tapes=tapes, ticks=np.array([len(c) for c in ctrl]))
    meta = dict(timestep=float(m0.opt.timestep), tick_steps=TICK_STEPS, gravity=m0.opt.gravity.tolist(),
                pelvis_pos=pelvis.pos.tolist(), pelvis_quat=pelvis.quat.tolist(), gravcomp=True,
                actuators=acts, table=geoms["floor"], shelf=geoms["shelf"], pinch=dict(body=g1.HAND_BODY, pos=g1.PINCH.tolist()),
                lift=LIFT, objects=objs)
    (out / f"{stem}.json").write_text(json.dumps(meta, indent=1))
    print(f"{out / stem}.npz/.json: {len(recs)} tapes x {T} ticks, {m0.nu} actuators, live lifted "
          f"{sum(o['live']['success'] for o in objs)}")


def main():
    args = dict((a[2:].split("=", 1) + [""])[:2] for a in sys.argv[2:] if a.startswith("--"))
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "make-scene":
        lv = args.get("level", "hull")
        return make_scene(lv, SIMPLE if lv == "hull" else ASSETS / f"scene_g1_{lv}.xml")
    {"tapes": cmd_tapes, "cpu": cmd_cpu, "gpu": cmd_gpu, "speed": cmd_speed, "report": cmd_report,
     "isaac": cmd_isaac}[cmd](args)


if __name__ == "__main__":
    main()
