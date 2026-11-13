"""Videos of Isaac runs, drawn by MuJoCo. Isaac's RTX renderer crashes on our AWS boxes (driver 595 with Isaac Sim 5.1,
2026-10-07), so the Isaac scripts record the motion (pelvis pose, every joint, the object's pose, per control step) and
this redraws it with MuJoCo Menagerie's G1 + Dex3 (same joint names). Physics is Isaac's; MuJoCo only draws.

  python isaac/render_mj.py walk runs/isaac/walk-vx0.5.npz OUT_DIR      -> OUT_DIR/walk-vx0.5.mp4
  python isaac/render_mj.py lift runs/isaac/g1-lift-n10.npz OUT_DIR     -> OUT_DIR/lift-<shape>-<lifted|held|missed>.mp4 (one each)
  python isaac/render_mj.py walkgrasp runs/isaac/walk-grasp-n12-box.npz OUT_DIR  -> OUT_DIR/walkgrasp-box-<lifted|missed>.mp4
    (walk_grasp.py --carry runs: also table B; outcomes placed / not-placed / dropped / missed, one robot each)
    (--objects gso|pool recordings: each object is drawn from its convex parts; HULLS=<hulls.npz> when the recording's path is another machine's)
  python isaac/render_mj.py boxstand sdg/boxbcplay-x.npz OUT_DIR  -> OUT_DIR/boxstand-<n>-<success|fail>.mp4
    (isaac/box_bcplay.py --record: the standing box task; 3 successes and 1 failure, or PICK=0,4,7,19 rollouts)
  (MUJOCO_GL=egl on a headless GPU box; needs so101/assets/fetch_g1.py run first) -> also OUT_DIR/clips.json
"""
import json
import sys
from pathlib import Path

import av
import mujoco
import numpy as np

kind, src, out_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
out_dir.mkdir(parents=True, exist_ok=True)
W, H, FPS = 640, 480, 25
d = np.load(src, allow_pickle=False)
names = [str(n) for n in d["joint_names"]]
dt = float(d["dt"])


def base_spec(floor_z=0.0):
    spec = mujoco.MjSpec.from_file(str(Path(__file__).resolve().parent.parent / "so101/assets/unitree_g1/g1_with_hands.xml"))
    spec.visual.global_.offwidth, spec.visual.global_.offheight = W, H
    spec.visual.headlight.ambient, spec.visual.headlight.diffuse = [0.45, 0.45, 0.45], [0.55, 0.55, 0.55]
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.86, 0.9, 0.92], rgb2=[0.55, 0.62, 0.68], width=256, height=256)
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.82, 0.85, 0.84], rgb2=[0.72, 0.76, 0.75], width=256, height=256)
    spec.add_material(name="grid", textures=["", "grid"], texrepeat=[8, 8], reflectance=0.05)
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, pos=[0, 0, floor_z], size=[0, 0, 0.05],
                            material="grid")
    spec.worldbody.add_light(pos=[0, 0, 4], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8], castshadow=True)
    return spec


def write(path, frames_fn, n, step):
    box = av.open(str(path), "w")
    st = box.add_stream("libx264", rate=FPS)
    st.width, st.height, st.pix_fmt = W, H, "yuv420p"
    for k in range(0, n, step):
        for p in st.encode(av.VideoFrame.from_ndarray(frames_fn(k), format="rgb24")):
            box.mux(p)
    for p in st.encode():
        box.mux(p)
    box.close()


def joint_map(m):
    return [(m.jnt_qposadr[m.joint(n).id], i) for i, n in enumerate(names) if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) >= 0]


clips = []
if kind == "walk":
    m = base_spec().compile()
    data, jm = mujoco.MjData(m), None
    jm = joint_map(m)
    f = d["frames"]  # per step: root pos (3), root quat wxyz (4), joints
    r = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 2.6, 120, -12

    def frame(k):
        data.qpos[:7] = f[k, :7]
        for a, i in jm:
            data.qpos[a] = f[k, 7 + i]
        mujoco.mj_kinematics(m, data)
        cam.lookat[:] = [f[k, 0], f[k, 1], 0.7]
        r.update_scene(data, cam)
        return r.render()
    name = src.stem + ".mp4"
    write(out_dir / name, frame, len(f), max(1, round(1 / (FPS * dt))))
    clips.append(dict(file=name, label=str(d["label"]), seconds=round(len(f) * dt, 1)))
elif kind == "boxstand":  # isaac/box_bcplay.py --record: one array per rollout, pelvis pos, quat, joints, box pos, quat
    rows = json.loads(src.with_suffix(".json").read_text())["rows"]
    J, sx, sy, sz = len(names), *d["size"]
    below, goal = float(d["below"]), d["goal"]
    pick = __import__("os").environ.get("PICK", "")
    if pick:
        ids = [int(v) for v in pick.split(",")]
    else:
        ok = [r["trial"] for r in rows if r["success"]]
        bad = [r["trial"] for r in rows if not r["success"]]
        ids = ok[:3] + bad[:1]
    top = 0.6996 - below  # the box starts on the packing table, its frame 'below' above its bottom
    for i in ids:
        f = d[f"r{i}"]
        spec = base_spec()
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0.55, top - 0.02], size=[0.6, 0.3, 0.02],
                                rgba=[0.55, 0.5, 0.45, 1])
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0.6, (top - 0.04) / 2], size=[0.4, 0.2, (top - 0.04) / 2],
                                rgba=[0.45, 0.4, 0.36, 1])
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[goal[0], goal[1], top + 0.001], size=[0.08, 0.08, 0.001],
                                rgba=[0.3, 0.7, 0.4, 0.6], contype=0, conaffinity=0)  # the success region, 8 cm each way
        b = spec.worldbody.add_body(name="obj", mocap=True)
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx / 2, sy / 2, sz / 2], pos=[0, 0, sz / 2 - below],
                   rgba=[0.62, 0.45, 0.28, 1], contype=0, conaffinity=0)
        m = spec.compile()
        data = mujoco.MjData(m)
        jm = joint_map(m)
        r = mujoco.Renderer(m, H, W)
        cam = mujoco.MjvCamera()
        cam.distance, cam.azimuth, cam.elevation = 1.6, 25, -28  # in front of the robot (it faces +y), a little to its right
        cam.lookat[:] = [0.0, 0.35, 0.85]

        def frame(k, f=f):
            x = f[k]
            data.qpos[:7] = x[:7]
            for a, j in jm:
                data.qpos[a] = x[7 + j]
            data.mocap_pos[0], data.mocap_quat[0] = x[7 + J:10 + J], x[10 + J:14 + J]
            mujoco.mj_kinematics(m, data)
            r.update_scene(data, cam)
            return r.render()
        outcome = "success" if rows[i]["success"] else "fail"
        name = f"boxstand-{i}-{outcome}.mp4"
        write(out_dir / name, frame, len(f), max(1, round(1 / (FPS * dt))))
        clips.append(dict(file=name, trial=i, outcome=outcome, steps=rows[i]["steps"], handoff=rows[i].get("handoff"),
                          seconds=round(len(f) * dt, 1)))
elif kind == "boxwalk":  # isaac/box_walk.py --record: same frame layout as boxstand; tables from the json rows
    rows = json.loads(src.with_suffix(".json").read_text())["rows"]
    J, sx, sy, sz = len(names), *d["size"]
    below = float(d["below"])
    pick = __import__("os").environ.get("PICK", "")
    if pick:
        ids = [int(v) for v in pick.split(",")]
    else:
        ok = [r["trial"] for r in rows if r["success"]]
        bad = [r["trial"] for r in rows if not r["success"]]
        ids = ok[:3] + bad[:4 - len(ok[:3])]  # 4 clips: up to 3 successes, the rest failures
    top = 0.6996 - below
    for i in ids:
        f = d[f"r{i}"]
        if len(f) == 0:
            continue
        spec = base_spec()
        for key, rgba in (("start_table", [0.55, 0.5, 0.45, 1]), ("end_table", [0.45, 0.55, 0.5, 1])):
            tp = rows[i].get(key)
            if tp is None:
                continue
            yaw = 2 * np.arctan2(tp[6], tp[3])  # tables turn about z only
            spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[tp[0], tp[1], top / 2], size=[0.6, 0.3, top / 2],
                                    quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)], rgba=rgba)
        b = spec.worldbody.add_body(name="obj", mocap=True)
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx / 2, sy / 2, sz / 2], pos=[0, 0, sz / 2 - below],
                   rgba=[0.62, 0.45, 0.28, 1], contype=0, conaffinity=0)
        m = spec.compile()
        data = mujoco.MjData(m)
        jm = joint_map(m)
        r = mujoco.Renderer(m, H, W)
        cam = mujoco.MjvCamera()
        cam.distance, cam.azimuth, cam.elevation = 2.4, 135, -25

        def frame(k, f=f):
            x = f[k]
            data.qpos[:7] = x[:7]
            for a, j in jm:
                data.qpos[a] = x[7 + j]
            data.mocap_pos[0], data.mocap_quat[0] = x[7 + J:10 + J], x[10 + J:14 + J]
            mujoco.mj_kinematics(m, data)
            cam.lookat[:] = [x[0], x[1], 0.8]  # follows the robot
            r.update_scene(data, cam)
            return r.render()
        outcome = "success" if rows[i]["success"] else ("dropped" if rows[i]["dropped"] else "fell" if rows[i]["fell"] else "fail")
        name = f"{src.stem}-{i}-{outcome}.mp4"
        write(out_dir / name, frame, len(f), max(1, round(1 / (FPS * dt))))
        clips.append(dict(file=name, trial=i, outcome=outcome, steps=rows[i]["steps"], handoff=rows[i].get("handoff"),
                          last_phase=rows[i].get("last_phase"), seconds=round(len(f) * dt, 1)))
elif kind == "walkgrasp":  # isaac/walk_grasp.py --record: per step and env, root pos, quat, joints, object pos, quat
    f, specs = d["frames"], json.loads(str(d["specs"]))
    side = json.loads(src.with_suffix(".json").read_text())
    rows, run_args = side["rows"], side.get("args", {})
    _, ty, top = d["table"]  # each env its own object (walk_grasp.py since 2026-10-22; before, all were env_0's)
    # --rand_layout runs: each env has its own tables (walk_grasp.py pose_fixture); without this the clips drew every
    # robot at the fixed layout's tables
    LAY = json.loads(str(d["layout"])) if "layout" in d.files and run_args.get("rand_layout") else None
    PICK = [int(v) for v in __import__("os").environ.get("PICK", "").split(",") if v]  # these robots, whatever their outcome
    J = len(names)
    HULL_NPZ = {}  # hulls.npz files of scanned objects, by path (HULLS=<file> overrides the path the recording carries)
    CLOSE = bool(int(__import__("os").environ.get("CLOSE", "0")))
    tb = d["table_b"] if "table_b" in d.files and not np.isnan(d["table_b"]).any() else None
    picks = {}
    BYSHAPE = bool(int(__import__("os").environ.get("BYSHAPE", "0")))  # one placed clip per shape (mixed-shape runs)
    for i, row in enumerate(rows):  # first robot per outcome
        if PICK:
            if i in PICK:
                picks[f"{row['shape'].replace(' ', '-')}-{i}-{'placed' if row['success'] else 'not-placed' if row.get('carried') else 'missed'}"] = i
            continue
        if BYSHAPE and tb is not None:
            if row["success"] and row["placed"] and row["upright"] and row["standing"]:
                picks.setdefault(f"{row['shape'].replace(' ', '-')}-placed", i)
            continue
        if tb is None:
            picks.setdefault("lifted" if row["success"] else "missed", i)
        else:
            full = row["success"] and row["placed"] and row["upright"] and row["standing"]
            picks.setdefault("placed" if full else "not-placed" if row["carried"] else "dropped" if row["success"] else "missed", i)
    for outcome, i in sorted(picks.items()):
        spec = base_spec()
        if LAY is not None:  # this env's own tables: A facing the robot at (ax, edge), B across the walk after the turn
            L = LAY[i]
            ax_, edge_ = L["ax"], L["edge"]
            byaw = np.pi / 2 + L["b_turn"]
            bd = np.array([np.cos(byaw), np.sin(byaw)])
            b_edge = np.array([ax_, edge_ - run_args.get("stand", 0.18) - run_args.get("back", 0.45)]) + L["b_dist"] * bd
            bq = [np.cos(byaw / 2), 0, 0, np.sin(byaw / 2)]
            for pos, size, rgba, quat in (
                    ([ax_, edge_ + 0.3, top - 0.02], [0.5, 0.3, 0.02], [0.55, 0.5, 0.45, 1], [1, 0, 0, 0]),
                    ([ax_, edge_ + 0.425, (top - 0.04) / 2], [0.3, 0.175, (top - 0.04) / 2], [0.45, 0.4, 0.36, 1], [1, 0, 0, 0]),
                    ([*(b_edge + 0.3 * bd), top - 0.02], [0.3, 0.8, 0.02], [0.55, 0.5, 0.45, 1], bq),
                    ([*(b_edge + 0.425 * bd), (top - 0.04) / 2], [0.175, 0.3, (top - 0.04) / 2], [0.45, 0.4, 0.36, 1], bq)):
                spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size, rgba=rgba, quat=quat)
        else:
            spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, ty, top - 0.02], size=[0.5, 0.3, 0.02],
                                    rgba=[0.55, 0.5, 0.45, 1])
            spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, ty + 0.125, (top - 0.04) / 2],
                                    size=[0.3, 0.175, (top - 0.04) / 2], rgba=[0.45, 0.4, 0.36, 1])
        if tb is not None and LAY is None:  # table B (walk_grasp.py --carry: 0.6 m deep, 1.6 m wide, pedestal set back)
            bx, by, _ = tb
            spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[bx, by, top - 0.02], size=[0.3, 0.8, 0.02],
                                    rgba=[0.55, 0.5, 0.45, 1])
            spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[bx + 0.125, by, (top - 0.04) / 2],
                                    size=[0.175, 0.3, (top - 0.04) / 2], rgba=[0.45, 0.4, 0.36, 1])
        b = spec.worldbody.add_body(name="obj", mocap=True)
        s0 = specs[i] if len(specs) > i else specs[0]
        if s0["type"] == "hull":  # a scanned or pooled object: its convex parts (hulls.npz), in the object's own frame as Isaac spawned them
            hf = Path(__import__("os").environ.get("HULLS") or s0["hull_file"])
            HULL_NPZ[str(hf)] = HULL_NPZ.get(str(hf)) or np.load(hf)
            hz = HULL_NPZ[str(hf)]
            V, F, fs = hz[f"{s0['name']}/v"], hz[f"{s0['name']}/f"], hz[f"{s0['name']}/fstart"]
            for k in range(len(fs) - 1):
                ids, f2 = np.unique(F[fs[k]:fs[k + 1]], return_inverse=True)
                spec.add_mesh(name=f"part{k}", uservert=np.ascontiguousarray(V[ids], dtype=np.float64).reshape(-1), userface=f2.reshape(-1).astype(np.int32))
                b.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"part{k}", rgba=s0.get("rgba", [0.8, 0.6, 0.3, 1]), contype=0, conaffinity=0)
        else:
            gt = {"box": mujoco.mjtGeom.mjGEOM_BOX, "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
                  "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE, "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
                  "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID}[s0["type"]]
            # lying shapes are built along their axis in Isaac (MuJoCo cylinders and capsules run along z)
            ax = s0.get("quat_axis") or {"lying cylinder": "y", "capsule": "y"}.get(s0.get("shape"))
            if s0["type"] == "ellipsoid":  # walk_grasp spawns eggs as capsules along x
                gt, s0 = mujoco.mjtGeom.mjGEOM_CAPSULE, dict(s0, size=[(s0["size"][1] + s0["size"][2]) / 2,
                                                                       max(s0["size"][0] - (s0["size"][1] + s0["size"][2]) / 2, 0.001), 0])
                ax = "x"
            gq = {"y": [0.7071, 0.7071, 0, 0], "x": [0.7071, 0, 0.7071, 0]}.get(ax, [1, 0, 0, 0])
            b.add_geom(type=gt, size=(list(s0["size"]) + [0, 0])[:3], quat=gq, rgba=s0.get("rgba", [0.6, 0.45, 0.3, 1]), contype=0, conaffinity=0)
        m = spec.compile()
        data = mujoco.MjData(m)
        jm = joint_map(m)
        r = mujoco.Renderer(m, H, W)
        cam = mujoco.MjvCamera()
        cam.distance, cam.azimuth, cam.elevation = 2.0, 215, -18  # front right of the robot, which walks along +y
        if tb is not None:
            cam.distance, cam.azimuth, cam.elevation = 3.4, 40, -30  # fixed, from behind-left of the start: both tables in view

        def frame(k, i=i):
            x = f[k, i]
            data.qpos[:7] = x[:7]
            for a, j in jm:
                data.qpos[a] = x[7 + j]
            data.mocap_pos[0], data.mocap_quat[0] = x[7 + J:10 + J], x[10 + J:14 + J]
            mujoco.mj_kinematics(m, data)
            cam.lookat[:] = [x[0], x[1] + 0.3, 0.8] if tb is None else [0.55, 0.75, 0.6]
            if CLOSE:  # follow the robot from its right-front, 1.5 m away
                w, qx, qy, qz = x[3:7]
                yaw = np.degrees(np.arctan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
                cam.lookat[:] = [x[0], x[1], 0.85]
                cam.distance, cam.azimuth, cam.elevation = 1.5, yaw + 180 + 50, -15
            r.update_scene(data, cam)
            return r.render()
        name = (f"{src.stem}-{outcome}.mp4" if BYSHAPE or PICK else
                f"{'close-' if CLOSE else ''}walk{'grasp' if tb is None else 'carry'}-{rows[i]['shape'].replace(' ', '-')}-{outcome}.mp4")
        write(out_dir / name, frame, len(f), max(1, round(1 / (FPS * dt))))
        r.close()  # one GL context at a time: on Windows a second live Renderer drew black frames (2026-11-11)
        clips.append(dict(file=name, shape=rows[i]["shape"], seed=rows[i].get("seed"), outcome=outcome, robot=i,
                          **({"name": rows[i]["name"]} if rows[i].get("name") else {}), moved_in=rows[i].get("moved_in")))
        print(name, flush=True)
else:
    specs = json.loads(str(d["specs"]))
    q, obj = d["q"].astype(np.float32), d["obj"]
    picks = {}
    for i, s in enumerate(specs):  # first layout per shape and outcome: lifted (9 cm), held (in the hand, 5 cm up), missed
        picks.setdefault((s["shape"], "lifted" if s["success"] else "held" if s.get("held") else "missed"), i)
    for (shape, outcome), i in sorted(picks.items()):
        s = specs[i]
        spec = base_spec(floor_z=float(d["pelvis"][2]) - 0.79)  # the lift scene's table top is z = 0; feet below
        tp, ts = d["table_pos"], d["table_size"]
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=tp, size=ts / 2, rgba=[0.55, 0.5, 0.45, 1])
        b = spec.worldbody.add_body(name="obj", mocap=True)
        gt = {"box": mujoco.mjtGeom.mjGEOM_BOX, "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
              "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE, "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
              "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID}[s["type"]]
        size = (s["size"] + [0, 0])[:3]
        b.add_geom(type=gt, size=size, rgba=s["rgba"], contype=0, conaffinity=0)
        m = spec.compile()
        data = mujoco.MjData(m)
        jm = joint_map(m)
        r = mujoco.Renderer(m, H, W)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [obj[0, i, 0], obj[0, i, 1], 0.07]
        cam.distance, cam.azimuth, cam.elevation = 0.5, 150, -20  # from the table's far side, facing the robot

        def frame(k, i=i):
            data.qpos[:3], data.qpos[3:7] = d["pelvis"], [1, 0, 0, 0]
            for a, j in jm:
                data.qpos[a] = q[k, i, j]
            data.mocap_pos[0], data.mocap_quat[0] = obj[k, i, :3], obj[k, i, 3:]
            mujoco.mj_kinematics(m, data)
            r.update_scene(data, cam)
            return r.render()
        name = f"lift-{shape.replace(' ', '-')}-{outcome}.mp4"
        write(out_dir / name, frame, len(q), max(1, round(1 / (FPS * dt))))
        clips.append(dict(file=name, shape=shape, seed=s["seed"], outcome=outcome))
        print(name, flush=True)
cj = out_dir / "clips.json"
old = json.loads(cj.read_text()) if cj.exists() else []
keep = [c for c in old if c["file"] not in {c["file"] for c in clips}]
cj.write_text(json.dumps(keep + clips, indent=1))
print(f"{len(clips)} clips -> {out_dir}")
