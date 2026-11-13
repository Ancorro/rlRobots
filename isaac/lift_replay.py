"""Milestone 1 of the Isaac Lab move (LOG.md 2026-10-07): the held-out G1 lift test in Isaac Lab, as a baseline every later
RL result compares against. The scripted grasp is not re-planned here: its MuJoCo run on each held-out layout recorded
every 30 ms tick's joint targets (a "tape", so101/g1_warp.py tapes --heldout, exported by `g1_warp isaac`), and Isaac
replays the same tapes from the same start state, one layout per environment, all in parallel. Same success rule as
g1_grasps.run: the object 9 cm above where it was when the grasp started, within 5 cm of the pinch point.

Matched to the MuJoCo scene: the MuJoCo world frame (table top z = 0, pelvis fixed at g1.PELVIS), Unitree's fixed-base
G1-29dof + Dex3 USD (unitree_sim_isaaclab; same joint names as the Menagerie model), every actuator's joint, kp, kv and
force limit as implicit PD drives, gravity off on the robot (MuJoCo's gravcomp), 2 ms physics steps, 15 per tick, the
object's primitive, pose, mass and friction. Not matched: contact model and solver (PhysX vs MuJoCo + noslip), and the
collision meshes of Unitree's USD vs Menagerie's: that difference is what this measures.

  python isaac/lift_replay.py --data=runs/isaac/tapes-heldout-n60 [--shapes=box,cylinder] [--limit=N]
  -> runs/isaac/<stem>-isaac.json (per layout: shape, seed, Isaac success / rise / near, MuJoCo live success)
"""
import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--data", default="runs/isaac/tapes-heldout-n60")
p.add_argument("--shapes", default="")
p.add_argument("--limit", type=int, default=0)
p.add_argument("--unitree", default=os.path.expandvars("/nfs/hpc/share/$USER/isaac/unitree"))
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_mul  # noqa: E402

os.environ["PROJECT_ROOT"] = args.unitree
sys.path.insert(0, args.unitree)
from robots.unitree import G129_CFG_WITH_DEX3_BASE_FIX  # noqa: E402

meta = json.loads(Path(args.data + ".json").read_text())
npz = np.load(args.data + ".npz")
keep = [i for i, o in enumerate(meta["objects"]) if not args.shapes or o["shape"] in args.shapes.split(",")]
if args.limit:
    keep = keep[:args.limit]
objs = [meta["objects"][i] for i in keep]
tapes, ticks = npz["tapes"][keep], npz["ticks"][keep]
N, T = len(objs), int(ticks.max())
acts = meta["actuators"]
dev = "cuda:0"

# --- robot: Unitree's fixed-base G1 + Dex3, MuJoCo's drives and pose ------------------------------------------------
robot_cfg = G129_CFG_WITH_DEX3_BASE_FIX.replace(prim_path="/World/envs/env_.*/Robot")
robot_cfg.spawn.rigid_props.disable_gravity = True  # MuJoCo's gravcomp on every body
robot_cfg.init_state = robot_cfg.init_state.replace(pos=tuple(meta["pelvis_pos"]), rot=tuple(meta["pelvis_quat"]),
                                                    joint_pos=dict(objs[0]["robot_q"]))
names = [a["joint"] for a in acts]
robot_cfg.actuators = {
    "tape": ImplicitActuatorCfg(joint_names_expr=names, stiffness={a["joint"]: a["kp"] for a in acts},
                                damping={a["joint"]: a["kv"] for a in acts},
                                effort_limit_sim={a["joint"]: a["force"] for a in acts}),
}


class SceneCfg(InteractiveSceneCfg):
    pass


scene_cfg = SceneCfg(num_envs=N, env_spacing=3.0, replicate_physics=False)
scene_cfg.robot = robot_cfg
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=meta["timestep"], device=dev,
                                                          gravity=tuple(meta["gravity"])))
scene = InteractiveScene(scene_cfg)
origins = scene.env_origins.cpu().numpy()

# --- table and shelf (static colliders), one object per environment -------------------------------------------------
for i in range(N):
    for name in ("table", "shelf"):
        g = meta[name]
        box = sim_utils.CuboidCfg(size=tuple(2 * np.array(g["half"])), collision_props=sim_utils.CollisionPropertiesCfg(),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.55, 0.5)))
        box.func(f"/World/envs/env_{i}/{name}", box, translation=tuple((origins[i] + g["pos"]).tolist()))


def wxyz(q):
    return torch.tensor(q, dtype=torch.float32)


for i, o in enumerate(objs):
    common = dict(rigid_props=sim_utils.RigidBodyPropertiesCfg(), mass_props=sim_utils.MassPropertiesCfg(mass=o["mass"]),
                  collision_props=sim_utils.CollisionPropertiesCfg(),
                  physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=o["friction"],
                                                                  dynamic_friction=o["friction"], restitution=0.0),
                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.3, 0.2)))
    s = o["size"]
    if o["geom"] == "box":
        cfg = sim_utils.CuboidCfg(size=(2 * s[0], 2 * s[1], 2 * s[2]), **common)
    elif o["geom"] == "cylinder":
        cfg = sim_utils.CylinderCfg(radius=s[0], height=2 * s[1], axis="Z", **common)
    elif o["geom"] == "capsule":
        cfg = sim_utils.CapsuleCfg(radius=s[0], height=2 * s[1], axis="Z", **common)
    elif o["geom"] == "sphere":
        cfg = sim_utils.SphereCfg(radius=s[0], **common)
    else:  # ellipsoid (egg): a unit mesh sphere, scaled per axis after spawning (convex hull collider)
        cfg = sim_utils.MeshSphereCfg(radius=1.0, **common)
    body_q, geom_q = wxyz(o["quat"]), wxyz(o["geom_quat"])
    pos = torch.tensor(o["pos"]) + quat_apply(body_q[None], torch.tensor(o["geom_pos"])[None])[0]
    rot = quat_mul(body_q[None], geom_q[None])[0]
    prim = cfg.func(f"/World/envs/env_{i}/Object", cfg, translation=tuple((origins[i] + pos.numpy()).tolist()),
                    orientation=tuple(rot.tolist()))
    if o["geom"] == "ellipsoid":  # the spawner already wrote translate/orient/scale ops: set its scale op, not a new one
        from pxr import Gf
        a = prim.GetAttribute("xformOp:scale")
        assert a.IsValid(), "ellipsoid: spawned prim has no xformOp:scale"
        a.Set(Gf.Vec3d(*s) if "double" in str(a.GetTypeName()) else Gf.Vec3f(*s))

obj = RigidObject(RigidObjectCfg(prim_path="/World/envs/env_.*/Object", spawn=None))
scene.rigid_objects["object"] = obj
robot: Articulation = scene["robot"]
sim.reset()

# --- start state: each layout's recorded joint positions --------------------------------------------------------------
jnames = robot.joint_names
q0 = robot.data.default_joint_pos.clone()
for i, o in enumerate(objs):
    for n, v in o["robot_q"].items():
        if n in jnames:
            q0[i, jnames.index(n)] = v
robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
ids = [jnames.index(n) for n in names]
missing = [n for n in names if n not in jnames]
assert not missing, f"tape joints missing in the USD: {missing}"
hand = robot.body_names.index(meta["pinch"]["body"])
pinch_local = torch.tensor(meta["pinch"]["pos"], device=dev)
tape = torch.tensor(tapes, device=dev)  # N x T x actuators
t0 = torch.tensor([o["t0"] for o in objs], device=dev)
end = torch.tensor(ticks, device=dev) - 1
z0 = torch.zeros(N, device=dev)
res = torch.zeros(N, 3, device=dev)  # rise, near, done
scene.update(meta["timestep"])
for k in range(T):
    robot.set_joint_position_target(tape[:, min(k, T - 1)], joint_ids=ids)
    for _ in range(meta["tick_steps"]):
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(meta["timestep"])
    oz = obj.data.root_pos_w[:, 2]
    z0 = torch.where(t0 == k, oz, z0)
    pinch = robot.data.body_pos_w[:, hand] + quat_apply(robot.data.body_quat_w[:, hand], pinch_local.expand(N, 3))
    near = torch.linalg.norm(obj.data.root_pos_w - pinch, dim=1)
    hit = end == k
    res[hit] = torch.stack([oz - z0, near, torch.ones_like(near)], 1)[hit]
rows = []
for i, o in enumerate(objs):
    rise, near = float(res[i, 0]), float(res[i, 1])
    rows.append(dict(shape=o["shape"], seed=o["seed"], success=bool(rise > meta["lift"] - 0.01 and near < 0.05),
                     rise=round(rise, 4), near=round(near, 4), mujoco=o["live"]["success"]))
out = Path(args.data + "-isaac.json")
out.write_text(json.dumps(dict(rows=rows), indent=1))
print(f"{out}: {N} layouts")
for s in dict.fromkeys(o["shape"] for o in objs):
    r = [x for x in rows if x["shape"] == s]
    agree = sum(x["success"] == x["mujoco"] for x in r)
    print(f"  {s:15s} Isaac {sum(x['success'] for x in r):3d}  MuJoCo {sum(x['mujoco'] for x in r):3d}  of {len(r)};"
          f" agree {agree}/{len(r)}")
app.close()
