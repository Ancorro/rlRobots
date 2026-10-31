"""Learned grasp adjustments for the G1 + Dex3 lift in Isaac (owner 2026-10-07: "the training for the skills"). The
scripted top-down pinch of isaac/g1_lift.py stays (same scene, same honest physics: self-collision on, camera-mount
collider off, MuJoCo hand drives, torsional friction, "max" combine, 32 solver iterations); a small policy reads the
object (shape, size, pose) and picks five adjustments to it: pinch offset along and across the closing direction,
height, heading, and how far to close the fingers. The same idea as the MuJoCo grasp chooser (scripted 154 -> v4 222 of
360), here trained by policy gradient (REINFORCE, per-shape baseline) on many parallel environments.

Train on seeds 800000+ (never the held-out 900000+); the policy starts at zero adjustment, i.e. exactly the scripted
pinch. Evaluate the policy's mean on the held-out lift test: 6 shapes x 60 layouts, success = object 9 cm up and within
5 cm of the pinch point; baseline 155/360 lifted, 187 held (LOG 2026-10-07).

  python isaac/g1_grasp_rl.py train [--n=60] [--reps=8] [--iters=60] [--out=runs/isaac/rl] [--redraw] [--params=8] [--snap=30]
  python isaac/g1_grasp_rl.py eval --policy=runs/isaac/rl/policy.pt[,more.pt] [--n=60]  (--zero: the scripted pinch, as a check)

--redraw: new object poses (position, heading) every round, shared by a layout's copies (more distinct layouts than the
scene holds). --params=8 adds hand tilt along/across the closing direction and finger closing speed to the five.

Realism (2026-10-08 audit; --legacy restores the v1-v7 settings, and each flag overrides it):
  --noise   perception error as so101/g1_grasps.perceived: centre off by 0.4 cm per axis (std), sizes x 1 +- 5 %; it
            reaches the policy's inputs, the aim and every re-aim (v1-v7: exact object state)
  --waist   real G1 motor torques, waist yaw 88 / roll, pitch 50, arm 25, wrist pitch/yaw 5 N m (Menagerie g1)
            (v1-v7: a 1000 N m waist, arms as Unitree's USD)
  --friction object friction range (v1-v7: 0.7-1.5, the grippier surface wins)
  --egg     egg collider = its convex hull (v1-v7: Isaac Lab gives sphere meshes a bounding sphere, so every egg was a
            ball as wide as the egg is long)
  --ccd     objects: a 1 m/s cap on how fast PhysX pushes overlapping bodies apart, and continuous collision detection
            (which PhysX ignores on the GPU; v1-v7: no cap)
--objects=household: the household benchmark instead (eval only, seeds 950000+): the 114 scanned household objects of
so101/assets/objects/gso_g1.py, --n poses each. Each settles for 1 s; perception then fits the settled object as the
policy's inputs expect: its narrowest horizontal width sets the closing direction, the pinch is at mid-height but at
most 4.5 cm below the top, the reach travels 6 cm above the top, and a primitive class and sizes
(box, upright cylinder, ball, or lying cylinder, from footprint shape and proportions) stand in for the shape.

Every eval also shakes each lifted object (3 cm at 2 Hz along x, y, then z: up to 4.7 m/s^2) and says why each grasp
failed, with the hand's peak force on the object and peak motor torques.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("mode", choices=["train", "eval"])  # train runs --reps parallel copies of each layout
p.add_argument("--n", type=int, default=60, help="layouts per shape")
p.add_argument("--iters", type=int, default=60)
p.add_argument("--reps", type=int, default=0, help="parallel copies of each layout (default: 8 train, 1 eval)")
p.add_argument("--seed", type=int, default=0, help="layout seed base (default: 800000 train; eval: --split's, val 700000, test 900000)")
p.add_argument("--split", choices=["train", "val", "test"], default="val",
               help="eval: which layout seeds (isaac/bench_spec.py SPLITS): val for every comparison; test once, at the end")
p.add_argument("--policy", default="", help="eval: one or more comma-separated policy files, one scene build")
p.add_argument("--redraw", action="store_true", help="train: new object poses every round")
p.add_argument("--params", type=int, default=5, choices=[5, 8], help="train: adjustments the policy picks")
p.add_argument("--snap", type=int, default=0, help="train: also save policy-it<k>.pt every k rounds")
p.add_argument("--retries", type=int, default=0, help="closed loop: after a 3 cm test lift that fails, open, re-aim at "
               "where the object is now and grasp again, up to this many times")
p.add_argument("--direct", action="store_true", help="the old reach: straight at the target (it knocks ~10%% of objects "
               "off the table or through it on the way)")
p.add_argument("--vary", choices=["same", "turn", "low"], default="same",
               help="each retry's grasp: the same, turned 90 deg more each time, or 6 mm lower each time")
p.add_argument("--zero", action="store_true", help="eval with zero adjustment (the scripted pinch)")
p.add_argument("--quick", action="store_true", help="quicker reach (5.0 s, not 5.7 s); a policy trained with it lost to "
               "v1 on the normal-timing test (218 vs 242/360), so train and test with the same setting")
p.add_argument("--legacy", action="store_true", help="the v1-v7 settings: no perception error, 1000 N m waist, friction "
               "0.7-1.5, eggs as bounding spheres, no CCD (flags below override it one by one)")
p.add_argument("--noise", type=float, default=None, help="perception error, cm std per axis (sizes +- 5%%); default 0.4")
p.add_argument("--waist", choices=["real", "strong"], default=None, help="motor torques: real G1 or the old 1000 N m waist")
p.add_argument("--friction", default=None, help="object friction range lo,hi; default 0.5,1.0")
p.add_argument("--egg", choices=["hull", "sphere"], default=None, help="egg collider: convex hull, or the old bounding sphere")
p.add_argument("--ccd", type=int, choices=[0, 1], default=None, help="objects: continuous collision detection (default 1)")
p.add_argument("--ik", choices=["dls", "limits", "clamp", "null"], default="dls", help="arm IK: Isaac Lab's damped least squares (ignores "
               "joint limits: the drives then push joints into them), or the same with joint targets kept 0.05 rad inside "
               "the limits and the spare joints (9 for a 6-D pose) pulled toward READY and away from limits; clamp / null: "
               "only the clamp / only the pull (limits: scripted 68 vs 200/360 with dls, 2026-10-08)")
p.add_argument("--pinch", choices=["center", "width"], default="center", help="scripted pinch: at the object's centre, or "
               "shifted so the index pad starts 6 mm from the object's side (as the MuJoCo planner: g1.Hand)")
p.add_argument("--shake", action="store_true", help="train: also reward keeping the object through the shake test")
p.add_argument("--objects", choices=["primitives", "household"], default="primitives",
               help="household: the scanned household objects benchmark (eval only; --n = poses per object)")
p.add_argument("--tag", default="", help="eval: suffix for the result names (e.g. which settings were changed)")
p.add_argument("--lr", type=float, default=3e-3)
p.add_argument("--out", default="runs/isaac/rl")
p.add_argument("--unitree", default="/opt/dlami/nvme/w/unitree_sim_isaaclab")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = True


def _pick(v, new, old):
    return v if v is not None else (old if args.legacy else new)


NOISE = _pick(args.noise, 0.4, 0.0)
WAIST = _pick(args.waist, "real", "strong")
FRICTION = [float(v) for v in _pick(args.friction, "0.5,1.0", "0.7,1.5").split(",")]
EGG = _pick(args.egg, "hull", "sphere")
CCD = bool(_pick(args.ccd, 1, 0))
SETTINGS = dict(noise_cm=NOISE, waist=WAIST, friction=FRICTION, egg=EGG, ccd=CCD, ik=args.ik, pinch=args.pinch)
print(f"settings: {SETTINGS}", flush=True)
app = AppLauncher(args).app


def _fail(*exc):  # Kit's python.sh exits 0 after an uncaught exception; exit 1 so --dependency=afterok chains stop
    import traceback
    traceback.print_exception(*exc)
    sys.stdout.flush()
    os._exit(1)


sys.excepthook = _fail

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import ContactSensor, ContactSensorCfg  # noqa: E402
from isaaclab.utils.math import (compute_pose_error, matrix_from_quat, quat_apply, quat_error_magnitude, quat_from_angle_axis,  # noqa: E402
                                 quat_from_matrix, quat_mul, subtract_frame_transforms)
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade, Vt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from grab_policy import NAMES, Policy, SCALE as _SCALE  # noqa: E402  (the policy, shared with walk_grasp.py --grab)
os.environ["PROJECT_ROOT"] = args.unitree
sys.path.insert(0, args.unitree)
from robots.unitree import G129_CFG_WITH_DEX3_BASE_FIX  # noqa: E402
from so101 import g1  # noqa: E402
from so101.objects import layout, random_objects  # noqa: E402

DT, DECIM = 0.005, 2
PHASES = [("approach", 1.5), ("descend", 1.0), ("close", 1.2), ("lift", 1.0), ("hold", 1.0)]
if args.quick:
    PHASES = [("approach", 1.0), ("descend", 0.8), ("close", 1.2), ("lift", 1.0), ("hold", 1.0)]
REPS = args.reps or (8 if args.mode == "train" else 1)
LIFT = 0.10
SHAKE_A, SHAKE_F = 0.03, 2.0  # shake test: amplitude (m), frequency (Hz); peak acceleration A (2 pi f)^2 = 4.7 m/s^2
SHAPES = ["box", "cylinder", "lying cylinder", "ball", "capsule", "egg"]
HOUSE = args.objects == "household"
assert not (HOUSE and args.mode == "train"), "the household objects are a test set only"
EVAL_BASE = {"train": 800000, "val": 700000, "test": 900000}  # bench_spec.SPLITS; household objects are a test set (950000)
SEED = args.seed or (800000 if args.mode == "train" else (950000 if HOUSE else EVAL_BASE[args.split]))
dev = "cuda:0"
# adjustment = SCALE * clip(a, -2, 2), a ~ N(policy mean, std); "close" maps to the finger closing target, "close_rate"
# to log2 of the closing speed; a 5-parameter policy leaves the last three at zero (the scripted values)
SCALE = _SCALE.to(dev)  # m, m, m, rad, fraction, rad, rad, log2 (grab_policy.SCALE; NAMES from there too)
# the real G1's motor torque limits (N m; MuJoCo Menagerie g1 actuatorfrcrange)
G1_TORQUE = {"waist_yaw_joint": 88.0, "waist_roll_joint": 50.0, "waist_pitch_joint": 50.0, ".*_shoulder_.*_joint": 25.0,
             ".*_elbow_joint": 25.0, ".*_wrist_roll_joint": 25.0, ".*_wrist_(pitch|yaw)_joint": 5.0}

# --- layouts -------------------------------------------------------------------------------------------------------------
layouts = []
OBJ_DIR = Path(__file__).resolve().parent.parent / "so101" / "assets" / "objects"
if HOUSE:  # the household objects: --n poses each, friction drawn as the primitives' (then rescaled the same way)
    HOUSEHOLD = [e for e in json.loads((OBJ_DIR / "gso_g1_index.json").read_text())["objects"] if e["kept"]]
    HULLS = np.load(OBJ_DIR / "gso_g1" / "hulls.npz")
    for j, e in enumerate(HOUSEHOLD):
        for i in range(args.n):
            rng = np.random.default_rng([SEED, j, i])
            spec = dict(shape="household", type="hull", name=e["name"], entry=e, mass=e["mass_kg"],
                        friction=float(rng.uniform(0.7, 1.5)), foot=0.5 * float(np.hypot(e["width_m"], e["length_m"])),
                        size=[e["width_m"] / 2, e["length_m"] / 2, e["height_m"] / 2], z=e["height_m"] / 2, quat_axis=None)
            x, y, a = layout(rng, [spec])[0]
            layouts += [dict(shape="household", seed=SEED + 100 * j + i, spec=spec, x=x, y=y, heading=a)] * REPS
for s in ([] if HOUSE else SHAPES):
    for i in range(args.n):
        rng = np.random.default_rng(SEED + i)
        spec = random_objects(rng, 1, [s])[0]
        x, y, a = layout(rng, [spec])[0]
        L = dict(shape=s, seed=SEED + i, spec=spec, x=x, y=y, heading=a)
        layouts += [L] * REPS  # parallel copies (one dict): each grasp is compared with its own layout's other tries
N = len(layouts)
LAYOUT_ID = torch.arange(N, device="cuda:0") // REPS
SHAPE_ID = torch.tensor([SHAPES.index(L["shape"]) if L["shape"] in SHAPES else 0 for L in layouts], device=dev)  # household: set by the fit

# --- scene (as isaac/g1_lift.py) ---------------------------------------------------------------------------------------------
ARM, HAND = list(g1.ARM), list(g1.HAND)
open_q = dict(zip(HAND, [g1.THUMB_0, *g1.THUMB[-1, 1:], *g1.INDEX, *g1.MIDDLE]))
closed_q = dict(zip(HAND, [g1.THUMB_0, *g1.THUMB[0, 1:], *g1.INDEX, *g1.MIDDLE]))
# The left arm parked out to the side and folded: the IK moves the waist (yaw, pitch), and the MuJoCo pose (g1.LEFT_ARM)
# then lowers the left hand onto the table, where it knocked objects off (--direct keeps the old pose too)
LEFT_PARK = dict(g1.LEFT_ARM) if args.direct else {"left_shoulder_pitch_joint": 0.0, "left_shoulder_roll_joint": 0.6,
                                                  "left_elbow_joint": 1.9}
start_q = {k: float(v) for k, v in (dict(zip(ARM, g1.READY.tolist())) | LEFT_PARK | open_q).items()}
cfg = G129_CFG_WITH_DEX3_BASE_FIX.replace(prim_path="/World/envs/env_.*/Robot")
cfg.spawn.rigid_props.disable_gravity = True
cfg.spawn.articulation_props.enabled_self_collisions = True
cfg.init_state = cfg.init_state.replace(pos=tuple(g1.PELVIS.tolist()), joint_pos=cfg.init_state.joint_pos | start_q)
cfg.actuators["waist"] = cfg.actuators["waist"].replace(
    velocity_limit=None, velocity_limit_sim=None, effort_limit=None, stiffness=3000.0, damping=60.0,
    effort_limit_sim=1000.0 if WAIST == "strong" else {k: v for k, v in G1_TORQUE.items() if k.startswith("waist")})
if WAIST == "real":
    cfg.actuators["arms"] = cfg.actuators["arms"].replace(
        effort_limit=None, effort_limit_sim={k: v for k, v in G1_TORQUE.items() if not k.startswith("waist")})
cfg.actuators["hands"] = cfg.actuators["hands"].replace(
    effort_limit=None, effort_limit_sim={".*_hand_thumb_0_joint": 2.45, ".*_hand_(thumb_[12]|index_.|middle_.)_joint": g1.THUMB_FORCE},
    stiffness=500.0, damping=4.5, armature=0.01)
scene_cfg = InteractiveSceneCfg(num_envs=N, env_spacing=2.0, replicate_physics=False)
scene_cfg.robot = cfg
sim_cfg = sim_utils.SimulationCfg(dt=DT, device=dev, physx=sim_utils.PhysxCfg(enable_ccd=CCD))
sim = sim_utils.SimulationContext(sim_cfg)
scene = InteractiveScene(scene_cfg)

TABLE_POS = np.array([g1.PELVIS[0] + g1.TABLE_EDGE + 0.4, 0.0, -0.5])
TABLE_X, TABLE_Y = (TABLE_POS[0] - 0.4, TABLE_POS[0] + 0.4), 0.6  # the table top's extent (env frame)
table = sim_utils.CuboidCfg(size=(0.8, 1.2, 1.0), collision_props=sim_utils.CollisionPropertiesCfg(),
                            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.8))
shelf = sim_utils.CuboidCfg(size=(0.08, 0.12, 0.06), collision_props=sim_utils.CollisionPropertiesCfg())
LIE = [np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0]


def obj_quat(L):
    q = torch.tensor([np.cos(L["heading"] / 2), 0.0, 0.0, np.sin(L["heading"] / 2)])
    return quat_mul(q[None], torch.tensor(LIE)[None])[0] if L["spec"]["quat_axis"] == "y" else q


def spawn_household(i, L, fr, rigid):
    """One scanned object from its convex parts: rigid body with the index's mass, the friction draw, contact reports;
    footprint centre at the layout's (x, y), turned by its heading, resting 2 mm above the table."""
    e, path = L["spec"]["entry"], f"/World/envs/env_{i}/Object"
    a, (cx, cy) = L["heading"], e["centre_xy_m"]
    ca, sa = np.cos(a), np.sin(a)
    pos = (L["x"] - (ca * cx - sa * cy), L["y"] - (sa * cx + ca * cy), -e["bottom_z_m"] + 0.002)
    sim_utils.create_prim(path, "Xform", translation=pos, orientation=(np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)))
    V, F, fs = HULLS[f"{e['name']}/v"], HULLS[f"{e['name']}/f"], HULLS[f"{e['name']}/fstart"]
    for k in range(len(fs) - 1):
        f = F[fs[k]:fs[k + 1]]
        ids, f2 = np.unique(f, return_inverse=True)
        m = UsdGeom.Mesh.Define(stage, f"{path}/part_{k}")
        m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(V[ids])))
        m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f), 3, np.int32)))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f2.reshape(-1).astype(np.int32)))
        UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr().Set("convexHull")
        sim_utils.define_collision_properties(f"{path}/part_{k}", sim_utils.CollisionPropertiesCfg(
            torsional_patch_radius=0.01, min_torsional_patch_radius=0.005))
    sim_utils.define_rigid_body_properties(path, sim_utils.RigidBodyPropertiesCfg(**rigid))
    sim_utils.define_mass_properties(path, sim_utils.MassPropertiesCfg(mass=L["spec"]["mass"]))
    sim_utils.spawn_rigid_body_material(f"{path}/material", sim_utils.RigidBodyMaterialCfg(
        static_friction=fr, dynamic_friction=fr, restitution=0.0, friction_combine_mode="max"))
    sim_utils.bind_physics_material(path, f"{path}/material")
    sim_utils.activate_contact_sensors(path)
    if CCD:
        PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath(path)).CreateEnableCCDAttr().Set(True)


import omni.usd  # noqa: E402
stage = omni.usd.get_context().get_stage()
egg_approx = set()
for i, L in enumerate(layouts):
    table.func(f"/World/envs/env_{i}/Table", table, translation=tuple(TABLE_POS.tolist()))
    shelf.func(f"/World/envs/env_{i}/Shelf", shelf, translation=(0.32, 0.0, 0.03))
    s = L["spec"]
    fr = FRICTION[0] + (s["friction"] - 0.7) / 0.8 * (FRICTION[1] - FRICTION[0])  # the draw's place in 0.7-1.5, rescaled
    rigid = dict(solver_position_iteration_count=32, solver_velocity_iteration_count=1)
    if CCD:
        rigid["max_depenetration_velocity"] = 1.0
    common = dict(rigid_props=sim_utils.RigidBodyPropertiesCfg(**rigid),
                  mass_props=sim_utils.MassPropertiesCfg(mass=s["mass"]),
                  collision_props=sim_utils.CollisionPropertiesCfg(torsional_patch_radius=0.01, min_torsional_patch_radius=0.005),
                  physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=fr, dynamic_friction=fr,
                                                                  restitution=0.0, friction_combine_mode="max"),
                  activate_contact_sensors=True)
    if s["type"] == "hull":
        spawn_household(i, L, fr, rigid)
        continue
    z = s["size"]
    ocfg = {"box": lambda: sim_utils.CuboidCfg(size=(2 * z[0], 2 * z[1], 2 * z[2]), **common),
            "cylinder": lambda: sim_utils.CylinderCfg(radius=z[0], height=2 * z[1], axis="Z", **common),
            "capsule": lambda: sim_utils.CapsuleCfg(radius=z[0], height=2 * z[1], axis="Z", **common),
            "sphere": lambda: sim_utils.SphereCfg(radius=z[0], **common),
            "ellipsoid": lambda: sim_utils.MeshSphereCfg(radius=1.0, **common)}[s["type"]]()
    prim = ocfg.func(f"/World/envs/env_{i}/Object", ocfg, translation=(L["x"], L["y"], s["z"] + 0.001),
                     orientation=tuple(obj_quat(L).tolist()))
    if CCD:
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateEnableCCDAttr().Set(True)
    if s["type"] == "ellipsoid":
        at = prim.GetAttribute("xformOp:scale")
        at.Set(Gf.Vec3d(*z) if "double" in str(at.GetTypeName()) else Gf.Vec3f(*z))
        for c in Usd.PrimRange(prim):
            if c.HasAPI(UsdPhysics.MeshCollisionAPI):
                ap = UsdPhysics.MeshCollisionAPI(c).GetApproximationAttr()
                if EGG == "hull":
                    ap.Set("convexHull")
                egg_approx.add(ap.Get())
for i in range(N):  # the hand camera mount's collider fights the palm with self-collision on (LOG 2026-10-07)
    for side in ("left", "right"):
        pr = stage.GetPrimAtPath(f"/World/envs/env_{i}/Robot/{side}_hand_camera_base_link/collisions")
        if pr.IsValid():
            pr.SetActive(False)


def report_materials():
    """The physics materials on the right hand's colliders (env 0); unbound ones use the scene default."""
    seen = {}
    for pr in Usd.PrimRange(stage.GetPrimAtPath("/World/envs/env_0/Robot")):
        if "right_hand" not in str(pr.GetPath()) or not pr.HasAPI(UsdPhysics.CollisionAPI):
            continue
        try:
            mat = UsdShade.MaterialBindingAPI(pr).ComputeBoundMaterial("physics")[0]
        except Exception as e:  # noqa: BLE001
            seen[f"error {e}"] = seen.get(f"error {e}", 0) + 1
            continue
        if not mat or not mat.GetPrim().IsValid():
            key = "unbound (scene default)"
        else:
            mp = mat.GetPrim()
            pm = UsdPhysics.MaterialAPI(mp)
            cm = PhysxSchema.PhysxMaterialAPI(mp).GetFrictionCombineModeAttr().Get() if mp.HasAPI(PhysxSchema.PhysxMaterialAPI) else None
            key = f"{mp.GetPath().name}: static {pm.GetStaticFrictionAttr().Get()}, dynamic {pm.GetDynamicFrictionAttr().Get()}, combine {cm}"
        seen[key] = seen.get(key, 0) + 1
    d = sim_cfg.physics_material
    print(f"hand collider materials: {seen}; scene default: static {d.static_friction}, dynamic {d.dynamic_friction}, "
          f"combine {d.friction_combine_mode}", flush=True)


report_materials()
if egg_approx:
    print(f"egg collision approximation: {sorted(egg_approx)}", flush=True)

obj = RigidObject(RigidObjectCfg(prim_path="/World/envs/env_.*/Object", spawn=None))
scene.rigid_objects["object"] = obj
# the right hand's force on the object: grip, and pushing it down into the table (one filter per hand link: a pattern
# must match one body per env)
HAND_LINKS = sorted(c.GetName() for c in stage.GetPrimAtPath("/World/envs/env_0/Robot").GetChildren()
                    if c.GetName().startswith("right_hand_") and "camera" not in c.GetName() and c.HasAPI(UsdPhysics.RigidBodyAPI))
contact = ContactSensor(ContactSensorCfg(prim_path="/World/envs/env_.*/Object",
                                         filter_prim_paths_expr=[f"/World/envs/env_.*/Robot/{n}" for n in HAND_LINKS]))
scene.sensors["contact"] = contact
robot: Articulation = scene["robot"]
sim.reset()
scene.update(DT)
paths = list(obj.root_physx_view.prim_paths)
perm = torch.tensor([paths.index(f"/World/envs/env_{i}/Object") for i in range(N)], device=dev)
cpaths = list(contact.body_physx_view.prim_paths)
cperm = torch.tensor([cpaths.index(f"/World/envs/env_{i}/Object") for i in range(N)], device=dev)
print(f"contact sensor: {contact.contact_physx_view.filter_count} hand bodies per object {HAND_LINKS}", flush=True)
if HOUSE:  # let each scanned object settle (robot at its start pose), then start every episode from there
    _spawn = obj.data.root_state_w.clone()
    for _ in range(200):
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(DT)
OBJ0 = obj.data.root_state_w.clone()  # view order: written back as-is at every reset
Q0 = robot.data.joint_pos.clone()
env_o = scene.env_origins

jn = robot.joint_names
arm_ids = [jn.index(j) for j in ARM]
hand_ids = [jn.index(j) for j in HAND]
hb = robot.body_names.index(g1.HAND_BODY)
jac_idx = hb - 1 if robot.is_fixed_base else hb
ik = DifferentialIKController(DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
                              num_envs=N, device=dev)
pinch_local = torch.tensor(g1.PINCH, dtype=torch.float32, device=dev)
H = torch.tensor(np.column_stack([g1.DOWN, g1.CLOSE, np.cross(g1.DOWN, g1.CLOSE)]), dtype=torch.float32, device=dev)
R0 = matrix_from_quat(robot.data.body_quat_w[:, hb])
OPEN = torch.tensor([float(open_q[j]) for j in HAND], device=dev)
CLOSED = torch.tensor([float(closed_q[j]) for j in HAND], device=dev)
UP = torch.tensor([0.0, 0.0, 1.0], device=dev)
_lim = robot.root_physx_view.get_dof_max_forces()[0]
_vel = robot.root_physx_view.get_dof_max_velocities()[0]
print("motor limits in the sim, N m / rad/s: " + ", ".join(f"{j.replace('_joint', '')} {float(_lim[jn.index(j)]):.4g}/"
      f"{float(_vel[jn.index(j)]):.3g}" for j in ARM), flush=True)
# peak torques are reported as a fraction of the real G1 motor limits (ARM order)
import re  # noqa: E402
REAL_LIM = torch.tensor([next(v for k, v in G1_TORQUE.items() if re.fullmatch(k, j)) for j in ARM], device=dev)


if HOUSE:
    import importlib.util  # noqa: E402
    _gs = importlib.util.spec_from_file_location("gso_g1", OBJ_DIR / "gso_g1.py")
    gso_g1 = importlib.util.module_from_spec(_gs)
    _gs.loader.exec_module(gso_g1)
    HV = [HULLS[f"{L['spec']['name']}/v"].astype(float) for L in layouts]


FINGER_DEPTH = 0.045  # deepest pinch below an object's top (Menagerie Dex3: palm and wrist 5.8 cm above the pinch point)


def fit_household(state):
    """Perception's fit of each env's scanned object (root states in env order, env frame positions): pinch centre
    (footprint box centre, half height), closing heading (across the narrowest width), primitive class and sizes."""
    st = state.cpu().numpy()
    R = matrix_from_quat(state[:, 3:7]).cpu().numpy()
    cen, head, cls, size, top = np.zeros((N, 3)), np.zeros(N), [], np.zeros((N, 3)), np.zeros(N)
    for i in range(N):
        v = HV[i] @ R[i].T + st[i, :3]
        w, nar, ln, area, c2 = gso_g1.footprint(v[:, :2])
        z0, z1 = v[:, 2].min(), v[:, 2].max()
        h, fill = z1 - z0, area / max(w * ln, 1e-9)
        # pinch at mid-height, but no deeper than FINGER_DEPTH below the top: the palm is 5.8 cm above the pinch point
        cen[i], head[i], top[i] = [c2[0], c2[1], max((z0 + z1) / 2, z1 - FINGER_DEPTH)], nar, z1
        if fill > 0.9:
            k, sz = "box", [w / 2, ln / 2, h / 2]
        elif w / ln > 0.8:
            k, sz = ("ball", [(w + ln + h) / 6, 0, 0]) if 0.75 < h / w < 1.33 else ("cylinder", [(w + ln) / 4, h / 2, 0])
        elif 0.7 < h / w < 1.3:
            k, sz = "lying cylinder", [(w + h) / 4, ln / 2, 0]
        else:
            k, sz = "box", [w / 2, ln / 2, h / 2]
        cls.append(k)
        size[i] = sz
    return cen, head, cls, size, top


def narrow_heading(L):
    s, a = L["spec"], L["heading"]
    if s["type"] == "box":
        return a if s["size"][0] <= s["size"][1] else a + np.pi / 2
    if s["quat_axis"] == "y" or s["type"] == "ellipsoid":
        return a + np.pi / 2
    return a


def perception(tag=()):
    """Perception error per env, as so101/g1_grasps.perceived: centre offset (m, `NOISE` cm std per axis) and one size
    factor (1 +- 5 %), the same for a layout's copies. Seeded by layout and tag: eval's first look uses (seed, 13) like
    the MuJoCo test, re-aims add the try number; training draws fresh errors every round."""
    dc, sc = np.zeros((N, 3)), np.ones(N)
    if NOISE > 0:
        for j in range(0, N, REPS):
            rng = np.random.default_rng([layouts[j]["seed"], 13, *tag])
            dc[j:j + REPS], sc[j:j + REPS] = rng.normal(0, NOISE / 100, 3), 1 + rng.normal(0, 0.05)
    return dc, sc


def pose_tensors(dc=None, sc=None):
    """Grasp heading, pinch centre and policy features per env, from the layouts' object poses as perceived (dc, sc:
    perception(); one size factor keeps which side is narrow, so the grasp heading is unchanged)."""
    dc = np.zeros((N, 3)) if dc is None else dc
    sc = np.ones(N) if sc is None else sc
    if HOUSE:
        cen, head, cls, size, _ = FIT
        c0 = torch.tensor(np.column_stack([cen[:, :2] + dc[:, :2], np.maximum(cen[:, 2] + dc[:, 2], g1.Hand.MIN_Z)]),
                          dtype=torch.float32, device=dev)
        feat = torch.tensor([[float(k == s) for s in SHAPES] + list(size[i] * sc[i] / 0.03)
                             + [np.sin(2 * head[i]), np.cos(2 * head[i]), (cen[i, 0] + dc[i, 0] - 0.15) / 0.05,
                                (cen[i, 1] + dc[i, 1]) / 0.1, (cen[i, 2] + dc[i, 2]) / 0.03] for i, k in enumerate(cls)],
                            dtype=torch.float32, device=dev)
        return torch.tensor(head, dtype=torch.float32, device=dev), c0, feat
    h0 = torch.tensor([narrow_heading(L) for L in layouts], dtype=torch.float32, device=dev)
    c0 = torch.tensor([[L["x"] + d[0], L["y"] + d[1], max(L["spec"]["z"] + d[2], g1.Hand.MIN_Z)] for L, d in zip(layouts, dc)],
                      dtype=torch.float32, device=dev)
    feat = torch.tensor([[float(L["shape"] == s) for s in SHAPES] + list(np.asarray(L["spec"]["size"][:3], float) * f / 0.03)
                         + [np.sin(2 * narrow_heading(L)), np.cos(2 * narrow_heading(L)), (L["x"] + d[0] - 0.15) / 0.05,
                            (L["y"] + d[1]) / 0.1, (L["spec"]["z"] + d[2]) / 0.03] for L, d, f in zip(layouts, dc, sc)],
                        dtype=torch.float32, device=dev)
    return h0, c0, feat


def obj_states():
    """Object root states (view order) for the layouts' current poses, at rest."""
    st = torch.zeros(N, 13, device=dev)
    st[:, :3] = env_o + torch.tensor([[L["x"], L["y"], L["spec"]["z"] + 0.001] for L in layouts], device=dev)
    st[:, 3:7] = torch.stack([obj_quat(L) for L in layouts]).float().to(dev)
    out = torch.empty_like(st)
    out[perm] = st
    if Z_OFF is not None:
        out[:, 2] += Z_OFF
    return out


def redraw(it):
    """New position and heading for every layout (its copies share them), drawn as the test draws them."""
    global OBJ0
    for L in layouts[::REPS]:
        L["x"], L["y"], L["heading"] = layout(np.random.default_rng([L["seed"], it, 17]), [L["spec"]])[0]
    OBJ0 = obj_states()


if HOUSE:
    _st = OBJ0[perm].clone()
    _st[:, :3] -= env_o
    FIT = fit_household(_st)
    SHAPE_ID = torch.tensor([SHAPES.index(k) for k in FIT[2]], device=dev)
    _sp = _spawn[perm]
    _tilt = torch.rad2deg(quat_error_magnitude(_sp[:, 3:7], OBJ0[perm, 3:7]))
    _drift = (OBJ0[perm, :2] - _sp[:, :2]).norm(dim=1)
    print(f"household: {len(HOUSEHOLD)} objects x {args.n} poses; after 1 s settling: tilted > 15 deg {int((_tilt > 15).sum())}, "
          f"moved > 2 cm {int((_drift > 0.02).sum())}, below the table {int((OBJ0[perm, 2] - env_o[:, 2] < -0.05).sum())}; "
          f"fitted as { {s: FIT[2].count(s) for s in SHAPES if FIT[2].count(s)} }", flush=True)
H0, C0, FEAT = pose_tensors(*perception())
# the rebuilt states must match what was spawned in x, y and orientation. Heights differ: startup steps drop objects
# 0.7 mm onto the table and PhysX pushed eggs (bounding spheres, --egg=sphere) up to ~3 cm out of it, so keep each env's
# offset and reuse it for new poses (eggs only turn about z, so the offset holds); every reset then starts an object as
# the test's does.
Z_OFF = None
_s = obj_states() if not HOUSE else OBJ0.clone()
_dxy = (_s[:, :2] - OBJ0[:, :2]).abs().max().item()
_dq = torch.minimum((_s[:, 3:7] - OBJ0[:, 3:7]).abs().amax(1), (_s[:, 3:7] + OBJ0[:, 3:7]).abs().amax(1)).max().item()
Z_OFF = OBJ0[:, 2] - _s[:, 2]
_dz = Z_OFF[perm]
print(f"object state check: xy {_dxy:.1e}, quat {_dq:.1e}; startup height offset per shape, max (mm) "
      f"{ {s: round(1000 * _dz[SHAPE_ID == j].abs().max().item(), 1) for j, s in enumerate(SHAPES) if (SHAPE_ID == j).any()} }", flush=True)
_eg = [i for i, L in enumerate(layouts) if L["spec"]["type"] == "ellipsoid"]
if HOUSE:
    Z_OFF = torch.zeros_like(Z_OFF)
if _eg:  # a bounding sphere rests its centre at the long half-axis, not the short one the egg lies on
    _sp = torch.tensor([layouts[i]["spec"]["size"][0] - layouts[i]["spec"]["size"][2] for i in _eg], device=dev)
    print(f"eggs: startup offset vs (long - short half-axis): max |difference| {1000 * (_dz[_eg] - _sp).abs().max().item():.1f} mm, "
          f"max offset {1000 * _dz[_eg].abs().max().item():.1f} mm, max (long - short) {1000 * _sp.max().item():.1f} mm", flush=True)
assert _dxy < 1e-4 and _dq < 1e-3, (_dxy, _dq)


def goals(adj):
    """Per env: pinch centre (env frame) and hand orientation for the adjusted grasp (adj: N x 5, physical units)."""
    h = H0 + adj[:, 3]
    best_q, best_s, best_cw = None, None, None
    dw = torch.tensor([0.0, 0.0, -1.0], device=dev).expand(N, 3)
    for sgn in (1.0, -1.0):  # the closing direction's two signs: the smaller turn from the start orientation
        cw = sgn * torch.stack([torch.cos(h), torch.sin(h), torch.zeros_like(h)], 1)
        W = torch.stack([dw, cw, torch.linalg.cross(dw, cw)], 2)
        R = W @ H.T
        score = torch.einsum("nij,nij->n", R0, R)
        if best_s is None:
            best_q, best_s, best_cw = quat_from_matrix(R), score, cw
        else:
            take = score > best_s
            best_q = torch.where(take[:, None], quat_from_matrix(R), best_q)
            best_cw = torch.where(take[:, None], cw, best_cw)
            best_s = torch.maximum(score, best_s)
    across = torch.linalg.cross(UP.expand(N, 3), best_cw)
    c = C0 + adj[:, :1] * best_cw + adj[:, 1:2] * across + adj[:, 2:3] * UP
    if args.pinch == "width":  # index pad 1.83 cm before the pinch point along the closing direction: start it 6 mm clear
        sz, k = FEAT[:, 6:9] * 0.03, FEAT[:, :6].argmax(1)  # perceived sizes and class (box 0, egg 5)
        half = torch.where(k == 0, sz[:, :2].min(1).values, torch.where(k == 5, sz[:, 1], sz[:, 0]))
        c = c + (PAD_TO_PINCH - g1.Hand.FIXED_CLEAR - half)[:, None] * best_cw
    c[:, 2] = torch.clamp(c[:, 2], min=0.008)
    # tilt about the pinch point, in a frame fixed by the heading (not by the closing direction's chosen sign)
    eh = torch.stack([torch.cos(h), torch.sin(h), torch.zeros_like(h)], 1)
    ep = torch.linalg.cross(UP.expand(N, 3), eh)
    tilt = quat_mul(quat_from_angle_axis(adj[:, 5], ep), quat_from_angle_axis(adj[:, 6], eh))
    return c, quat_mul(tilt, best_q)


PAD_TO_PINCH = 0.0183  # index pad to the pinch point along the closing direction (python -m so101.g1 hand)
SAFE_Z = 0.15  # the reach's travel height above the table (pinch point); household: 6 cm above the object's top
SAFE = torch.full((N, 1), SAFE_Z, device=dev)
if HOUSE:
    SAFE = torch.clamp(torch.tensor(FIT[4], dtype=torch.float32, device=dev)[:, None] + 0.06, min=SAFE_Z)


def pinch_now():
    """Each env's pinch point now, env frame."""
    return robot.data.body_pos_w[:, hb] + quat_apply(robot.data.body_quat_w[:, hb], pinch_local.expand(N, 3)) - env_o


def nlerp(q0, q1, s):
    q1 = torch.where(((q0 * q1).sum(1, keepdim=True) < 0), -q1, q1)
    q = q0 + s * (q1 - q0)
    return q / q.norm(dim=1, keepdim=True)


def reach(p0, q0, goal, gq, frac):
    """The approach path: up from p0 to the travel height (hand turning as it was), across while it turns to the grasp's
    orientation, then down to goal; smooth in each third. --direct: straight at goal, turned at once (the old reach)."""
    if args.direct:
        return goal, gq
    top0 = torch.cat([p0[:, :2], torch.maximum(p0[:, 2:], SAFE)], 1)
    top1 = torch.cat([goal[:, :2], SAFE], 1)
    seg, s = min(int(frac * 3), 2), min(1.0, frac * 3 - min(int(frac * 3), 2))
    s = s * s * (3 - 2 * s)
    a, b = [(p0, top0), (top0, top1), (top1, goal)][seg]
    return a + s * (b - a), [q0, nlerp(q0, gq, s), gq][seg]


def reset():
    robot.write_joint_state_to_sim(Q0, torch.zeros_like(Q0))
    robot.set_joint_position_target(Q0)
    obj.write_root_state_to_sim(OBJ0)
    ik.reset()
    for _ in range(10):
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(DT)


# --- what happened in each env, for the failure breakdown -------------------------------------------------------------------
TR = {}


def track_start():
    TR.update(p0=obj.data.root_pos_w[perm] - env_o, k=0, fell=torch.full((N,), -1, device=dev),
              fell_xy=torch.zeros(N, 2, device=dev), max_rise=torch.zeros(N, device=dev),
              f_hand=torch.zeros(N, device=dev), f_down=torch.zeros(N, device=dev), tau=torch.zeros(N, len(ARM), device=dev),
              ik_pos=torch.zeros(N, device=dev), ik_ang=torch.zeros(N, device=dev),
              at_lim=torch.zeros(N, len(ARM), dtype=torch.bool, device=dev))


def track():
    op = obj.data.root_pos_w[perm] - env_o
    TR["max_rise"] = torch.maximum(TR["max_rise"], op[:, 2] - TR["p0"][:, 2])
    new = (TR["fell"] < 0) & (op[:, 2] < -0.05)  # below the table top (z = 0)
    TR["fell"][new] = TR["k"]
    TR["fell_xy"][new] = op[new, :2]
    fh = contact.data.force_matrix_w[cperm, 0].sum(1)  # the right hand's force on the object (N x 3)
    TR["f_hand"] = torch.maximum(TR["f_hand"], fh.norm(dim=1))
    TR["f_down"] = torch.maximum(TR["f_down"], -fh[:, 2])
    TR["tau"] = torch.maximum(TR["tau"], robot.data.applied_torque[:, arm_ids].abs())
    TR["k"] += 1


def track_aim(pin, gq, where=None):
    """The arm's pose error at the end of the descent: how far the pinch point and hand are from the grasp."""
    pe, ae = (pinch_now() - pin).norm(dim=1), quat_error_magnitude(robot.data.body_quat_w[:, hb], gq)
    where = torch.ones(N, dtype=torch.bool, device=dev) if where is None else where
    TR["ik_pos"], TR["ik_ang"] = torch.where(where, pe, TR["ik_pos"]), torch.where(where, ae, TR["ik_ang"])
    lim, q = robot.data.joint_pos_limits[:, arm_ids], robot.data.joint_pos[:, arm_ids]  # which arm joints sit at a limit
    near = (q < lim[..., 0] + 0.03) | (q > lim[..., 1] - 0.03)
    TR["at_lim"] = torch.where(where[:, None], near, TR["at_lim"])


ARM_LO = robot.data.joint_pos_limits[:, arm_ids, 0] + 0.05
ARM_HI = robot.data.joint_pos_limits[:, arm_ids, 1] - 0.05
READY_Q = torch.tensor(g1.READY, dtype=torch.float32, device=dev)
EYE6, EYE9 = torch.eye(6, device=dev), torch.eye(len(ARM), device=dev)


def ik_limits(ep, eq, tp, tq, jac, q):
    """One IK step as Isaac Lab's damped least squares (lambda 0.01), plus, in the pose task's null space, a pull toward
    READY and a push out of the last 0.25 rad before each limit; targets clamped 0.05 rad inside the limits."""
    pe, re = compute_pose_error(ep, eq, tp, tq, rot_error_type="axis_angle")
    Jt = jac.transpose(1, 2)
    Jp = Jt @ torch.linalg.solve(jac @ Jt + 1e-4 * EYE6, EYE6.expand(len(q), 6, 6))
    dq = (Jp @ torch.cat([pe, re], 1).unsqueeze(-1)).squeeze(-1)
    push = (ARM_LO + 0.25 - q).clamp(min=0) - (q - (ARM_HI - 0.25)).clamp(min=0)
    g = 0.05 * (READY_Q - q) + 0.5 * push
    if args.ik != "clamp":
        dq = dq + ((EYE9 - Jp @ jac) @ g.unsqueeze(-1)).squeeze(-1)
    return q + dq if args.ik == "null" else torch.clamp(q + dq, ARM_LO, ARM_HI)


def drive(pinch, gq, f):
    """One control step: the hand's pinch point to pinch (env frame) at orientation gq, fingers closed by fraction f."""
    tp = pinch + env_o - quat_apply(gq, pinch_local.expand(N, 3))
    root_p, root_q = robot.data.root_pos_w, robot.data.root_quat_w
    tpl, tql = subtract_frame_transforms(root_p, root_q, tp, gq)
    ep, eq = subtract_frame_transforms(root_p, root_q, robot.data.body_pos_w[:, hb], robot.data.body_quat_w[:, hb])
    jac = robot.root_physx_view.get_jacobians()[:, jac_idx, :, arm_ids]
    if args.ik != "dls":
        qt = ik_limits(ep, eq, tpl, tql, jac, robot.data.joint_pos[:, arm_ids])
    else:
        ik.set_command(torch.cat([tpl, tql], 1))
        qt = ik.compute(ep, eq, jac, robot.data.joint_pos[:, arm_ids])
    robot.set_joint_position_target(qt, joint_ids=arm_ids)
    robot.set_joint_position_target(OPEN + f[:, None] * (CLOSED - OPEN), joint_ids=hand_ids)
    for _ in range(DECIM):
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(DT)
    track()


def measure(z0):
    op = obj.data.root_pos_w[perm]
    pw = robot.data.body_pos_w[:, hb] + quat_apply(robot.data.body_quat_w[:, hb], pinch_local.expand(N, 3))
    return op[:, 2] - z0, torch.linalg.norm(op - pw, dim=1)


def finish(top, q_hold, close_to, z0, shake):
    """After the hold: rise and nearness (the lift test), then optionally the shake test: the hand moves 3 cm at 2 Hz
    along x, then y, then z (one cycle each), holds 0.5 s, and the object must still be in it, 5 cm up."""
    rise, near = measure(z0)
    TR["p_end"] = obj.data.root_pos_w[perm] - env_o
    if not shake:
        return rise, near, None
    n = int(round(0.5 / (DT * DECIM)))
    for ax in range(4):
        for s in range(n):
            off = torch.zeros(3, device=dev)
            if ax < 3:
                off[ax] = SHAKE_A * np.sin(2 * np.pi * SHAKE_F * s * DT * DECIM)
            drive(top + off, q_hold, close_to)
    r2, n2 = measure(z0)
    return rise, near, (r2 > 0.05) & (n2 < 0.05)


def episode(adj, shake=False):
    """One grasp per environment with the given adjustments; returns rise, near (object to pinch point) and, with shake,
    whether the object stayed in the hand through the shake test."""
    reset()
    c, gq = goals(adj)
    close_to = torch.clamp(1.0 - adj[:, 4].abs(), 0.6, 1.0)  # how far the fingers close (1 = the scripted full close)
    rate = torch.pow(2.0, adj[:, 7])  # closing speed relative to the scripted 1.2 s close
    z0 = obj.data.root_pos_w[perm, 2].clone()
    p0, q0 = pinch_now(), robot.data.body_quat_w[:, hb].clone()
    track_start()
    zero = torch.zeros(N, device=dev)
    steps = int(round(sum(d for _, d in PHASES) / (DT * DECIM)))
    edges = np.cumsum([0] + [d for _, d in PHASES])
    last = None
    for k in range(steps):
        t = k * DT * DECIM
        i = int(np.searchsorted(edges, t, side="right") - 1)
        ph, frac = PHASES[i][0], (t - edges[i]) / PHASES[i][1]
        if ph == "close" and last == "descend":
            track_aim(c, gq)
        last = ph
        s = min(1.0, frac * 1.5)
        q = gq
        if ph == "approach":
            (pinch, q), f = reach(p0, q0, c + 0.08 * UP, gq, frac), zero
        elif ph == "descend":
            pinch, f = c + (1 - s) * 0.08 * UP, zero
        elif ph == "close":
            pinch, f = c, torch.minimum(frac * rate, close_to)
        elif ph == "lift":
            pinch, f = c + s * LIFT * UP, close_to
        else:
            pinch, f = c + LIFT * UP, close_to
        drive(pinch, q, f)
    return finish(c + LIFT * UP, gq, close_to, z0, shake)


NH_OFF = torch.tensor([0.0 if HOUSE else narrow_heading(L) - L["heading"] for L in layouts], dtype=torch.float32, device=dev)


def obj_now(k):
    """Each env's object as a camera would see it now (look k's perception error): centre (env frame), grasp heading."""
    st = obj.data.root_state_w[perm]
    if HOUSE:  # fit the object again where it is now
        s2 = st.clone()
        s2[:, :3] -= env_o
        cen, head, _, _, _ = fit_household(s2)
        c = torch.tensor(cen + perception((k,))[0], dtype=torch.float32, device=dev)
        c[:, 2] = torch.clamp(c[:, 2], min=g1.Hand.MIN_Z)
        return c, torch.tensor(head, dtype=torch.float32, device=dev)
    R = matrix_from_quat(st[:, 3:7])
    a = torch.atan2(R[:, 1, 1], R[:, 0, 1]) - np.pi / 2  # the body's y axis: heading + 90 deg, upright or lying
    c = st[:, :3] - env_o + torch.tensor(perception((k,))[0], dtype=torch.float32, device=dev)
    c[:, 2] = torch.clamp(c[:, 2], min=g1.Hand.MIN_Z)
    return c, a + NH_OFF


def episode_retry(adj, retries, shake=False):
    """Closed loop: grasp, test-lift 3 cm; where the object did not come up, open, re-aim at where it is now and grasp
    again (up to retries times). Then every env lifts 10 cm and holds, as in episode()."""
    global H0, C0
    h0, c0 = H0.clone(), C0.clone()
    reset()
    close_to = torch.clamp(1.0 - adj[:, 4].abs(), 0.6, 1.0)
    rate = torch.pow(2.0, adj[:, 7])
    z0 = obj.data.root_pos_w[perm, 2].clone()
    zero = torch.zeros(N, device=dev)
    got = torch.zeros(N, dtype=torch.bool, device=dev)
    base, q_hold, tries = None, None, torch.zeros(N, device=dev)
    track_start()
    for k in range(retries + 1):
        a = adj.clone()  # a retry may change the grasp (--vary): the same one tends to fail the same way
        if k and args.vary == "turn":
            a[:, 3] += k * np.pi / 2
        elif k and args.vary == "low":
            a[:, 2] -= 0.006 * k
        c, gq = goals(a)
        base = c if base is None else torch.where(got[:, None], base, c)
        q_hold = gq if q_hold is None else torch.where(got[:, None], q_hold, gq)
        tries += (~got).float()
        p0, q0 = pinch_now(), robot.data.body_quat_w[:, hb].clone()
        for ph, dur in (("approach", 1.5 if k == 0 else 1.0), ("descend", 1.0), ("close", 1.2), ("test", 0.6)):
            n = int(round(dur / (DT * DECIM)))
            if ph == "close":
                track_aim(base, q_hold, ~got)
            for s in range(n):
                frac = s / n
                sm = min(1.0, frac * 1.5)
                qa = gq
                if ph == "approach":
                    (pin, qa), f = reach(p0, q0, base + 0.08 * UP, gq, frac), zero
                elif ph == "descend":
                    pin, f = base + (1 - sm) * 0.08 * UP, zero
                elif ph == "close":
                    pin, f = base, torch.minimum(frac * rate, close_to)
                else:
                    pin, f = base + sm * 0.03 * UP, close_to
                held = got[:, None]  # envs already holding stay put, closed
                drive(torch.where(held, base + 0.03 * UP, pin), torch.where(held, q_hold, qa), torch.where(got, close_to, f))
        rise, near = measure(z0)
        got |= (rise > 0.02) & (near < 0.05)
        if k < retries:  # re-aim where the object is now (not at objects that left the table: they stay lost)
            nc, nh = obj_now(k + 1)
            keep = got | (obj.data.root_pos_w[perm, 2] - env_o[:, 2] < -0.05)
            C0 = torch.where(keep[:, None], C0, nc)
            H0 = torch.where(keep, H0, nh)
    n = int(round(1.0 / (DT * DECIM)))
    for s in range(2 * n):  # lift the rest of the way (from 3 cm), then hold
        sm = min(1.0, s / n * 1.5)
        drive(base + (0.03 + sm * (LIFT - 0.03)) * UP, q_hold, close_to)
    H0, C0 = h0, c0
    episode_retry.tries = tries
    return finish(base + LIFT * UP, q_hold, close_to, z0, shake)


def why(ok, kept):
    """Each env's outcome: kept / dropped in shake (lifted; or 'lifted' without the shake test), else the first that
    applies: fell off or through the table, arm off target (> 1 cm or 10 deg at the end of the descent), slipped (came
    up > 2 cm, then lost), knocked (moved > 2 cm on the table), missed."""
    fell = (TR["fell"] >= 0).tolist()
    fx, fy = TR["fell_xy"][:, 0], TR["fell_xy"][:, 1]
    on_table = ((fx > TABLE_X[0]) & (fx < TABLE_X[1]) & (fy.abs() < TABLE_Y)).tolist()
    off = ((TR["ik_pos"] > 0.01) | (TR["ik_ang"] > np.radians(10))).tolist()
    slip = (TR["max_rise"] > 0.02).tolist()
    moved = ((TR["p_end"][:, :2] - TR["p0"][:, :2]).norm(dim=1) > 0.02).tolist()
    out = []
    for i in range(N):
        if ok[i]:
            out.append("lifted" if kept is None else ("kept" if kept[i] else "dropped in shake"))
        elif fell[i]:
            out.append("fell through table" if on_table[i] else "fell off table")
        elif off[i]:
            out.append("arm off target")
        elif slip[i]:
            out.append("slipped")
        elif moved[i]:
            out.append("knocked")
        else:
            out.append("missed")
    return out


def pct(x, qs=(0.5, 0.9, 1.0)):
    return "/".join(f"{torch.quantile(x.float(), q).item():.3g}" for q in qs)


out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)


def load(path):
    sd = torch.load(path, map_location=dev)
    pol = Policy(FEAT.shape[1], sd["net.4.weight"].shape[0]).to(dev)
    pol.load_state_dict(sd)
    return pol


def act(a):
    """Physical adjustments (N x 8) from policy outputs (N x 5 or 8)."""
    return SCALE * torch.nn.functional.pad(a, (0, 8 - a.shape[1])).clamp(-2, 2)


pol = Policy(FEAT.shape[1], args.params).to(dev)


def summary(rise, near):
    ok, held = (rise > LIFT - 0.01) & (near < 0.05), (rise > 0.05) & (near < 0.05)
    per = {s: int(ok[SHAPE_ID == j].sum()) for j, s in enumerate(SHAPES)}
    return int(ok.sum()), int(held.sum()), per, ok, held


if args.mode == "eval":
    for path in ([""] if args.zero else args.policy.split(",")):  # "zero" in the list: the scripted pinch
        zero = args.zero or path == "zero"
        with torch.no_grad():
            adj = act(torch.zeros(N, 5, device=dev) if zero else load(path)(FEAT)[0])
        rise, near, kept = episode_retry(adj, args.retries, True) if args.retries else episode(adj, True)
        lifted, held, per, ok, hd = summary(rise, near)
        kept = kept & ok
        cls = why(ok.tolist(), kept.tolist())
        tag = ("zero" if zero else (Path(path).parent.name + "-" + Path(path).stem)) + ("-quick" if args.quick else "") + \
            (f"-retry{args.retries}" if args.retries else "") + (f"-{args.vary}" if args.vary != "same" else "") + \
            ("-legacy" if args.legacy else "") + (f"-ik{args.ik}" if args.ik != "dls" else "") + \
            (f"-pinch{args.pinch}" if args.pinch != "center" else "") + (f"-{args.tag}" if args.tag else "") + ("-household" if HOUSE else "")
        if args.retries:
            print(f"tries: {episode_retry.tries.mean().item():.2f} per env on average", flush=True)
        tau = TR["tau"] / REAL_LIM  # peak torque as a fraction of the real G1 motor's limit
        res = dict(seed=SEED, n=args.n, policy=path, zero=zero, settings=SETTINGS, lifted=lifted, held=held,
                   kept=int(kept.sum()), of=N, per_shape=per, why={c: cls.count(c) for c in sorted(set(cls))},
                   objects=args.objects,
                   rows=[dict(shape=L["shape"] if not HOUSE else FIT[2][i], name=L["spec"].get("name"), seed=L["seed"], success=bool(ok[i]), held=bool(hd[i]), kept=bool(kept[i]),
                              why=cls[i], rise=round(float(rise[i]), 4), near=round(float(near[i]), 4),
                              max_rise=round(float(TR["max_rise"][i]), 4), hand_force=round(float(TR["f_hand"][i]), 2),
                              push_down=round(float(TR["f_down"][i]), 2), aim_err_mm=round(1000 * float(TR["ik_pos"][i]), 1),
                              aim_err_deg=round(float(np.degrees(TR["ik_ang"][i].item())), 1),
                              torque_frac=[round(v, 2) for v in tau[i].tolist()],
                              at_limit=[ARM[j].replace("_joint", "") for j in TR["at_lim"][i].nonzero().flatten().tolist()], adj=dict(zip(NAMES, adj[i].tolist())))
                         for i, L in enumerate(layouts)])
        (out / f"eval-{tag}-s{SEED}-n{args.n}.json").write_text(json.dumps(res, indent=1))
        print(f"EVAL {tag} seeds {SEED}+: lifted {lifted}/{N}, held {held}/{N}; per shape {per}; kept {int(kept.sum())}/{N} "
              f"after the shake", flush=True)
        print(f"WHY {tag}: {res['why']}", flush=True)
        print(f"FORCE {tag}: hand on object peak N (median/p90/max) {pct(TR['f_hand'])}; pushing down {pct(TR['f_down'])}; "
              f"peak torque / real G1 limit, p90/max: waist yaw {pct(tau[:, 0], (0.9, 1.0))}, waist pitch "
              f"{pct(tau[:, 1], (0.9, 1.0))}, arm {pct(tau[:, 2:6].amax(1), (0.9, 1.0))}, wrist {pct(tau[:, 6:].amax(1), (0.9, 1.0))}",
              flush=True)
else:
    opt = torch.optim.Adam(pol.parameters(), lr=args.lr)
    log = open(out / "train.jsonl", "a")
    if args.policy:
        pol = load(args.policy)
    for it in range(args.iters):
        t0 = time.time()
        if args.redraw:
            redraw(it)
        if args.redraw or NOISE > 0:
            H0, C0, FEAT = pose_tensors(*perception((7, it)))
        mu, std = pol(FEAT)
        dist = torch.distributions.Normal(mu, std)
        a = dist.sample()
        rise, near, kept = episode(act(a), args.shake)
        lifted, held, per, ok, _ = summary(rise, near)
        r = ok.float() + 0.5 * (rise / LIFT).clamp(0, 1) - 0.3 * (near > 0.2).float()
        if args.shake:
            r = r + (kept & ok).float()
        if REPS > 1:  # baseline: the mean over the same layout's copies (low variance)
            g = r.view(-1, REPS)
            adv = ((g - g.mean(1, keepdim=True)) / (g.std(1, keepdim=True) + 0.1)).view(-1)
        else:
            adv = torch.zeros_like(r)
            for j in range(len(SHAPES)):  # per-shape baseline
                m = SHAPE_ID == j
                adv[m] = (r[m] - r[m].mean()) / (r[m].std() + 1e-6)
        loss = -(dist.log_prob(a).sum(1) * adv).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        rec = dict(it=it, lifted=lifted, held=held, of=N, per_shape=per, reward=round(float(r.mean()), 4),
                   std=[round(v, 3) for v in std.tolist()], mean_adj=dict(zip(NAMES, act(mu).mean(0).tolist())),
                   s=round(time.time() - t0, 1))
        if args.shake:
            rec["kept"] = int((kept & ok).sum())
        log.write(json.dumps(rec) + "\n")
        log.flush()
        torch.save(pol.state_dict(), out / "policy.pt")
        if args.snap and (it + 1) % args.snap == 0:
            torch.save(pol.state_dict(), out / f"policy-it{it + 1}.pt")
        print(f"iter {it}: lifted {lifted}/{N} (sampled), held {held}, reward {rec['reward']}, std {rec['std']}, {rec['s']} s", flush=True)
    print("TRAIN-DONE", flush=True)
sys.stdout.flush()
os._exit(0)
