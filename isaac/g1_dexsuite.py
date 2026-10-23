"""Closed-loop grasp policy for the fixed-base G1 + Dex3 (owner 2026-10-08: adapt NVIDIA's example, option 1).

Isaac Lab's Dexsuite lift task (isaaclab_tasks/manager_based/manipulation/dexsuite, built on DextrAH and DexPBT for a
KUKA arm + Allegro hand) with our robot and table: a PPO policy (rsl_rl) moves the waist, right arm and right hand
joints every 20 ms to carry the object to a commanded point above the table. Kept from Dexsuite: rewards (reach, finger
contacts, tracking, success), observation groups with 5-step history, domain randomization, the adaptive curriculum
that phases gravity in. Ours: the G1 (real G1 torques, MuJoCo-matched hand drives, self-collision on, camera-mount
collider off, left arm parked), sim dt 5 ms as validated, table and shelf as isaac/g1_grasp_rl.py, objects drawn from
our primitive distribution (training seeds 800000+, one per env), friction 0.5-1.0 ("max" combine), and the same
perception error as v8 (object centre off by 0.4 cm per axis, size x 1 +- 5 %) on the object points.

Stage 1, a TEACHER: it sees the object's exact orientation and 64 points over its whole surface every step (simulator
data, with the perception error above). A camera-only student is stage 2.

Fix to upstream: the curriculum's point-cloud noise had max = min = -0.01 (a constant 1 cm shift); here +-0.01.

  python isaac/g1_dexsuite.py train --num_envs=2048 --iters=5000 --out=runs/isaac/dex-v1 [--resume]
  python isaac/g1_dexsuite.py eval --checkpoint=runs/isaac/dex-v1/model_4999.pt [--objects=household] [--n=60]
  python isaac/g1_dexsuite.py diag --num_envs=512 [--checkpoint=a.pt]   # one training episode: zero, random, policy

--base=walking (owner 2026-10-15: "try tuning v6b, see if it can learn"): the same task on the walking G1 (Isaac Lab's
locomanipulation G1, Nucleus g1.usd, free base, gravity on) standing on a floor in front of a slab table (pedestal set
back so the feet fit), legs driven by NVIDIA's AGILE walking policy with a stored command (zero speed, hip height 0.72).
Waist, right arm and hand drives as the fixed-base robot (so v6b's checkpoint loads and acts on the same inputs); the
policy's observations and actions are unchanged. Floor height from the walk runs' settled pelvis (0.721 m). A fall
(pelvis below 0.5 m above the floor) ends the episode. Robot gravity is on, so --gravity=full is forced.

eval scores on g1_grasp_rl.py's benchmarks (same layouts and seeds: test 900000+, validation 700000+, household
950000+, settings as its defaults): the target is the object's start + 10 cm; after 6 s, lifted = up >= 9 cm and
within 5 cm of the pinch point; then the target shakes 3 cm at 2 Hz along x, y, z and kept = still in hand, 5 cm up.
(The shake moves the policy's target, not a scripted hand path as in g1_grasp_rl.py.)
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

if "--base=walking" in sys.argv:
    import pinocchio  # noqa: F401  (before AppLauncher: Isaac Lab's pinocchio; the locomanipulation package imports pink)

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("mode", choices=["train", "eval", "diag"])
p.add_argument("--num_envs", type=int, default=2048)
p.add_argument("--iters", type=int, default=5000)
p.add_argument("--out", default="runs/isaac/dex-v1")
p.add_argument("--resume", action="store_true", help="train: continue from the newest model_*.pt in --out")
p.add_argument("--checkpoint", default="", help="eval: one or more comma-separated model files")
p.add_argument("--objects", choices=["primitives", "household", "pool"], default="primitives",
               help="pool: so101/assets/objects/pool_g1.py's one-hand objects (train: its train split mixed with primitives; "
                    "eval: --split)")
p.add_argument("--pool", default=os.path.expandvars("/nfs/hpc/share/$USER/datasets/pool_g1"), help="pool_g1.py output folder")
p.add_argument("--split", choices=["train", "val", "test"], default="test", help="eval --objects=pool: which split")
p.add_argument("--prim_frac", type=float, default=0.3, help="train --objects=pool: share of envs with a primitive")
p.add_argument("--n", type=int, default=60, help="eval: layouts per shape (household: poses per object)")
p.add_argument("--seed", type=int, default=0, help="eval: layout seed base (default 900000; household 950000)")
p.add_argument("--noise", type=float, default=0.4, help="perception error, cm std per axis (sizes +- 5%%)")
p.add_argument("--settle", type=float, default=1.0, help="eval: seconds the objects settle before the start state")
p.add_argument("--tag", default="")
p.add_argument("--reset", choices=["rest", "float"], default="rest",
               help="train: objects reset resting on the table (as the benchmark) or floating 7-9 cm up at random angles (dex-v1)")
p.add_argument("--park", default="0.0,0.6,1.9", help="left shoulder pitch, roll, elbow (rad) of the parked left arm")
p.add_argument("--target_z", default="0.10,0.30",
               help="train: commanded height range above the table, m (dex-v1/v2: 0.05,0.22, within reach of the resting object)")
p.add_argument("--success_gate", type=int, default=0,
               help="train: 1 = success reward only while thumb + finger touch the object (no reward for leaving it alone)")
p.add_argument("--points", choices=["pads", "links"], default="pads",
               help="hand points for the reach reward and observations: finger pads (1 cm in from the tip ends) or link "
                    "origins (dex-v1 to v3: the finger links' origins sit 5.2 cm back from their tips)")
p.add_argument("--gravity", choices=["curriculum", "full"], default="curriculum",
               help="train: Dexsuite's curriculum (no gravity until the policy succeeds) or full gravity throughout")
p.add_argument("--hand", choices=["joints", "pinch"], default="joints",
               help="hand actions: 7 joint deltas (Dexsuite) or one pinch-closing value driving g1.hand_q (thumb to index pad)")
p.add_argument("--pad_reward", type=float, default=0.0,
               help="train: weight of an extra reach reward on the thumb and index pads alone (1 - tanh(farther pad / 5 cm))")
p.add_argument("--contact_n", type=float, default=1.0, help="thumb + finger contact threshold, N (Dexsuite: 1)")
p.add_argument("--rate_w", type=float, default=0.005, help="train: action-change penalty weight (Dexsuite: 0.005)")
p.add_argument("--acc_w", type=float, default=0.0, help="train: arm joint acceleration penalty weight (sum of squares, rad/s^2)")
p.add_argument("--jerk_w", type=float, default=0.0,
               help="train: arm joint jerk penalty weight (sum of squares, rad/s^3, from joint velocities per policy step)")
p.add_argument("--posture_w", type=float, default=0.0,
               help="train: penalty weight on --posture_joints' squared distance from g1.READY (odd wrist and forearm poses)")
p.add_argument("--posture_joints", default="right_elbow_joint,right_wrist_roll_joint,right_wrist_pitch_joint,right_wrist_yaw_joint")
p.add_argument("--ema", type=float, default=1.0,
               help="arm actions low-pass filtered: a_f = ema * a + (1 - ema) * previous a_f (1 = no filter); train and eval")
p.add_argument("--lcp", type=float, default=0.0,
               help="train: Lipschitz-constrained policy gradient penalty, lcp * |d log pi(a|s) / d s|^2 on normalized "
                    "observations (Chen et al. 2024 use 0.002), one extra optimizer pass per update")
p.add_argument("--record", action="store_true",
               help="eval: also save every layout's motion (joints, object pose per step) for isaac/render_mj.py")
p.add_argument("--parks", default="", help="diag: left-arm poses to measure, p,r,e/p,r,e")
p.add_argument("--unitree", default="/opt/dlami/nvme/w/unitree_sim_isaaclab")
p.add_argument("--base", choices=["fixed", "walking"], default="fixed", help="see the docstring")
p.add_argument("--self_collide", type=int, default=1, help="walking base: robot self-collisions")
p.add_argument("--obj_iters", type=int, default=32, help="object solver position iterations (PhysX uses the island's max)")
p.add_argument("--steps_per_env", type=int, default=32, help="PPO rollout length per iteration")
p.add_argument("--root_jitter",default="0.03,0.1", help="walking base, train: root reset offset, m in x and y, rad in yaw")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = True
WALK = args.base == "walking"
if WALK:
    args.gravity = "full"  # Dexsuite's gravity curriculum changes world gravity: the standing robot would float
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
from isaaclab.assets import RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.envs import mdp as base_mdp  # noqa: E402
from isaaclab.envs.mdp.actions.joint_actions import RelativeJointPositionAction  # noqa: E402
from isaaclab.managers import ActionTerm, ActionTermCfg, ManagerTermBase  # noqa: E402
from isaaclab.managers import EventTermCfg as EventTerm  # noqa: E402
from isaaclab.managers import ObservationTermCfg as ObsTerm  # noqa: E402
from isaaclab.managers import RewardTermCfg as RewTerm  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.managers import TerminationTermCfg as DoneTerm  # noqa: E402
from isaaclab.sensors import ContactSensorCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import combine_frame_transforms, quat_apply  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.manager_based.manipulation.dexsuite import dexsuite_env_cfg as dexsuite  # noqa: E402
from isaaclab_tasks.manager_based.manipulation.dexsuite import mdp  # noqa: E402
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, Vt  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

if WALK:
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401  (blacklisted by default)
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sim.spawners.from_files import GroundPlaneCfg
    from isaaclab_assets.robots.unitree import G1_29DOF_CFG
    from isaaclab_tasks.manager_based.locomanipulation.pick_place.configs.action_cfg import AgileBasedLowerBodyActionCfg
    from isaaclab_tasks.manager_based.locomanipulation.pick_place.configs.agile_locomotion_observation_cfg import (
        AgileTeacherPolicyObservationsCfg)
    from isaaclab_tasks.manager_based.locomanipulation.pick_place.mdp.actions import AgileBasedLowerBodyAction
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["PROJECT_ROOT"] = args.unitree
sys.path.insert(0, args.unitree)
from robots.unitree import G129_CFG_WITH_DEX3_BASE_FIX  # noqa: E402
from so101 import g1  # noqa: E402
from so101.objects import layout, random_objects  # noqa: E402

DT, DECIM = 0.005, 4  # 5 ms physics (the hand drives were matched to MuJoCo at this rate), 50 Hz policy
SHAPES = ["box", "cylinder", "lying cylinder", "ball", "capsule", "egg"]
ARM, HAND = list(g1.ARM), list(g1.HAND)
JOINTS = ARM + HAND  # the 16 the policy moves: waist yaw, pitch, right arm (7), right hand (7)
TIPS = ["right_hand_thumb_2_link", "right_hand_index_1_link", "right_hand_middle_1_link"]
HAND_BODIES = ["right_hand_palm_link"] + TIPS
# finger pads in their link frames: 1 cm in from the tip ends (g1.TIPS: tip ends 5.2 cm from the link origins; the middle
# finger's links are the index finger's)
PADS = [(0.0, 0.0, 0.0), (0.0, 0.042, 0.0), (0.042, 0.0, 0.0), (0.042, 0.0, 0.0)] if args.points == "pads" else [(0.0, 0.0, 0.0)] * 4
FRICTION = (0.5, 1.0)  # object friction range (g1_grasp_rl.py default), "max" combine against the hand's 0.5
G1_TORQUE = {"waist_yaw_joint": 88.0, "waist_roll_joint": 50.0, "waist_pitch_joint": 50.0, ".*_shoulder_.*_joint": 25.0,
             ".*_elbow_joint": 25.0, ".*_wrist_roll_joint": 25.0, ".*_wrist_(pitch|yaw)_joint": 5.0}  # Menagerie g1
TABLE_POS = (g1.PELVIS[0] + g1.TABLE_EDGE + 0.4, 0.0, -0.5)
ROOT = np.asarray(g1.PELVIS, float)  # robot root (pelvis) in the env frame; commands are in this frame
LIFT = 0.10
STAND_H = 0.721  # walking base: settled pelvis height above the floor under AGILE at hip height 0.72 (walk runs)
FLOOR_Z = ROOT[2] - STAND_H  # the floor in this env's frame (table top at z = 0)
PARK = [float(v) for v in args.park.split(",")]
OBJ_DIR = Path(__file__).resolve().parent.parent / "so101" / "assets" / "objects"
EVAL = args.mode == "eval"
HOUSE = args.objects == "household"
POOL = args.objects == "pool"
assert not (HOUSE and not EVAL), "the household objects are a test set only"
POOL_INDEX = [e for e in json.loads((Path(args.pool) / "pool_g1_index.json").read_text())["objects"]
              if e["kept"] and e["hands"] == 1] if POOL else []


def hull_spec(e, rng):
    return dict(shape="pool" if POOL else "household", type="hull", name=e["name"], entry=e, mass=e["mass_kg"],
                friction=float(rng.uniform(0.7, 1.5)), foot=0.5 * float(np.hypot(e["width_m"], e["length_m"])),
                size=[e["width_m"] / 2, e["length_m"] / 2, e["height_m"] / 2], z=e["height_m"] / 2, quat_axis=None)


# --- objects: one per env ---------------------------------------------------------------------------------------------
def make_objects(n_envs):
    """Per env: the object spec and (eval) its layout pose. Train: our primitive distribution, shapes in turn, seeds
    800000+. Eval: g1_grasp_rl.py's layouts (same seeds, same order)."""
    objs = []
    if not EVAL and POOL:  # the pool's train split, each object in turn, and --prim_frac of envs with primitives
        train = sorted((e for e in POOL_INDEX if e["split"] == "train"), key=lambda e: e["name"])
        k = 0
        for i in range(n_envs):
            rng = np.random.default_rng(800000 + i)
            if rng.uniform() < args.prim_frac:
                s = SHAPES[i % len(SHAPES)]
                objs.append(dict(spec=random_objects(rng, 1, [s])[0], seed=800000 + i, shape=s))
            else:
                e = train[k % len(train)]
                k += 1
                objs.append(dict(spec=hull_spec(e, rng), seed=800000 + i, shape="pool", name=e["name"]))
        return objs
    if not EVAL:
        for i in range(n_envs):
            rng = np.random.default_rng(800000 + i)
            objs.append(dict(spec=random_objects(rng, 1, [SHAPES[i % len(SHAPES)]])[0], seed=800000 + i, shape=SHAPES[i % 6]))
        return objs
    seed = args.seed or (950000 if HOUSE else {"test": 970000, "val": 710000, "train": 810000}[args.split] if POOL else 900000)
    if POOL:  # every object of the split, --n poses each
        split = sorted((e for e in POOL_INDEX if e["split"] == args.split), key=lambda e: e["name"])
        for j, e in enumerate(split):
            for i in range(args.n):
                rng = np.random.default_rng([seed, j, i])
                spec = hull_spec(e, rng)
                x, y, a = layout(rng, [spec])[0]
                objs.append(dict(spec=spec, x=x, y=y, heading=a, seed=seed + 100 * j + i, shape="pool", name=e["name"]))
    elif HOUSE:
        index = [e for e in json.loads((OBJ_DIR / "gso_g1_index.json").read_text())["objects"] if e["kept"]]
        for j, e in enumerate(index):
            for i in range(args.n):
                rng = np.random.default_rng([seed, j, i])
                spec = dict(shape="household", type="hull", name=e["name"], entry=e, mass=e["mass_kg"],
                            friction=float(rng.uniform(0.7, 1.5)), foot=0.5 * float(np.hypot(e["width_m"], e["length_m"])),
                            size=[e["width_m"] / 2, e["length_m"] / 2, e["height_m"] / 2], z=e["height_m"] / 2, quat_axis=None)
                x, y, a = layout(rng, [spec])[0]
                objs.append(dict(spec=spec, x=x, y=y, heading=a, seed=seed + 100 * j + i, shape="household", name=e["name"]))
    else:
        for s in SHAPES:
            for i in range(args.n):
                rng = np.random.default_rng(seed + i)
                spec = random_objects(rng, 1, [s])[0]
                x, y, a = layout(rng, [spec])[0]
                objs.append(dict(spec=spec, x=x, y=y, heading=a, seed=seed + i, shape=s))
    return objs


OBJECTS = make_objects(args.num_envs)
N_ENVS = len(OBJECTS) if EVAL else args.num_envs
LIE = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])
HULLS = np.load(Path(args.pool) / "hulls.npz") if POOL else np.load(OBJ_DIR / "gso_g1" / "hulls.npz") if HOUSE else None
# training reset: root height resting on the table, and (hulls) the footprint centre's offset from the root
REST_Z = [-o["spec"]["entry"]["bottom_z_m"] + 0.002 if o["spec"]["type"] == "hull" else float(o["spec"]["z"]) + 0.001 for o in OBJECTS]
CENTRE_XY = [o["spec"]["entry"]["centre_xy_m"] if o["spec"]["type"] == "hull" else [0.0, 0.0] for o in OBJECTS]
LYING = [o["spec"].get("quat_axis") == "y" for o in OBJECTS]


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def spawn_pose(o):
    """Env-frame position and orientation (w, x, y, z) of an object's root (training: a placeholder; resets move it)."""
    s, a = o["spec"], o.get("heading", 0.0)
    qz = (np.cos(a / 2), 0.0, 0.0, np.sin(a / 2))
    if s["type"] == "hull":
        e, (cx, cy) = s["entry"], s["entry"]["centre_xy_m"]
        return ((o.get("x", 0.17) - (np.cos(a) * cx - np.sin(a) * cy), o.get("y", 0.0) - (np.sin(a) * cx + np.cos(a) * cy),
                 -e["bottom_z_m"] + 0.002 + (0.0 if EVAL else 0.05)), qz)
    q = _qmul(qz, LIE) if s["quat_axis"] == "y" else qz
    return (o.get("x", 0.17), o.get("y", 0.0), s["z"] + 0.001 + (0.0 if EVAL else 0.05)), tuple(float(v) for v in q)


def spawn_objects(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawner for {ENV_REGEX_NS}/Object: each env gets its own object (OBJECTS), built as isaac/g1_grasp_rl.py builds
    it (primitives; eggs with a convex-hull collider; household objects from their convex parts)."""
    stage = sim_utils.get_current_stage()
    root, name = prim_path.rsplit("/", 1)
    envs = sorted(sim_utils.find_matching_prim_paths(root), key=lambda s: int(s.rsplit("_", 1)[1]))
    rigid = sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=args.obj_iters, solver_velocity_iteration_count=1,
                                             max_depenetration_velocity=1.0)
    coll = sim_utils.CollisionPropertiesCfg(torsional_patch_radius=0.01, min_torsional_patch_radius=0.005)
    first = None
    for path_env, o in zip(envs, OBJECTS):
        i = int(path_env.rsplit("_", 1)[1])
        assert OBJECTS[i] is o
        s, path = o["spec"], f"{path_env}/{name}"
        fr = FRICTION[0] + (s["friction"] - 0.7) / 0.8 * (FRICTION[1] - FRICTION[0])
        mat = sim_utils.RigidBodyMaterialCfg(static_friction=fr, dynamic_friction=fr, restitution=0.0, friction_combine_mode="max")
        pos, quat = spawn_pose(o)
        if s["type"] == "hull":
            prim = spawn_hull(path, o, pos, quat, rigid, coll, mat)
        else:
            z = s["size"]
            common = dict(rigid_props=rigid, mass_props=sim_utils.MassPropertiesCfg(mass=s["mass"]), collision_props=coll,
                          physics_material=mat, activate_contact_sensors=True)
            ocfg = {"box": lambda: sim_utils.CuboidCfg(size=(2 * z[0], 2 * z[1], 2 * z[2]), **common),
                    "cylinder": lambda: sim_utils.CylinderCfg(radius=z[0], height=2 * z[1], axis="Z", **common),
                    "capsule": lambda: sim_utils.CapsuleCfg(radius=z[0], height=2 * z[1], axis="Z", **common),
                    "sphere": lambda: sim_utils.SphereCfg(radius=z[0], **common),
                    "ellipsoid": lambda: sim_utils.MeshSphereCfg(radius=1.0, **common)}[s["type"]]()
            prim = ocfg.func(path, ocfg, translation=pos, orientation=quat)
            if s["type"] == "ellipsoid":
                prim.GetAttribute("xformOp:scale").Set(type(prim.GetAttribute("xformOp:scale").Get())(*z))
                for c in Usd.PrimRange(prim):
                    if c.HasAPI(UsdPhysics.MeshCollisionAPI):  # Isaac Lab's default for sphere meshes is a bounding sphere
                        UsdPhysics.MeshCollisionAPI(c).GetApproximationAttr().Set("convexHull")
        first = first or prim
    import carb
    carb.settings.get_settings().set_bool("/isaaclab/spawn/multi_assets", True)
    return first


def spawn_hull(path, o, pos, quat, rigid, coll, mat):
    """A scanned household object from its convex parts (as isaac/g1_grasp_rl.py spawn_household)."""
    stage = sim_utils.get_current_stage()
    e = o["spec"]["entry"]
    prim = sim_utils.create_prim(path, "Xform", translation=pos, orientation=quat)
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
        sim_utils.define_collision_properties(f"{path}/part_{k}", coll)
    sim_utils.define_rigid_body_properties(path, rigid)
    sim_utils.define_mass_properties(path, sim_utils.MassPropertiesCfg(mass=o["spec"]["mass"]))
    sim_utils.spawn_rigid_body_material(f"{path}/material", mat)
    sim_utils.bind_physics_material(path, f"{path}/material")
    sim_utils.activate_contact_sensors(path)
    return prim


@configclass
class ObjectsSpawnerCfg(sim_utils.RigidObjectSpawnerCfg):
    func = spawn_objects


# --- robot ---------------------------------------------------------------------------------------------------------------
def robot_cfg():
    open_q = dict(zip(HAND, [g1.THUMB_0, *g1.THUMB[-1, 1:], *g1.INDEX, *g1.MIDDLE]))
    park = dict(zip(["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_elbow_joint"], PARK))
    start_q = {k: float(v) for k, v in (dict(zip(ARM, g1.READY.tolist())) | park | open_q).items()}
    if WALK:
        return walking_robot_cfg(start_q)
    cfg = G129_CFG_WITH_DEX3_BASE_FIX.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.spawn.activate_contact_sensors = True
    cfg.spawn.rigid_props.disable_gravity = True  # stands in for the joint controllers' gravity feed-forward
    cfg.spawn.articulation_props.enabled_self_collisions = True
    cfg.init_state = cfg.init_state.replace(pos=tuple(ROOT.tolist()), joint_pos=cfg.init_state.joint_pos | start_q)
    cfg.actuators["waist"] = cfg.actuators["waist"].replace(
        velocity_limit=None, velocity_limit_sim=None, effort_limit=None, stiffness=3000.0, damping=60.0,
        effort_limit_sim={k: v for k, v in G1_TORQUE.items() if k.startswith("waist")})
    cfg.actuators["arms"] = cfg.actuators["arms"].replace(
        effort_limit=None, effort_limit_sim={k: v for k, v in G1_TORQUE.items() if not k.startswith("waist")})
    cfg.actuators["hands"] = cfg.actuators["hands"].replace(
        effort_limit=None, effort_limit_sim={".*_hand_thumb_0_joint": 2.45, ".*_hand_(thumb_[12]|index_.|middle_.)_joint": g1.THUMB_FORCE},
        stiffness=500.0, damping=4.5, armature=0.01)
    return cfg


def walking_robot_cfg(start_q):
    """The locomanipulation G1 (legs as AGILE expects), upper body drives as robot_cfg's fixed-base robot."""
    from isaaclab.actuators import ImplicitActuatorCfg
    cfg = G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.spawn.activate_contact_sensors = True
    cfg.spawn.articulation_props.enabled_self_collisions = bool(args.self_collide)
    cfg.init_state = cfg.init_state.replace(pos=(float(ROOT[0]), float(ROOT[1]), FLOOR_Z + 0.75), rot=(1.0, 0.0, 0.0, 0.0),
                                            joint_pos=dict(cfg.init_state.joint_pos) | start_q)
    acts = dict(cfg.actuators)
    acts["waist"] = ImplicitActuatorCfg(joint_names_expr=["waist_.*_joint"], stiffness=3000.0, damping=60.0, armature=0.001,
                                        effort_limit_sim={k: v for k, v in G1_TORQUE.items() if k.startswith("waist")})
    acts["arms"] = ImplicitActuatorCfg(  # unitree_sim_isaaclab's G1 arm gains (what v6b trained with), G1 torques
        joint_names_expr=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint"],
        stiffness={".*_shoulder_.*_joint": 300.0, ".*_elbow_joint": 400.0, ".*_wrist_.*_joint": 400.0},
        damping={".*_shoulder_.*_joint": 3.0, ".*_elbow_joint": 2.5, ".*_wrist_.*_joint": 2.5},
        effort_limit_sim={k: v for k, v in G1_TORQUE.items() if not k.startswith("waist")})
    acts["hands"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_index_.*", ".*_middle_.*", ".*_thumb_.*"], stiffness=500.0, damping=4.5, armature=0.01,
        effort_limit_sim={".*_hand_thumb_0_joint": 2.45, ".*_hand_(thumb_[12]|index_.|middle_.)_joint": g1.THUMB_FORCE})
    cfg.actuators = acts
    return cfg


if WALK:
    class StoredCommandLowerBodyAction(AgileBasedLowerBodyAction):
        """AGILE legs with no policy action: the command [vx, vy, wz, hip height] comes from self.command (zero speed,
        hip 0.72 by default; a walking script sets it)."""

        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self.command = torch.tensor([0.0, 0.0, 0.0, 0.72], device=self.device).repeat(self.num_envs, 1)

        @property
        def action_dim(self):
            return 0

        def process_actions(self, actions):
            super().process_actions(self.command)

    @configclass
    class StoredCommandLowerBodyActionCfg(AgileBasedLowerBodyActionCfg):
        class_type: type = StoredCommandLowerBodyAction

    def lower_body_cfg():
        return StoredCommandLowerBodyActionCfg(
            asset_name="robot", joint_names=[".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint"], policy_output_scale=0.25,
            obs_group_name="lower_body_policy",
            policy_path=f"{__import__('isaaclab.utils.assets', fromlist=['x']).ISAACLAB_NUCLEUS_DIR}/Policies/Agile/agile_locomotion.pt")

    def robot_fell(env, minimum_height: float):
        return env.scene["robot"].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] < minimum_height


def drop_camera_colliders(env, env_ids):
    """The hand camera mount's collider fights the palm with self-collision on (LOG 2026-10-07): deactivate it."""
    stage = sim_utils.get_current_stage()
    for i in range(env.num_envs):
        for side in ("left", "right"):
            pr = stage.GetPrimAtPath(f"/World/envs/env_{i}/Robot/{side}_hand_camera_base_link/collisions")
            if pr.IsValid():
                pr.SetActive(False)


# --- observations, rewards ------------------------------------------------------------------------------------------------
class noisy_object_point_cloud_b(mdp.object_point_cloud_b):
    """Dexsuite's object surface points, with v8's perception error: one centre offset (args.noise cm std per axis) and
    one size factor (1 +- 5 %) per env, drawn at every reset (eval: seeded per layout as g1_grasp_rl.py's first look)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.off = torch.zeros(env.num_envs, 3, device=env.device)
        self.scale = torch.ones(env.num_envs, 1, 1, device=env.device)

    def reset(self, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else torch.as_tensor(env_ids, device=self.device)
        if args.noise <= 0 or len(ids) == 0:
            return
        if EVAL:
            for i in ids.tolist():
                rng = np.random.default_rng([OBJECTS[i]["seed"], 13])
                self.off[i] = torch.tensor(rng.normal(0, args.noise / 100, 3), device=self.device)
                self.scale[i] = float(1 + rng.normal(0, 0.05))
        else:
            self.off[ids] = torch.randn(len(ids), 3, device=self.device) * args.noise / 100
            self.scale[ids] = 1 + 0.05 * torch.randn(len(ids), 1, 1, device=self.device)

    def __call__(self, env, ref_asset_cfg=SceneEntityCfg("robot"), object_cfg=SceneEntityCfg("object"), num_points=64,
                 flatten=False, visualize=False):
        pts = super().__call__(env, ref_asset_cfg, object_cfg, num_points, False, False)
        c = pts.mean(1, keepdim=True)
        pts = c + (pts - c) * self.scale + self.off[:, None, :]
        return pts.reshape(env.num_envs, -1) if flatten else pts


def g1_contacts(env, threshold: float):
    """Thumb and index or middle tip touching the object (Dexsuite's rule: thumb + another finger)."""
    f = {n: env.scene.sensors[f"{n}_object_s"].data.force_matrix_w.view(env.num_envs, 3).norm(dim=-1) for n in TIPS}
    return (f[TIPS[0]] > threshold) & ((f[TIPS[1]] > threshold) | (f[TIPS[2]] > threshold))


def g1_position_command_error_tanh(env, std: float, command_name: str, asset_cfg: SceneEntityCfg, align_asset_cfg: SceneEntityCfg):
    """Dexsuite's tracking reward with our contact gate."""
    asset, obj = env.scene[asset_cfg.name], env.scene[align_asset_cfg.name]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, env.command_manager.get_command(command_name)[:, :3])
    return (1 - torch.tanh(torch.norm(obj.data.root_pos_w - des_pos_w, dim=1) / std)) * g1_contacts(env, args.contact_n).float()


def g1_success_gated(env, command_name, asset_cfg, align_asset_cfg, pos_std, rot_std=None):
    """Dexsuite's success reward, paid only while thumb + index or middle touch the object."""
    return mdp.success_reward(env, command_name, asset_cfg, align_asset_cfg, pos_std, rot_std) * g1_contacts(env, args.contact_n).float()


def hand_points_w(env, asset_cfg):
    """World position, orientation and velocity of the palm and the finger pads (HAND_BODIES order)."""
    a = env.scene[asset_cfg.name]
    ids = [a.body_names.index(b) for b in HAND_BODIES]
    q, w = a.data.body_quat_w[:, ids], a.data.body_ang_vel_w[:, ids]
    r = quat_apply(q, torch.tensor(PADS, dtype=torch.float32, device=env.device).expand(env.num_envs, 4, 3))
    return a.data.body_pos_w[:, ids] + r, q, a.data.body_lin_vel_w[:, ids] + torch.cross(w, r, dim=-1), w


def hand_points_state_b(env, body_asset_cfg, base_asset_cfg):
    """Dexsuite's body_state_b for the palm and finger pads: pos, quat, lin vel, ang vel in the robot root frame."""
    from isaaclab.utils.math import quat_apply_inverse, subtract_frame_transforms
    base = env.scene[base_asset_cfg.name]
    p, q, v, w = hand_points_w(env, body_asset_cfg)
    rp = base.data.root_link_pos_w[:, None].expand(-1, 4, -1)
    rq = base.data.root_link_quat_w[:, None].expand(-1, 4, -1)
    pb, qb = subtract_frame_transforms(rp.reshape(-1, 3), rq.reshape(-1, 4), p.reshape(-1, 3), q.reshape(-1, 4))
    vb, wb = quat_apply_inverse(rq.reshape(-1, 4), v.reshape(-1, 3)), quat_apply_inverse(rq.reshape(-1, 4), w.reshape(-1, 3))
    return torch.cat([pb, qb, vb, wb], dim=-1).view(env.num_envs, -1)


def g1_fingers_to_object(env, std, object_cfg=SceneEntityCfg("object"), asset_cfg=SceneEntityCfg("robot")):
    """Dexsuite's reach reward on the palm and finger pads: 1 - tanh(farthest point's distance to the object / std)."""
    p = hand_points_w(env, asset_cfg)[0]
    return 1 - torch.tanh((p - env.scene[object_cfg.name].data.root_pos_w[:, None]).norm(dim=-1).max(dim=-1).values / std)


def g1_pads_to_object(env, std, object_cfg=SceneEntityCfg("object"), asset_cfg=SceneEntityCfg("robot")):
    """Reach reward on the thumb and index pads alone: 1 - tanh(the farther pad's distance to the object / std)."""
    p = hand_points_w(env, asset_cfg)[0][:, 1:3]
    return 1 - torch.tanh((p - env.scene[object_cfg.name].data.root_pos_w[:, None]).norm(dim=-1).max(dim=-1).values / std)


class arm_jerk_l2(ManagerTermBase):
    """Arm joint jerk, sum of squares (rad/s^3), from the joint velocities at each policy step; 0 for an env's first
    two steps after a reset."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.ids = [env.scene["robot"].joint_names.index(j) for j in ARM]
        self.v1 = torch.zeros(env.num_envs, len(self.ids), device=env.device)
        self.a1 = torch.zeros_like(self.v1)
        self.k = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.v1[ids], self.a1[ids], self.k[ids] = 0.0, 0.0, 0

    def __call__(self, env):
        v = env.scene["robot"].data.joint_vel[:, self.ids]
        a = (v - self.v1) / env.step_dt
        j = (a - self.a1) / env.step_dt
        out = j.pow(2).sum(-1) * (self.k >= 2).float()
        self.v1, self.a1 = v.clone(), a
        self.k += 1
        return out


def arm_posture_l2(env, names):
    """Squared distance of the named joints (elbow, wrist) from g1.READY: keeps the forearm and wrist natural."""
    r = env.scene["robot"]
    ids = [r.joint_names.index(n) for n in names]
    ready = torch.tensor([float(g1.READY[ARM.index(n)]) for n in names], device=env.device)
    return (r.data.joint_pos[:, ids] - ready).pow(2).sum(-1)


class PinchAction(ActionTerm):
    """One action for the hand: the pinch's opening value (1 open, 0 closed) moves by scale x action each step, and
    the 7 hand joints follow g1.hand_q (thumb_1/thumb_2 along the solved closing path to the index pad; index and
    middle held). The scripted grasp closes the hand the same way."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._ids, _ = self._asset.find_joints(list(HAND), preserve_order=True)
        self._raw = torch.zeros(self.num_envs, 1, device=self.device)
        self._open = torch.ones(self.num_envs, device=self.device)
        k = torch.tensor(g1.THUMB, dtype=torch.float32, device=self.device)
        self._kx, self._k1, self._k2 = k[:, 0].contiguous(), k[:, 1], k[:, 2]
        self._fixed = torch.tensor([g1.THUMB_0, *g1.INDEX, *g1.MIDDLE], dtype=torch.float32, device=self.device)

    @property
    def action_dim(self):
        return 1

    @property
    def raw_actions(self):
        return self._raw

    @property
    def processed_actions(self):
        return self._open[:, None]

    def process_actions(self, actions):
        self._raw[:] = actions
        self._open = (self._open + self.cfg.scale * actions[:, 0]).clamp(0.0, 1.0)

    def apply_actions(self):
        i = torch.searchsorted(self._kx, self._open).clamp(1, len(self._kx) - 1)
        w = (self._open - self._kx[i - 1]) / (self._kx[i] - self._kx[i - 1])
        t1 = self._k1[i - 1] + w * (self._k1[i] - self._k1[i - 1])
        t2 = self._k2[i - 1] + w * (self._k2[i] - self._k2[i - 1])
        f = self._fixed.expand(self.num_envs, 5)
        q = torch.stack([f[:, 0], t1, t2, f[:, 1], f[:, 2], f[:, 3], f[:, 4]], dim=1)
        self._asset.set_joint_position_target(q, joint_ids=self._ids)

    def reset(self, env_ids=None):
        env_ids = slice(None) if env_ids is None else env_ids
        self._raw[env_ids] = 0.0
        self._open[env_ids] = 1.0


@configclass
class PinchActionCfg(ActionTermCfg):
    class_type: type = PinchAction
    scale: float = 0.1


class EMARelativeJointPositionAction(RelativeJointPositionAction):
    """Relative joint position actions through a first-order low-pass filter (the filtered action is what the
    policy sees as its last action)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._filt = torch.zeros(self.num_envs, self.action_dim, device=self.device)

    def process_actions(self, actions):
        self._filt = self.cfg.alpha * actions + (1.0 - self.cfg.alpha) * self._filt
        super().process_actions(self._filt)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self._filt[slice(None) if env_ids is None else env_ids] = 0.0


@configclass
class EMARelativeJointPositionActionCfg(mdp.RelativeJointPositionActionCfg):
    class_type: type = EMARelativeJointPositionAction
    alpha: float = 1.0


@configclass
class G1Actions:
    action = EMARelativeJointPositionActionCfg(asset_name="robot", joint_names=JOINTS, scale=0.1, alpha=args.ema)


@configclass
class G1PinchActions:
    action = EMARelativeJointPositionActionCfg(asset_name="robot", joint_names=list(ARM), scale=0.1, alpha=args.ema)
    hand = PinchActionCfg(asset_name="robot")


@configclass
class G1Rewards(dexsuite.RewardsCfg):
    good_finger_contact = RewTerm(func=g1_contacts, weight=0.5, params={"threshold": 1.0})


# --- the task --------------------------------------------------------------------------------------------------------------
@configclass
class G1LiftEnvCfg(dexsuite.DexsuiteLiftEnvCfg):
    actions: G1Actions = G1Actions()
    rewards: G1Rewards = G1Rewards()

    def __post_init__(self):
        super().__post_init__()
        sc = self.scene
        sc.num_envs, sc.env_spacing, sc.replicate_physics = N_ENVS, 2.0, False
        sc.robot = robot_cfg()
        sc.object = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Object", spawn=ObjectsSpawnerCfg(),
                                   init_state=RigidObjectCfg.InitialStateCfg(pos=(0.17, 0.0, 0.07)))
        sc.table = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/table",
            spawn=sim_utils.CuboidCfg(size=(0.8, 1.2, 1.0), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                                      collision_props=sim_utils.CollisionPropertiesCfg(),
                                      physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.8)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=TABLE_POS))
        sc.shelf = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/shelf",
            spawn=sim_utils.CuboidCfg(size=(0.08, 0.12, 0.06), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                                      collision_props=sim_utils.CollisionPropertiesCfg()),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.32, 0.0, 0.03)))
        sc.plane = None  # the legs hang below the table top; no floor needed
        sc.sky_light = None  # its texture comes from the Nucleus server
        for n in TIPS:
            setattr(sc, f"{n}_object_s", ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/" + ("right_hand/" if WALK else "") + n,
                                                         filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"]))
        # observations: our 16 joints, palm + finger tips, tip contacts, noisy object points
        obs = self.observations
        obs.proprio.joint_pos.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS)}
        obs.proprio.joint_vel.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS)}
        obs.proprio.hand_tips_state_b.params["body_asset_cfg"] = SceneEntityCfg("robot", body_names=HAND_BODIES)
        obs.proprio.hand_tips_state_b.func = hand_points_state_b
        obs.proprio.contact = ObsTerm(func=mdp.fingers_contact_force_b, params={"contact_sensor_names": [f"{n}_object_s" for n in TIPS]},
                                      clip=(-20.0, 20.0))
        obs.perception.object_point_cloud = ObsTerm(func=noisy_object_point_cloud_b, noise=obs.perception.object_point_cloud.noise,
                                                    clip=(-2.0, 2.0), params={"num_points": 64, "flatten": True, "visualize": False})
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg("robot", body_names=HAND_BODIES)
        self.rewards.fingers_to_object.func = g1_fingers_to_object
        self.rewards.position_tracking.func = g1_position_command_error_tanh
        # commands (robot root frame: pelvis at ROOT, facing +x): --target_z above the table top, over the object area
        r = self.commands.object_pose.ranges
        zlo, zhi = (float(v) for v in args.target_z.split(","))
        r.pos_x, r.pos_y, r.pos_z = (0.20, 0.32), (-0.15, 0.15), (zlo - ROOT[2], zhi - ROOT[2])
        if args.success_gate:
            self.rewards.success.func = g1_success_gated
        self.rewards.good_finger_contact.params["threshold"] = args.contact_n
        if args.pad_reward > 0:
            self.rewards.pads_to_object = RewTerm(func=g1_pads_to_object, weight=args.pad_reward, params={"std": 0.05})
        if args.hand == "pinch":
            self.actions = G1PinchActions()
        if WALK:  # floor, slab table on a set-back pedestal, AGILE legs, fall termination
            sc.table = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/table",
                spawn=sim_utils.CuboidCfg(size=(0.8, 1.2, 0.04), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                                          collision_props=sim_utils.CollisionPropertiesCfg(),
                                          physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.8)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(TABLE_POS[0], 0.0, -0.02)))
            ped_h = -0.04 - FLOOR_Z
            sc.pedestal = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/pedestal",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(TABLE_POS[0] + 0.125, 0.0, FLOOR_Z + ped_h / 2)),
                spawn=sim_utils.CuboidCfg(size=(0.55, 0.6, ped_h), collision_props=sim_utils.CollisionPropertiesCfg()))
            sc.plane = AssetBaseCfg(prim_path="/World/GroundPlane", init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, FLOOR_Z)),
                                    spawn=GroundPlaneCfg())
            sc.env_spacing = 2.5
            self.actions.lower_body_joint_pos = lower_body_cfg()
            self.observations.lower_body_policy = AgileTeacherPolicyObservationsCfg()
            self.terminations.robot_fell = DoneTerm(func=robot_fell, params={"minimum_height": FLOOR_Z + 0.5})
        self.rewards.action_rate_l2.weight = -args.rate_w
        if args.acc_w > 0:
            self.rewards.arm_joint_acc = RewTerm(func=base_mdp.joint_acc_l2, weight=-args.acc_w,
                                                 params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(ARM))})
        if args.jerk_w > 0:
            self.rewards.arm_jerk = RewTerm(func=arm_jerk_l2, weight=-args.jerk_w)
        if args.posture_w > 0:
            self.rewards.arm_posture = RewTerm(func=arm_posture_l2, weight=-args.posture_w,
                                               params={"names": args.posture_joints.split(",")})
        if args.gravity == "full" and not EVAL:
            self.events.variable_gravity = None
            if self.curriculum is not None:
                self.curriculum.gravity_adr = None
        # events: no table moves; resets over our table; our 16 joints only
        ev = self.events
        ev.reset_table = None
        ev.reset_robot_wrist_joint = None
        ev.randomize_object_scale = None  # each env already has its own object size
        ev.robot_physics_material = None  # the hand keeps the scene default (0.5), as in the benchmark
        ev.object_physics_material = None  # objects carry their own 0.5-1.0 draw
        ev.object_scale_mass.params["mass_distribution_params"] = [0.5, 1.5]
        ev.reset_robot_joints.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS),
                                        "position_range": [-0.2, 0.2], "velocity_range": [0.0, 0.0]}
        for k in ("joint_stiffness_and_damping", "joint_friction"):
            getattr(ev, k).params["asset_cfg"] = SceneEntityCfg("robot", joint_names=JOINTS)
        if args.reset == "rest":
            ev.reset_object = EventTerm(func=reset_object_rest, mode="reset")
        else:  # dex-v1: floating 7-9 cm above the table at random angles
            ev.reset_object.params["pose_range"] = {"x": [-0.045, 0.045], "y": [-0.12, 0.12], "z": [0.0, 0.02],
                                                    "roll": [-3.14, 3.14], "pitch": [-3.14, 3.14], "yaw": [-3.14, 3.14]}
        ev.drop_camera_colliders = EventTerm(func=drop_camera_colliders, mode="prestartup")
        if WALK:  # every episode: legs and root back to standing (train: root offset, as where the walk stops)
            jx, jyaw = (float(v) for v in args.root_jitter.split(","))
            jx, jyaw = (0.0, 0.0) if EVAL else (jx, jyaw)
            ev.reset_robot_root = EventTerm(func=base_mdp.reset_root_state_uniform, mode="reset", params={
                "asset_cfg": SceneEntityCfg("robot"), "velocity_range": {},
                "pose_range": {"x": (-jx, jx), "y": (-jx, jx), "yaw": (-jyaw, jyaw)}})
            ev.reset_robot_legs = EventTerm(func=base_mdp.reset_joints_by_offset, mode="reset", params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint"]),
                "position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)})
        self.terminations.object_out_of_bound.params["in_bound_range"] = {"x": (-0.3, 0.8), "y": (-0.6, 0.6), "z": (-0.05, 1.0)}
        if self.curriculum is not None:  # upstream sets the point-cloud noise max to -0.01 (= min): make it +-0.01
            self.curriculum.object_obs_unoise_max_adr.params["modify_params"]["final_value"] = 0.01
        self.episode_length_s = 6.0
        self.sim.dt = DT
        self.decimation = DECIM
        self.sim.render_interval = DECIM
        if EVAL:  # nominal physics, full gravity, no curriculum, objects start where the benchmark puts them
            for k in ("joint_stiffness_and_damping", "joint_friction", "object_scale_mass", "variable_gravity", "reset_object"):
                setattr(ev, k, None)
            ev.reset_object = EventTerm(func=reset_object_to_start, mode="reset")
            ev.reset_robot_joints.params = {"asset_cfg": SceneEntityCfg("robot"), "position_range": [0.0, 0.0],
                                            "velocity_range": [0.0, 0.0]}  # every joint back to its start
            self.curriculum = None
            self.terminations.object_out_of_bound = None
            self.terminations.abnormal_robot = None
            self.rewards.early_termination = None
            self.episode_length_s = 60.0
            self.commands.object_pose.resampling_time_range = (1e6, 1e6)


START = None  # eval: per-env object start state (env frame position, quaternion), set after settling


def reset_object_rest(env, env_ids):
    """Training reset like the benchmark's layouts: each env's object resting on the table in its natural pose (lying
    shapes on their side), random heading, centre in the benchmark's area (x 0.125-0.215, y +-0.12)."""
    obj: RigidObject = env.scene["object"]
    dev, n = env.device, len(env_ids)
    rest_z, lie = (torch.tensor(v, device=dev) for v in (REST_Z, LYING))
    st = obj.data.default_root_state[env_ids].clone()
    st[:, 0] = 0.125 + 0.09 * torch.rand(n, device=dev)
    st[:, 1] = -0.12 + 0.24 * torch.rand(n, device=dev)
    st[:, 2] = rest_z[env_ids] + 0.002
    a = (2 * torch.rand(n, device=dev) - 1) * np.pi
    qz = torch.stack([torch.cos(a / 2), 0 * a, 0 * a, torch.sin(a / 2)], dim=1)
    ql = torch.tensor(LIE, dtype=torch.float32, device=dev).expand(n, 4)
    w1, x1, y1, z1 = qz.unbind(1)
    w2, x2, y2, z2 = ql.unbind(1)
    q_lie = torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], dim=1)
    st[:, 3:7] = torch.where(lie[env_ids, None], q_lie, qz)
    c = torch.tensor(CENTRE_XY, dtype=torch.float32, device=dev)[env_ids]  # hulls: the root sits off the footprint centre
    st[:, 0] -= torch.cos(a) * c[:, 0] - torch.sin(a) * c[:, 1]
    st[:, 1] -= torch.sin(a) * c[:, 0] + torch.cos(a) * c[:, 1]
    st[:, :3] += env.scene.env_origins[env_ids]
    st[:, 7:] = 0.0
    obj.write_root_state_to_sim(st, env_ids)


def reset_object_to_start(env, env_ids):
    obj: RigidObject = env.scene["object"]
    st = obj.data.default_root_state[env_ids].clone()
    if START is not None:
        st[:, :7] = START[env_ids]
    st[:, :3] += env.scene.env_origins[env_ids]
    st[:, 7:] = 0.0
    obj.write_root_state_to_sim(st, env_ids)


@configclass
class G1PPORunnerCfg(RslRlOnPolicyRunnerCfg):  # as Dexsuite's KUKA Allegro runner
    num_steps_per_env = 32
    obs_groups = {"policy": ["policy", "proprio", "perception"], "critic": ["policy", "proprio", "perception"]}
    max_iterations = 5000
    save_interval = 250
    experiment_name = "g1_dexsuite"
    policy = RslRlPpoActorCriticCfg(init_noise_std=1.0, actor_obs_normalization=True, critic_obs_normalization=True,
                                    actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128], activation="elu")
    algorithm = RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.005,
                                     num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive",
                                     gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0)


env_cfg = G1LiftEnvCfg()
env_cfg.sim.device = "cuda:0"
agent_cfg = G1PPORunnerCfg()
agent_cfg.max_iterations = args.iters
agent_cfg.num_steps_per_env = args.steps_per_env
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
env = ManagerBasedRLEnv(cfg=env_cfg)
uenv = env
obj: RigidObject = env.scene["object"]
order = [int(re.search(r"env_(\d+)/", s).group(1)) for s in obj.root_physx_view.prim_paths]
print(f"objects: {len(OBJECTS) if EVAL else N_ENVS} envs; view order is env order: {order == list(range(N_ENVS))}", flush=True)
assert order == list(range(N_ENVS)), "object view order differs from env order"
robot = env.scene["robot"]
print("motor limits, N m: " + ", ".join(f"{j.replace('_joint', '')} {float(robot.root_physx_view.get_dof_max_forces()[0, robot.joint_names.index(j)]):.4g}"
                                         for j in ARM), flush=True)

if args.mode == "diag":  # one training episode (curriculum at its start) per controller: why episodes end
    dev, org = "cuda:0", env.scene.env_origins
    venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    hand_ids = [robot.body_names.index(b) for b in HAND_BODIES]
    try:
        print(f"DIAG gravity {env.sim.physics_sim_view.get_gravity()}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"DIAG gravity unknown ({e})", flush=True)
    shape_of = np.array([o["shape"] for o in OBJECTS])
    if args.hand == "pinch":  # close the pinch for 1 s with the arm still: the thumb pad should reach the index pad
        with torch.inference_mode():
            env.reset()
            na = env.action_manager.total_action_dim
            a = torch.zeros(N_ENVS, na, device=dev)
            a[:, -1] = -1.0
            gaps = []
            for k in range(50):
                venv.step(a)
                if k in (0, 4, 9, 19, 49):
                    hp = hand_points_w(env, SceneEntityCfg("robot"))[0]
                    gaps.append(f"step {k + 1}: {(hp[:, 1] - hp[:, 2]).norm(dim=-1).median():.3f}")
            print("DIAG pinch closing, thumb pad to index pad median: " + ", ".join(gaps), flush=True)
    with torch.inference_mode():
        hp = hand_points_w(env, SceneEntityCfg("robot"))[0]
        ho = robot.data.body_pos_w[:, [robot.body_names.index(b) for b in HAND_BODIES]]
        print("DIAG hand points (median over envs, start pose): " + ", ".join(
            f"{b.split('_')[2]} origin-palm {(ho[:, k] - ho[:, 0]).norm(dim=-1).median():.3f} pad-palm {(hp[:, k] - hp[:, 0]).norm(dim=-1).median():.3f}"
            for k, b in enumerate(HAND_BODIES) if k) + f"; thumb pad to index pad {(hp[:, 1] - hp[:, 2]).norm(dim=-1).median():.3f} m", flush=True)
    left_ids = [i for i, b in enumerate(robot.body_names) if b.startswith(("left_wrist", "left_hand", "left_elbow"))]
    lj = [robot.joint_names.index(j) for j in ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_elbow_joint"]]
    lo_box = torch.tensor([0.125 - 0.06, -0.12 - 0.06, 0.0], device=dev)
    hi_box = torch.tensor([0.215 + 0.06, 0.12 + 0.06, 0.12], device=dev)  # where objects can reach at reset
    for cand in [args.park] + [c for c in args.parks.split("/") if c]:
        with torch.inference_mode():
            env.reset()
            q = robot.data.joint_pos.clone()
            q[:, lj] = torch.tensor([float(v) for v in cand.split(",")], device=dev)
            robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            robot.set_joint_position_target(q)
            for _ in range(10):
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(DT)
            lp = robot.data.body_pos_w[:, left_ids] - org[:, None]
            gap = torch.maximum(lo_box - lp, lp - hi_box).clamp(min=0).norm(dim=-1).min(dim=1).values  # to the object area
            in_table = ((lp[..., 0] > 0.03) & (lp[..., 2] < 0.01)).any(dim=1)
            print(f"DIAG park {cand}: left hand/wrist to object area min {gap.min():.3f} median {gap.median():.3f} m, "
                  f"envs touching it {(gap < 0.01).float().mean():.2f}, inside the table {in_table.float().mean():.2f}; "
                  f"left palm median {lp[:, [robot.body_names[i] for i in left_ids].index('left_hand_palm_link')].median(dim=0).values.cpu().numpy().round(3)}",
                  flush=True)
    for name in ["zero", "random"] + [c for c in args.checkpoint.split(",") if c]:
        if name not in ("zero", "random"):
            runner = OnPolicyRunner(venv, agent_cfg.to_dict(), log_dir=None, device=dev)
            runner.load(name)
            policy = runner.get_inference_policy(device=dev)
        with torch.inference_mode():  # every tensor the env writes stays an inference tensor across runs
            env.reset()
            obs = venv.get_observations()
            n = N_ENVS
            end = torch.full((n,), -1, dtype=torch.long, device=dev)
            kind = torch.zeros(n, dtype=torch.long, device=dev)  # 1 out of bound, 2 abnormal robot, 3 time out
            last_pos, last_vel = obj.data.root_pos_w - org, obj.data.root_lin_vel_w.clone()
            p0 = last_pos.clone()
            near0 = (robot.data.body_pos_w - obj.data.root_pos_w[:, None]).norm(dim=-1)  # every robot body at the start
            q0 = (robot.data.joint_pos - robot.data.default_joint_pos)[:, [robot.joint_names.index(j) for j in JOINTS]]
            v_early = torch.zeros(n, device=dev)
            d0 = None
            d_min = torch.full((n,), 9.0, device=dev)
            touched = torch.zeros(n, dtype=torch.bool, device=dev)
            tip_f = torch.zeros(n, len(TIPS), device=dev)  # peak force on each tip from the object (filtered sensor)
            tip_net = torch.zeros(n, len(TIPS), device=dev)  # peak net contact force on each tip (anything)
            for k in range(int(env_cfg.episode_length_s / (DT * DECIM))):
                na = env.action_manager.total_action_dim
                a = (torch.zeros(n, na, device=dev) if name == "zero" else torch.randn(n, na, device=dev)
                     if name == "random" else policy(obs))
                obs, _, dones, _ = venv.step(a)
                tm = env.termination_manager
                k_now = torch.where(tm.get_term("object_out_of_bound"), 1, torch.where(tm.get_term("abnormal_robot"), 2, 3))
                new = (end < 0) & dones.bool()
                end[new], kind[new] = k, k_now[new]
                live = end < 0
                pos = obj.data.root_pos_w - org
                last_pos[live], last_vel[live] = pos[live], obj.data.root_lin_vel_w[live]
                if k < 5:
                    v_early = torch.maximum(v_early, torch.where(live, obj.data.root_lin_vel_w.norm(dim=1), 0.0))
                d = (robot.data.body_pos_w[:, hand_ids] - obj.data.root_pos_w[:, None]).norm(dim=-1).max(-1).values
                d0 = d.clone() if d0 is None else d0
                d_min = torch.where(live, torch.minimum(d_min, d), d_min)
                touched |= live & g1_contacts(env, 1.0)
                for c, tip in enumerate(TIPS):
                    sd = env.scene.sensors[f"{tip}_object_s"].data
                    fm = sd.force_matrix_w.reshape(n, -1, 3).norm(dim=-1).max(dim=1).values
                    nf = sd.net_forces_w.reshape(n, -1, 3).norm(dim=-1).max(dim=1).values
                    tip_f[:, c] = torch.where(live, torch.maximum(tip_f[:, c], fm), tip_f[:, c])
                    tip_net[:, c] = torch.where(live, torch.maximum(tip_net[:, c], nf), tip_net[:, c])
        kn, lp = kind.cpu().numpy(), last_pos.cpu().numpy()
        oob = kn == 1
        side = np.where(lp[:, 2] > 0.5, "up", np.where(lp[:, 2] < 0.0, "below table", np.where(lp[:, 0] < -0.1, "back",
                        np.where(lp[:, 0] > 0.6, "front", np.where(np.abs(lp[:, 1]) > 0.4, "side", "other")))))
        ve, en = v_early.cpu().numpy(), end.cpu().numpy()
        print(f"DIAG {name}: ends out of bound {oob.mean():.2f}, abnormal {(kn == 2).mean():.2f}, time out or none "
              f"{(kn == 3).mean() + (kn == 0).mean():.2f}; out-of-bound step median {np.median(en[oob]) if oob.any() else -1:.0f}; "
              f"where {dict(zip(*np.unique(side[oob], return_counts=True)))}", flush=True)
        print(f"DIAG {name}: out of bound by shape {{{', '.join(f'{s}: {oob[shape_of == s].mean():.2f}' for s in SHAPES)}}}; "
              f"speed in first 0.1 s median {np.median(ve):.3f} p90 {np.percentile(ve, 90):.3f} m/s, > 0.2 m/s "
              f"{(ve > 0.2).mean():.2f}; last speed before out of bound median "
              f"{np.median(last_vel.norm(dim=1).cpu().numpy()[oob]) if oob.any() else -1:.2f}", flush=True)
        launched = (v_early > 0.2).cpu().numpy()
        nb = near0.min(dim=1)
        names, dist, q = np.array(robot.body_names)[nb.indices.cpu().numpy()], nb.values.cpu().numpy(), q0.abs().cpu().numpy()
        offs = ", ".join(f"{j.removesuffix('_joint')} {q[launched, c].mean() if launched.any() else 0:.2f}/{q[~launched, c].mean():.2f}"
                         for c, j in enumerate(JOINTS[:9]))
        print(f"DIAG {name}: launched in the first 0.1 s {launched.sum()}; nearest robot body (origin) for them "
              f"{dict(zip(*np.unique(names[launched], return_counts=True)))}, distance median "
              f"{np.median(dist[launched]) if launched.any() else -1:.3f} (others {np.median(dist[~launched]):.3f}); "
              f"|reset offset| rad, launched/others: {offs}", flush=True)
        tf, tn = tip_f.cpu().numpy(), tip_net.cpu().numpy()
        print(f"DIAG {name}: tip contacts, share of envs > 0.5 N from the object / from anything, max N: "
              + ", ".join(f"{tip.split('_')[2]} {(tf[:, c] > 0.5).mean():.2f}/{(tn[:, c] > 0.5).mean():.2f} {tf[:, c].max():.1f}"
                          for c, tip in enumerate(TIPS)), flush=True)
        print(f"DIAG {name}: object start z median {np.median(p0[:, 2].cpu().numpy()):.3f} x {np.median(p0[:, 0].cpu().numpy()):.3f}; "
              f"farthest hand body to object at start median {d0.median():.3f}, closest over episode median {d_min.median():.3f}; "
              f"thumb + finger contact {touched.float().mean():.2f}", flush=True)
    sys.stdout.flush()
    os._exit(0)

if not EVAL:
    venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(venv, agent_cfg.to_dict(), log_dir=str(out), device="cuda:0")
    if args.lcp > 0:
        import types
        from rsl_rl.algorithms import PPO

        def lcp_update(alg):
            """Lipschitz-constrained policy (Chen et al. 2024): before PPO's own update, one pass over the rollout
            minimizing lcp * |grad_s log pi(a|s)|^2 (s = normalized policy observation)."""
            pol, total, k = alg.policy, 0.0, 0
            for batch in alg.storage.mini_batch_generator(alg.num_mini_batches, 1):
                obs_b, act_b = batch[0], batch[1]
                x = pol.actor_obs_normalizer(pol.get_actor_obs(obs_b)).detach().requires_grad_(True)
                pol._update_distribution(x)
                g = torch.autograd.grad(pol.get_actions_log_prob(act_b).sum(), x, create_graph=True)[0]
                gp = g.pow(2).sum(dim=-1).mean()
                alg.optimizer.zero_grad()
                (args.lcp * gp).backward()
                torch.nn.utils.clip_grad_norm_(pol.parameters(), alg.max_grad_norm)
                alg.optimizer.step()
                total, k = total + float(gp), k + 1
            out = PPO.update(alg)
            out["lcp_grad_sq"] = total / max(k, 1)
            return out
        runner.alg.update = types.MethodType(lcp_update, runner.alg)
    ckpts = sorted(out.glob("model_*.pt"), key=lambda f: int(f.stem.split("_")[1]))
    if args.resume and ckpts:
        print(f"resuming from {ckpts[-1]}", flush=True)
        runner.load(str(ckpts[-1]))
    t0 = time.time()
    runner.learn(num_learning_iterations=args.iters, init_at_random_ep_len=True)
    print(f"TRAIN-DONE {time.time() - t0:.0f} s", flush=True)
    sys.stdout.flush()
    os._exit(0)

# --- eval on the benchmark layouts ----------------------------------------------------------------------------------------
sim = env.sim
for _ in range(int(args.settle / DT)):  # objects settle (robot holds its start pose), then that is every start state
    env.scene.write_data_to_sim()
    sim.step(render=False)
    env.scene.update(DT)
st = obj.data.root_state_w.clone()
st[:, :3] -= env.scene.env_origins
START = st[:, :7].clone()
venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)  # resets every env: objects to START
z0 = obj.data.root_pos_w[:, 2].clone()
hb = robot.body_names.index(g1.HAND_BODY)
pinch_local = torch.tensor(g1.PINCH, dtype=torch.float32, device="cuda:0")
cmd = env.command_manager.get_term("object_pose")
target_b = START[:, :3] - torch.tensor(ROOT, dtype=torch.float32, device="cuda:0") + torch.tensor([0.0, 0.0, LIFT], device="cuda:0")


def measure():
    op = obj.data.root_pos_w
    pw = robot.data.body_pos_w[:, hb] + quat_apply(robot.data.body_quat_w[:, hb], pinch_local.expand(N_ENVS, 3))
    return op[:, 2] - z0, torch.linalg.norm(op - pw, dim=1)


for path in args.checkpoint.split(","):
    runner = OnPolicyRunner(venv, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
    runner.load(path)
    policy = runner.get_inference_policy(device="cuda:0")
    with torch.inference_mode():  # the previous checkpoint's run left inference tensors in the env
        env.reset()
        cmd.pose_command_b[:, :3] = target_b
        obs = venv.get_observations()
    fell = torch.zeros(N_ENVS, dtype=torch.bool, device="cuda:0")
    robot_down = torch.zeros(N_ENVS, dtype=torch.bool, device="cuda:0")
    max_rise = torch.zeros(N_ENVS, device="cuda:0")
    rec_q, rec_o = [], []
    arm_ids = [robot.joint_names.index(j) for j in ARM]
    sm_q, sm_t = [], []  # arm joint positions and targets per step (lift phase), for the smoothness metrics

    def rec():
        if args.record:
            rec_q.append(robot.data.joint_pos.cpu().numpy().astype(np.float32))
            o = obj.data.root_state_w[:, :7].clone()
            o[:, :3] -= env.scene.env_origins
            rec_o.append(o.cpu().numpy().astype(np.float32))
    with torch.inference_mode():
        for k in range(int(6.0 / (DT * DECIM))):
            obs, _, _, _ = venv.step(policy(obs))
            cmd.pose_command_b[:, :3] = target_b
            rec()
            sm_q.append(robot.data.joint_pos[:, arm_ids].clone())
            sm_t.append(robot.data.joint_pos_target[:, arm_ids].clone())
            r, _ = measure()
            max_rise = torch.maximum(max_rise, r)
            fell |= (obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]) < -0.05
            if WALK:
                robot_down |= (robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]) < FLOOR_Z + 0.5
        rise, near = measure()
        ok, held = (rise > LIFT - 0.01) & (near < 0.05), (rise > 0.05) & (near < 0.05)
        n = int(0.5 / (DT * DECIM))
        for ax in range(4):  # shake the target: 3 cm at 2 Hz along x, y, z (one cycle each), then hold 0.5 s
            for s in range(n):
                off = torch.zeros(3, device="cuda:0")
                if ax < 3:
                    off[ax] = 0.03 * np.sin(2 * np.pi * 2.0 * s * DT * DECIM)
                cmd.pose_command_b[:, :3] = target_b + off
                obs, _, _, _ = venv.step(policy(obs))
                rec()
        r2, n2 = measure()
        kept = ok & (r2 > 0.05) & (n2 < 0.05)
    sdt = DT * DECIM
    Q, Tg = torch.stack(sm_q), torch.stack(sm_t)  # (steps, envs, 9 arm joints)
    rms = lambda x: x.pow(2).mean(dim=(0, 2)).sqrt()  # per env, over time and joints  # noqa: E731
    vel, acc = torch.diff(Q, dim=0) / sdt, torch.diff(Q, n=2, dim=0) / sdt ** 2
    jerk = torch.diff(Q, n=3, dim=0) / sdt ** 3
    step = torch.diff(Tg, dim=0)  # target change per policy step, rad
    X = torch.fft.rfft(Tg - Tg.mean(dim=0), dim=0).abs() * 2 / Tg.shape[0]  # amplitude per frequency, (freqs, envs, joints)
    f = torch.fft.rfftfreq(Tg.shape[0], d=sdt, device=X.device)
    sm = (2 / (len(f) * (1 / sdt)) * (X * f[:, None, None]).sum(dim=0)).mean(dim=1)  # CAPS Sm per env (mean over joints)
    smooth = {k: round(float(v.median()), 4) for k, v in dict(arm_vel_rms=rms(vel), arm_acc_rms=rms(acc), arm_jerk_rms=rms(jerk),
                                                               arm_target_step_rms=rms(step), arm_sm=sm).items()}
    print(f"SMOOTH {Path(path).parent.name}-{Path(path).stem}: medians over envs, arm joints, lift phase: "
          + ", ".join(f"{k} {v}" for k, v in smooth.items()), flush=True)
    shapes = SHAPES + ["household", "pool"]
    per = {s: int(ok[[i for i, o in enumerate(OBJECTS) if o["shape"] == s]].sum()) for s in shapes
           if any(o["shape"] == s for o in OBJECTS)}
    why = ["kept" if kept[i] else "dropped in shake" if ok[i] else "fell off table" if fell[i] else
           "slipped" if max_rise[i] > 0.02 else "not lifted" for i in range(N_ENVS)]
    tag = (Path(path).parent.name + "-" + Path(path).stem + ("-walkbase" if WALK else "") + ("-household" if HOUSE else f"-pool-{args.split}" if POOL else "")
           + (f"-{args.tag}" if args.tag else ""))
    seed = OBJECTS[0]["seed"] if POOL else args.seed or (950000 if HOUSE else 900000)
    res = dict(seed=seed, n=args.n, policy=path, objects=args.objects, noise_cm=args.noise, lifted=int(ok.sum()), held=int(held.sum()),
               kept=int(kept.sum()), of=N_ENVS, per_shape=per, smooth=smooth, why={c: why.count(c) for c in sorted(set(why))},
               rows=[dict(shape=o["shape"], name=o.get("name"), seed=o["seed"], success=bool(ok[i]), held=bool(held[i]),
                          kept=bool(kept[i]), why=why[i], rise=round(float(rise[i]), 4), near=round(float(near[i]), 4),
                          max_rise=round(float(max_rise[i]), 4)) for i, o in enumerate(OBJECTS)])
    (out / f"eval-{tag}-s{seed}-n{args.n}.json").write_text(json.dumps(res, indent=1))
    print(f"EVAL {tag} seeds {seed}+: lifted {res['lifted']}/{N_ENVS}, held {res['held']}/{N_ENVS}; per shape {per}; "
          f"kept {res['kept']}/{N_ENVS} after the shake", flush=True)
    print(f"WHY {tag}: {res['why']}" + (f"; robot fell {int(robot_down.sum())}/{N_ENVS}" if WALK else ""), flush=True)
    if args.record:  # isaac/render_mj.py lift format (as isaac/g1_lift.py --record)
        np.savez_compressed(out / f"rec-{tag}.npz", q=np.stack(rec_q), obj=np.stack(rec_o), joint_names=np.array(robot.joint_names),
                            dt=DT * DECIM, pelvis=g1.PELVIS, table_pos=np.array(TABLE_POS), table_size=np.array([0.8, 1.2, 1.0]),
                            specs=json.dumps([dict(shape=o["shape"], seed=o["seed"], type=o["spec"]["type"],
                                                   size=list(map(float, o["spec"]["size"])),
                                                   rgba=list(map(float, o["spec"].get("rgba", [0.8, 0.3, 0.2, 1.0]))),
                                                   success=bool(ok[i]), held=bool(held[i]), why=why[i])
                                              for i, o in enumerate(OBJECTS)]))
        print(f"recorded {out / f'rec-{tag}.npz'}", flush=True)
sys.stdout.flush()
os._exit(0)
