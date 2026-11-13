"""PPO for the closed-loop grab on the walking G1 (path 3, 2026-11-11): a residual on the open-loop pinch.

Scene, as isaac/walk_grasp.py's at the moment its grasp starts: Isaac Lab's locomanipulation G1 (AGILE legs, standing;
batch IK arms, isaac/batch_ik.py), standing --stand_mean m from the edge of a slab table, hands raised 20 cm, one object
on the table within reach of the right hand. Each episode is one grab: the open-loop pinch (isaac/grab_residual.py
`schedule`: raise, approach, descend, close, lift, hold; 6.7 s) toward the object where it lies after a 0.5 s settle,
with v8's adjustments (isaac/grab_policy.py; --base scripted leaves them out), and the policy's residual on it every
20 ms (pinch offset, hand yaw, finger closing). Zero residual is the open-loop pinch.

What differs from the benchmark, on purpose: the robot does not walk here. It is placed where the walk would stop
(the walk's stop distance, val 2026-11-11: mean 13.1 cm, std 1.3) with a random sideways offset and heading error.
The benchmark runs the trained residual after its own walk (walk_grasp.py --residual).

Reward: per step, the hand near the object before the close, the object rising with the hand on it after; once per
episode at its end, `lifted` as the benchmark judges it (9 cm up, not tipped, within 5 cm of the pinch point, robot
up; balls are exempt from the tilt test here, which the contract's `lifted` is not: LOG 2026-11-11). Episodes end at
6.7 s, or early when the robot falls or the object leaves the table.

Objects (--objects): the six primitive shapes; the split's scanned household objects or the object pool's one-hand
objects (gso, pool: isaac/hull_objects.py, the benchmark's own objects, grasp point, placement and v8 stand-in class);
or mix, for training: --pool_share of the robots hold a pool object of the split, the rest primitives. A scanned
object's class input to the residual is all zeros and its sizes are its half extents, as walk_grasp.py gives them.

  python isaac/grab_rl.py train --num_envs 1024 --iters 300 --out runs/grabrl/v1 --base v8 --policy bench/policies/v8.pt
  python isaac/grab_rl.py train ... --objects mix --pool DIR --resume runs/grabrl/v1/model_250.pt   (iterations go on from 250)
  python isaac/grab_rl.py eval  --num_envs 384 --out runs/grabrl/v1 [--checkpoint runs/grabrl/v1/model_299.pt] --split val
  python isaac/grab_rl.py eval  --out runs/grabrl/v1 --objects pool --pool DIR --n 4 [--checkpoint ...]
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import pinocchio  # noqa: F401  (before AppLauncher: Isaac Lab's pinocchio, not Isaac Sim's)
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("mode", choices=["train", "eval"])
p.add_argument("--num_envs", type=int, default=1024)
p.add_argument("--iters", type=int, default=300)
p.add_argument("--out", required=True)
p.add_argument("--base", default="v8", help="the open-loop grab under the residual: scripted, or a learned grab's name with --policy")
p.add_argument("--policy", default="", help="--base <name>: g1_grasp_rl.py's policy file (v8)")
p.add_argument("--noise", type=float, default=0.4, help="perception error on the open-loop grab's inputs, cm (v8 trained with 0.4)")
p.add_argument("--checkpoint", default="", help="eval: the residual to run (none: zero residual, the open-loop grab alone)")
p.add_argument("--resume", default="", help="train: a model_*.pt to start from")
p.add_argument("--split", choices=["train", "val"], default="train", help="object layout seeds (bench_spec walk_carry bases)")
p.add_argument("--shape", default="box,cylinder,lying-cylinder,ball,capsule,egg")
p.add_argument("--objects", choices=["primitives", "gso", "pool", "mix"], default="primitives",
               help="primitives: the shapes of --shape. gso | pool: the split's scanned household objects or the object pool's one-hand "
                    "objects (isaac/hull_objects.py); eval: --n poses each, the benchmark's layouts. mix (train): --pool_share of the "
                    "robots hold a pool object of the split, the rest primitives")
p.add_argument("--pool", default=os.path.expandvars("/nfs/hpc/share/$USER/datasets/pool_g1"), help="--objects pool | mix: pool_g1.py's output folder")
p.add_argument("--pool_share", type=float, default=0.5, help="--objects mix: the share of the robots that hold a pool object")
p.add_argument("--n", type=int, default=4, help="eval with --objects gso | pool: poses per object (the benchmark's runs: gso 16, pool 4)")
p.add_argument("--stand_mean", type=float, default=0.115,
               help="pelvis to table edge at the reset, m. The robot settles about 1 cm back, so 0.115 gives the 12.4 cm the "
                    "benchmark's robots stand at when their goal is read (val 2026-11-11, 96 robots: median 0.124, p10 0.109, p90 0.140)")
p.add_argument("--stand_std", type=float, default=0.012)
p.add_argument("--side_std", type=float, default=0.01, help="the robot's sideways position error at the start, m")
p.add_argument("--yaw_std", type=float, default=0.03, help="the robot's heading error at the start, rad")
p.add_argument("--settle_s", type=float, default=2.0,
               help="from the reset to the goal reading: the hands go up over --raise_s, then the robot stands. Pilot p2 used 0.5 s "
                    "with the hands sent up at once: at the goal reading its robots were tilted 6.8 deg and moving 7 cm/s "
                    "(the benchmark's, after the walk and its 1.5 s settle: 2.1 deg, 0.3 cm/s)")
p.add_argument("--raise_s", type=float, default=1.0, help="the hands go from the default pose to --carry_up above it over this long (walk_grasp.py: 1 s)")
p.add_argument("--carry_up", type=float, default=0.20, help="hands raised this much before the grasp, as the walk carries them")
p.add_argument("--table_z", type=float, default=-0.03)
p.add_argument("--roll_damp", type=float, default=20.0)
# Rewards are a tenth of pilot p2's (0.5, 2, 60, 0.002) and the value loss is not clipped: in p2 the value loss was
# about 250 on every update that held the end reward and near 0 otherwise, from first to last iteration (a clipped value
# loss moves the estimate at most 0.2 per update toward returns near 60), so the critic never learned the end reward.
p.add_argument("--w_reach", type=float, default=0.05)
p.add_argument("--w_rise", type=float, default=0.2)
p.add_argument("--w_lifted", type=float, default=6.0)
p.add_argument("--w_rate", type=float, default=0.0002)
p.add_argument("--lam", type=float, default=0.98)
p.add_argument("--gamma", type=float, default=0.998)
p.add_argument("--clip_value", type=int, default=0, help="PPO's clipped value loss (p2: 1)")
p.add_argument("--init_std", type=float, default=0.15, help="the policy's starting exploration noise (action units)")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--desync", type=int, default=1,
               help="train: start every robot at a random point of its first episode, so later episodes are spread over all "
                    "phases (pilot p2 ran with all 1,024 robots in step: each PPO update then saw one 0.96 s slice of the "
                    "grab, and the lifted reward arrived in one update of every 7.5). A robot that starts past the settle "
                    "has no goal for that first episode and sits it out.")
p.add_argument("--steps_per_env", type=int, default=48)
p.add_argument("--time_limit_h", type=float, default=0.0)
p.add_argument("--tag", default="")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = True
app = AppLauncher(args).app


def _fail(*exc):  # Kit's python.sh exits 0 after an uncaught exception
    import traceback
    traceback.print_exception(*exc)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


sys.excepthook = _fail

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.envs.mdp as base_mdp  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as mu  # noqa: E402
import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: E402,F401
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import EventTermCfg as EventTerm  # noqa: E402
from isaaclab.managers import ObservationGroupCfg as ObsGroup  # noqa: E402
from isaaclab.managers import ObservationTermCfg as ObsTerm  # noqa: E402
from isaaclab.managers import RewardTermCfg as RewTerm  # noqa: E402
from isaaclab.managers import TerminationTermCfg as DoneTerm  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.manager_based.locomanipulation.pick_place.configs.action_cfg import AgileBasedLowerBodyActionCfg  # noqa: E402
from isaaclab_tasks.manager_based.locomanipulation.pick_place.locomanipulation_g1_env_cfg import LocomanipulationG1EnvCfg  # noqa: E402
from isaaclab_tasks.manager_based.locomanipulation.pick_place.mdp.actions import AgileBasedLowerBodyAction  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_spec  # noqa: E402
import grab_policy  # noqa: E402
import grab_residual as GR  # noqa: E402
import hull_objects  # noqa: E402
from batch_ik import BatchIKAction, BatchIKActionCfg  # noqa: E402
from so101 import g1  # noqa: E402
from so101.objects import layout, random_objects  # noqa: E402

if (args.base != "scripted") != bool(args.policy):
    raise SystemExit("--base <name> needs --policy <file>; --base scripted takes none")
N = args.num_envs
DEV = "cuda:0"
STAND_Z = 0.75
TOP = STAND_Z + args.table_z
EDGE_Y = 1.0  # the table's near edge, env y; the robot stands in front of it, facing +y
SETTLE = int(round(args.settle_s / 0.02))
STEPS = SETTLE + int(round(GR.T_TOTAL / 0.02))
TRAIN = args.mode == "train"

# --- objects: one per env, fixed for the run (its pose is redrawn at every reset) -------------------------------------------
shapes = [s.replace("-", " ") for s in args.shape.split(",")]
SEED = bench_spec.BASE["walk_carry"][args.split]
HULL_FILE = None


def primitive_layouts(n_each):  # the benchmark's layouts, shape by shape (seeds SEED + i, as walk_grasp.py)
    out = []
    for s in shapes:
        for i in range(n_each):
            rng = np.random.default_rng(SEED + i)
            spec = random_objects(rng, 1, [s])[0]
            x, y, a = layout(rng, [spec])[0]
            out.append(dict(shape=s, seed=SEED + i, spec=spec, d=float(x) - 0.03, y=float(min(y, 0.04)), heading=float(a)))
    return out


if args.objects == "primitives":
    objects = primitive_layouts(math.ceil(N / len(shapes)))
    objects = objects[:N] if TRAIN else objects
else:
    kind, label = ("gso", "household") if args.objects == "gso" else ("pool", "pool")
    ES, HULL_FILE = hull_objects.entries(kind, args.split, args.pool)
    assert ES, f"no {kind} objects in the {args.split} split"
    if not TRAIN:
        assert args.objects != "mix", "eval: one object set at a time (--objects primitives | gso | pool)"
        objects = hull_objects.layouts(ES, args.n, SEED, label)  # walk_grasp.py --objects's layouts
    else:
        # every object of the split in turn; a repeat of an object draws a new friction and nominal pose. 40 seeds per
        # object: the pool's 425 training objects stay inside the split's seed range (the benchmark's 100 would not)
        n_hull = int(round(N * args.pool_share)) if args.objects == "mix" else N
        assert SEED + 40 * len(ES) < bench_spec.SPLITS[args.split][1] and (n_hull - 1) // len(ES) < 40, "seeds would leave the split"
        hulls = [hull_objects.layout_of(ES[k % len(ES)], SEED + 40 * (k % len(ES)) + k // len(ES), label) for k in range(n_hull)]
        objects = (primitive_layouts(math.ceil((N - n_hull) / len(shapes)))[:N - n_hull] if n_hull < N else []) + hulls
if not TRAIN:
    N = len(objects)
SPECS = [o["spec"] for o in objects]
HULL = torch.tensor([s["type"] == "hull" for s in SPECS], device=DEV)  # scanned or pooled: judged at the grasp point, not the root
GRASP_LOCAL = torch.tensor([hull_objects.grasp_local(s) for s in SPECS], dtype=torch.float32, device=DEV)  # where the pinch closes, object frame
CENTRE_XY = torch.tensor([s["entry"]["centre_xy_m"] if s["type"] == "hull" else [0.0, 0.0] for s in SPECS], dtype=torch.float32, device=DEV)
REST_UP = torch.where(HULL, 0.002, 0.001)  # walk_grasp.py place_of: scanned objects are set down from 2 mm, primitives from 1
GROUPS = list(dict.fromkeys(o["shape"] for o in objects))
AXIS = [s.get("quat_axis") or ("x" if s["type"] == "ellipsoid" else None) for s in SPECS]
LYING = torch.tensor([a is not None and s["type"] != "sphere" for a, s in zip(AXIS, SPECS)], device=DEV)
BALL = torch.tensor([s["type"] == "sphere" for s in SPECS], device=DEV)
LONG = torch.tensor([[1.0, 0, 0] if a == "x" else [0, 1.0, 0] for a in AXIS], device=DEV)
SHAPE_ID = GR.shape_ids([o["shape"] for o in objects]).to(DEV)
SIZE = torch.tensor([list(map(float, s["size"][:3])) for s in SPECS], dtype=torch.float32, device=DEV)


def rest_height(s):  # walk_grasp.py's: the egg's capsule stand-in rests on its radius
    return (s["size"][1] + s["size"][2]) / 2 if s["type"] == "ellipsoid" else s["z"]


REST = torch.tensor([rest_height(s) for s in SPECS], dtype=torch.float32, device=DEV)


def obj_cfg(s):  # walk_grasp.py's obj_cfg for primitives
    z = s["size"]
    rolls = s.get("quat_axis") is not None or s["type"] in ("sphere", "ellipsoid")
    common = dict(rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=32, solver_velocity_iteration_count=1,
                                                               angular_damping=args.roll_damp if rolls else None),
                  mass_props=sim_utils.MassPropertiesCfg(mass=s["mass"]),
                  collision_props=sim_utils.CollisionPropertiesCfg(torsional_patch_radius=0.01, min_torsional_patch_radius=0.005),
                  physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=s["friction"], dynamic_friction=s["friction"],
                                                                  restitution=0.0, friction_combine_mode="max"),
                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(s["rgba"][:3])))
    if s["type"] == "hull":
        common.pop("visual_material")
        return hull_objects.hull_cfg(s["name"], HULL_FILE, **common)
    return {"box": lambda: sim_utils.CuboidCfg(size=(2 * z[0], 2 * z[1], 2 * z[2]), **common),
            "cylinder": lambda: sim_utils.CylinderCfg(radius=z[0], height=2 * z[1], axis=(s.get("quat_axis") or "z").upper(), **common),
            "capsule": lambda: sim_utils.CapsuleCfg(radius=z[0], height=2 * z[1], axis=(s.get("quat_axis") or "z").upper(), **common),
            "sphere": lambda: sim_utils.SphereCfg(radius=z[0], **common),
            "ellipsoid": lambda: sim_utils.CapsuleCfg(radius=(z[1] + z[2]) / 2, height=max(2 * z[0] - (z[1] + z[2]), 0.002),
                                                      axis="X", **common)}[s["type"]]()


# --- the grab's state, per env (module level: the terms below read and write it) --------------------------------------------
class S:
    res = torch.zeros(N, GR.N_ACT, device=DEV)  # the residual so far
    goal_c = torch.zeros(N, 3, device=DEV)  # env frame: where the open-loop pinch closes
    goal_q = torch.zeros(N, 4, device=DEV)
    hand0 = torch.zeros(N, 3, device=DEV)  # the pinch point when the grasp starts
    close_to = torch.ones(N, device=DEV)
    rate = torch.ones(N, device=DEV)
    z0 = torch.zeros(N, device=DEV)  # the object's height when the goal is read
    q0 = torch.zeros(N, 4, device=DEV)
    ready = torch.zeros(N, dtype=torch.bool, device=DEV)  # the goal has been read this episode
    pinch_target = torch.zeros(N, 3, device=DEV)
    finger_target = torch.zeros(N, 7, device=DEV)
    left = None  # (pos, quat) of the left wrist in the pelvis frame, held there
    right_rest = None
    lifted = torch.zeros(N, dtype=torch.bool, device=DEV)
    obj_pose = torch.zeros(N, 7, device=DEV)  # eval: the benchmark's object pose per env (robot frame); train: redrawn
    act = torch.zeros(N, GR.N_ACT, device=DEV)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    printed = False
    skip = torch.zeros(N, dtype=torch.bool, device=DEV)  # --desync: a robot's first, shortened episode runs no grab
    obj_start = torch.zeros(N, 3, device=DEV)  # the object's position as placed at reset (world)
    n_ended = 0  # episodes ended since the last log line (any reason), and how many of them ended lifted
    n_lifted = 0
    n_fell = 0
    n_ended_h = 0  # the same two counts for the robots holding a scanned or pooled object
    n_lifted_h = 0


POL = grab_policy.load(args.policy, DEV) if args.policy else None
H = torch.tensor(np.column_stack([g1.DOWN, g1.CLOSE, np.cross(g1.DOWN, g1.CLOSE)]), dtype=torch.float32, device=DEV)
PINCH_LOCAL = torch.tensor(g1.PINCH, dtype=torch.float32, device=DEV)
UP = torch.tensor([0.0, 0.0, 1.0], device=DEV)
Z_MIN = TOP + g1.Hand.MIN_Z
open_q = dict(zip(g1.HAND, [g1.THUMB_0, *g1.THUMB[-1, 1:], *g1.INDEX, *g1.MIDDLE]))
closed_q = dict(zip(g1.HAND, [g1.THUMB_0, *g1.THUMB[0, 1:], *g1.INDEX, *g1.MIDDLE]))


def _robot(env):
    return env.scene["robot"]


def _obj(env):
    return env.scene["object"]


def pinch_w(env):
    r = _robot(env)
    hb = r.body_names.index(g1.HAND_BODY)
    return r.data.body_pos_w[:, hb] + mu.quat_apply(r.data.body_quat_w[:, hb], PINCH_LOCAL.expand(env.num_envs, 3))


def grasp_w(env):  # the point of the object the pinch closes on (walk_grasp.py grasp_pt; primitives: the centre)
    o = _obj(env)
    return o.data.root_pos_w + mu.quat_apply(o.data.root_quat_w, GRASP_LOCAL)


def grasp_time(env):  # seconds since the goal was read (negative during the settle)
    return (env.episode_length_buf - SETTLE).float() * env.step_dt


def tilt_now(env):
    q = _obj(env).data.root_quat_w
    n = q.shape[0]
    zq, zs = mu.quat_apply(q, UP.expand(n, 3)), mu.quat_apply(S.q0, UP.expand(n, 3))
    axis = torch.rad2deg(torch.arccos(torch.clamp((zq * zs).sum(1), -1, 1)))
    flat = torch.rad2deg(torch.arcsin(torch.clamp((mu.quat_apply(q, LONG) * UP).sum(1).abs(), 0, 1)))
    return torch.where(LYING, flat, axis)


def is_lifted(env):
    """bench_spec.lifted (9 cm up, tilt under 45 deg) with the object's grasp point within 5 cm of the pinch point and
    the robot up; balls are not tilt-tested (a ball has no up)."""
    rise = _obj(env).data.root_pos_w[:, 2] - S.z0
    near = (grasp_w(env) - pinch_w(env)).norm(dim=1)
    up = _robot(env).data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] > 0.5
    return (rise >= bench_spec.LIFT_M) & ((tilt_now(env) < bench_spec.LIFTED_TILT_DEG) | BALL) & (near < 0.05) & up & S.ready


def read_goal(env, ids):
    """The open-loop grab for envs `ids`, from the object where it lies now and the robot where it stands (walk_grasp.py's
    goal reading and its --grab path)."""
    if len(ids) == 0:
        return
    r, o, org = _robot(env), _obj(env), env.scene.env_origins
    hb = r.body_names.index(g1.HAND_BODY)
    oc = (grasp_w(env) - org)[ids]  # scanned objects: above the root, over the footprint centre
    c0 = oc.clone()
    c0[:, 2] = torch.clamp(c0[:, 2], min=Z_MIN)
    R0 = mu.matrix_from_quat(r.data.body_quat_w[ids, hb])
    Ro = mu.matrix_from_quat(o.data.root_quat_w[ids])
    heading = (torch.atan2(Ro[:, 1, 1], Ro[:, 0, 1]) - math.pi / 2).cpu().numpy()
    yaw = GR.yaw_of(r.data.root_quat_w[ids]).cpu().numpy()
    pel = (r.data.root_pos_w - org)[ids].cpu().numpy()
    ocn = oc.cpu().numpy()
    h0, feat = [], []
    for k, i in enumerate(ids.tolist()):
        s = SPECS[i]
        hull = s["type"] == "hull"
        h = float(heading[k]) + s["entry"]["narrow_rad"] if hull else grab_policy.narrow_heading(s, float(heading[k]))
        h0.append(h)
        if POL is not None:
            g = np.random.default_rng([objects[i]["seed"], int(env.common_step_counter), 13])
            dc, sc = (g.normal(0, args.noise / 100, 3), 1 + g.normal(0, 0.05)) if args.noise > 0 else (np.zeros(3), 1.0)
            x, y, z = grab_policy.robot_frame(ocn[k, :2] + dc[:2], ocn[k, 2] + dc[2], pel[k, :2], yaw[k], TOP)
            shape, size = hull_objects.stand_in(s["entry"]) if hull else (objects[i]["shape"], s["size"])  # v8 sees a primitive class
            feat.append(grab_policy.features(shape, size, h - yaw[k], x, y, z, sc))
    h0 = torch.tensor(h0, dtype=torch.float32, device=DEV)
    adj = grab_policy.adjustments(POL, feat) if POL is not None else torch.zeros(len(ids), 8, device=DEV)
    c, q = grab_policy.grasp_goals(adj, c0, h0, R0, H, z_min=Z_MIN, up=UP)
    S.goal_c[ids], S.goal_q[ids] = c, q
    S.close_to[ids], S.rate[ids] = grab_policy.close_params(adj)
    S.hand0[ids] = (pinch_w(env) - org)[ids]
    S.z0[ids], S.q0[ids] = o.data.root_pos_w[ids, 2], o.data.root_quat_w[ids]
    S.ready[ids] = True
    if len(ids) >= N // 2 and not S.printed:  # where the robots stand when the goal is read (once per run)
        S.printed = True
        rp = (r.data.root_pos_w - org)[ids]
        gap = EDGE_Y - rp[:, 1]
        ye = (GR.yaw_of(r.data.root_quat_w[ids]) - math.pi / 2).abs()
        grav = mu.quat_apply_inverse(r.data.root_quat_w[ids], -UP.expand(len(ids), 3))
        tilt = torch.rad2deg(torch.arccos(torch.clamp(-grav[:, 2], -1, 1)))
        moved = (o.data.root_pos_w[ids, :2] - S.obj_start[ids, :2]).norm(dim=1)
        print(f"GOAL-READ after {args.settle_s} s: pelvis to edge m median {gap.median():.3f} (p10 {gap.quantile(0.1):.3f}, p90 {gap.quantile(0.9):.3f}); "
              f"sideways |x| {rp[:, 0].abs().median():.3f}; heading error rad {ye.median():.3f}; pelvis height {rp[:, 2].median():.3f}; tilt deg {tilt.median():.1f}; "
              f"base speed m/s {r.data.root_lin_vel_w[ids].norm(dim=1).median():.3f}; object moved since reset m median {moved.median():.4f}, p90 {moved.quantile(0.9):.4f}; "
              f"pinch point above the table m {(S.hand0[ids, 2] - TOP).median():.3f}", flush=True)


class GrabArm(BatchIKAction):
    """5 policy numbers -> the 28-number wrist pose + hand action batch_ik.py tracks: the open-loop pinch with the
    residual on it for the right hand; the left wrist held where it starts, relative to the pelvis."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._raw = torch.zeros(self.num_envs, 28, device=self.device)
        self._open = torch.tensor([float(open_q.get(j, 0.0)) for j in cfg.hand_joint_names], dtype=torch.float32, device=self.device)
        self._shut = torch.tensor([float(closed_q.get(j, 0.0)) for j in cfg.hand_joint_names], dtype=torch.float32, device=self.device)
        self._right = torch.tensor([j in open_q for j in cfg.hand_joint_names], device=self.device)
        self._rf = [list(cfg.hand_joint_names).index(j) for j in g1.HAND]  # the right hand's 7 columns, g1.HAND order

    @property
    def action_dim(self):
        return GR.N_ACT

    @property
    def raw_actions(self):
        return S.act

    def process_actions(self, actions):
        env, a, n = self._env, self._asset, self.num_envs
        org = env.scene.env_origins
        rp, rq = a.data.root_pos_w, a.data.root_quat_w
        lw, hb = a.body_names.index("left_wrist_yaw_link"), a.body_names.index(g1.HAND_BODY)
        if S.left is None:  # first call: every robot at its default pose; the hands ride --carry_up above it
            S.left = list(mu.subtract_frame_transforms(rp, rq, a.data.body_pos_w[:, lw], a.data.body_quat_w[:, lw]))
            S.right_rest = list(mu.subtract_frame_transforms(rp, rq, a.data.body_pos_w[:, hb], a.data.body_quat_w[:, hb]))
        t = env.episode_length_buf
        read_goal(env, ((t == SETTLE) & ~S.ready & ~S.skip).nonzero().flatten())
        S.act[:] = actions.clamp(-1.0, 1.0)
        live = S.ready
        S.res[:] = GR.apply(S.res, S.act, live)
        up_now = torch.zeros(n, 3, device=self.device)  # the hands' raise so far (pelvis frame), ramped as the walk ramps it
        up_now[:, 2] = args.carry_up * torch.clamp((t.float() + 1) * env.step_dt / max(args.raise_s, 1e-6), 0.0, 1.0)
        lp, lq = mu.combine_frame_transforms(rp, rq, S.left[0] + up_now, S.left[1])
        hp, hq = mu.combine_frame_transforms(rp, rq, S.right_rest[0] + up_now, S.right_rest[1])  # before the goal: the raised pose
        pinch, f, _ = GR.schedule(grasp_time(env).clamp(min=0.0), S.goal_c, S.hand0, S.rate)
        pc, qc, grip = GR.corrected(pinch, f, S.goal_q, S.close_to, S.res, GR.yaw_of(rq), Z_MIN)
        wp = pc + org - mu.quat_apply(qc, PINCH_LOCAL.expand(n, 3))
        rpos = torch.where(live[:, None], wp, hp)
        rquat = torch.where(live[:, None], qc, hq)
        grip = torch.where(live, grip, torch.zeros_like(grip))
        hand = torch.where(self._right, self._open + grip[:, None] * (self._shut - self._open), torch.zeros_like(self._open))
        S.pinch_target[:] = torch.where(live[:, None], pc, pinch_w(env) - org)
        S.finger_target[:] = hand[:, self._rf]
        self._raw[:] = torch.cat([lp - org, lq, rpos - org, rquat, hand], 1)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        S.res[env_ids] = 0.0
        S.act[env_ids] = 0.0
        S.ready[env_ids] = False
        S.lifted[env_ids] = False
        S.skip[env_ids] = False


@configclass
class GrabArmCfg(BatchIKActionCfg):
    class_type: type = GrabArm


class Legs(AgileBasedLowerBodyAction):
    """AGILE legs, standing: no policy numbers, zero speed, hip height 0.72."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.command = torch.tensor([0.0, 0.0, 0.0, 0.72], device=self.device).repeat(self.num_envs, 1)

    @property
    def action_dim(self):
        return 0

    def process_actions(self, actions):
        super().process_actions(self.command)


@configclass
class LegsCfg(AgileBasedLowerBodyActionCfg):
    class_type: type = Legs


# --- reset: the robot where the walk would stop, the object on the table in front of it -------------------------------------
def reset_grab(env, env_ids):
    n = len(env_ids)
    r, o, org = _robot(env), _obj(env), env.scene.env_origins
    ids = env_ids.cpu()
    u = lambda lo, hi: torch.rand(n, generator=S.gen) * (hi - lo) + lo  # noqa: E731
    g = lambda std: torch.randn(n, generator=S.gen) * std  # noqa: E731
    if TRAIN:
        stand = (args.stand_mean + g(args.stand_std)).clamp(0.10, 0.18)
        side, yaw_e = g(args.side_std), g(args.yaw_std)
        d = torch.tensor([objects[i]["d"] for i in ids.tolist()]) + u(-0.02, 0.02)
        y = (torch.tensor([objects[i]["y"] for i in ids.tolist()]) + u(-0.03, 0.03)).clamp(-0.14, 0.04)
        head = u(-math.pi, math.pi)
    else:  # the benchmark's layout for this env, the robot at the walk's mean stop
        stand = torch.full((n,), args.stand_mean)
        side, yaw_e = torch.zeros(n), torch.zeros(n)
        d = torch.tensor([objects[i]["d"] for i in ids.tolist()])
        y = torch.tensor([objects[i]["y"] for i in ids.tolist()])
        head = torch.tensor([objects[i]["heading"] for i in ids.tolist()])
    st = r.data.default_root_state[env_ids].clone()
    st[:, 0] = side.to(DEV)
    st[:, 1] = EDGE_Y - stand.to(DEV)
    yaw = math.pi / 2 + yaw_e.to(DEV)
    st[:, 3:7] = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], 1)
    st[:, :3] += org[env_ids]
    st[:, 7:] = 0.0
    r.write_root_pose_to_sim(st[:, :7], env_ids=env_ids)
    r.write_root_velocity_to_sim(st[:, 7:], env_ids=env_ids)
    r.write_joint_state_to_sim(r.data.default_joint_pos[env_ids].clone(), r.data.default_joint_vel[env_ids].clone(), env_ids=env_ids)
    # the object: d past the edge, y to the robot's left (the robot faces +y: left is -x), as walk_grasp.py place_of
    # scanned objects: their footprint centre at that spot (the root is offset from it, turned with the heading)
    pose = torch.zeros(n, 7, device=DEV)
    hd, cx, cy = head.to(DEV), CENTRE_XY[env_ids, 0], CENTRE_XY[env_ids, 1]
    pose[:, 0] = -y.to(DEV) - (torch.cos(hd) * cx - torch.sin(hd) * cy)
    pose[:, 1] = EDGE_Y + d.to(DEV) - (torch.sin(hd) * cx + torch.cos(hd) * cy)
    pose[:, 2] = TOP + REST[env_ids] + REST_UP[env_ids]
    pose[:, 3], pose[:, 6] = torch.cos(hd / 2), torch.sin(hd / 2)
    pose[:, :3] += org[env_ids]
    o.write_root_pose_to_sim(pose, env_ids=env_ids)
    o.write_root_velocity_to_sim(torch.zeros(n, 6, device=DEV), env_ids=env_ids)
    S.obj_start[env_ids] = pose[:, :3]


# --- observations, rewards, terminations ------------------------------------------------------------------------------------
def obs_grab(env):
    r, o, org = _robot(env), _obj(env), env.scene.env_origins
    hb = r.body_names.index(g1.HAND_BODY)
    t = grasp_time(env).clamp(min=0.0)
    _, _, phase = GR.schedule(t, S.goal_c, S.hand0, S.rate)
    ids, _ = r.find_joints(list(g1.HAND), preserve_order=True)
    root = r.data.root_pos_w - org
    gp = grasp_w(env) - org
    return GR.observation(t, phase, S.res, gp, o.data.root_quat_w, o.data.root_lin_vel_w, o.data.root_ang_vel_w,
                          pinch_w(env) - org, S.pinch_target, r.data.body_quat_w[:, hb], torch.where(S.ready[:, None], S.goal_c, gp),
                          r.data.joint_pos[:, ids], S.finger_target, root, r.data.root_quat_w, r.data.root_ang_vel_w, SHAPE_ID, SIZE)


def r_reach(env):  # before the close ends: the pinch point near the object
    d = (grasp_w(env) - pinch_w(env)).norm(dim=1)
    return (1 - torch.tanh(d / 0.03)) * S.ready * (grasp_time(env) < GR.EDGES[4])


def r_rise(env):  # the object up with the hand on it
    near = (grasp_w(env) - pinch_w(env)).norm(dim=1) < 0.05
    return ((_obj(env).data.root_pos_w[:, 2] - S.z0).clamp(0.0, 0.10) / 0.10) * near * S.ready


def grab_end(env):
    """The grab is over. A termination, not a time-out: rsl_rl adds the critic's value to the last reward of a
    timed-out episode, as if it went on; this one does not."""
    return env.episode_length_buf >= env.max_episode_length


def r_lifted(env):  # once, on the grab's last step (the termination manager has just run for this step)
    last = grab_end(env)
    S.lifted[:] = is_lifted(env)
    won = S.lifted & last
    ended = env.reset_buf & S.ready  # episodes that ran a grab (a desynced first episode has no goal)
    S.n_ended += int(ended.sum())
    S.n_lifted += int(won.sum())
    S.n_ended_h += int((ended & HULL).sum())
    S.n_lifted_h += int((won & HULL).sum())
    S.n_fell += int((env.reset_buf & S.ready & robot_fell(env)).sum())
    return won.float()


def r_rate(env):
    return S.act.pow(2).sum(1)


def robot_fell(env):
    return _robot(env).data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] < 0.5


def object_gone(env):  # off the table (below its top) or far from where it lay
    org = env.scene.env_origins
    return ((_obj(env).data.root_pos_w - org)[:, 2] < TOP - 0.05) | ((((grasp_w(env) - org)[:, :2] - S.goal_c[:, :2]).norm(dim=1) > 0.30) & S.ready)


@configclass
class PolicyObs(ObsGroup):
    grab = ObsTerm(func=obs_grab)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class Rewards:
    reach = RewTerm(func=r_reach, weight=args.w_reach)
    rise = RewTerm(func=r_rise, weight=args.w_rise)
    lifted = RewTerm(func=r_lifted, weight=args.w_lifted / 0.02)  # the manager multiplies by the 20 ms step
    rate = RewTerm(func=r_rate, weight=-args.w_rate)


@configclass
class Terminations:
    grab_end = DoneTerm(func=grab_end)
    fell = DoneTerm(func=robot_fell)
    gone = DoneTerm(func=object_gone)


@configclass
class Events:
    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")
    grab = EventTerm(func=reset_grab, mode="reset")


static = dict(collision_props=sim_utils.CollisionPropertiesCfg(),
              physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.8),
              visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.5, 0.45)))


@configclass
class GrabEnvCfg(LocomanipulationG1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        pink = self.actions.upper_body_ik
        self.actions.upper_body_ik = GrabArmCfg(asset_name="robot", hand_joint_names=list(pink.hand_joint_names))
        legs = self.actions.lower_body_joint_pos
        self.actions.lower_body_joint_pos = LegsCfg(asset_name="robot", joint_names=legs.joint_names, policy_output_scale=legs.policy_output_scale,
                                                    obs_group_name=legs.obs_group_name, policy_path=legs.policy_path)
        self.observations.policy = PolicyObs()
        self.rewards = Rewards()
        self.terminations = Terminations()
        self.events = Events()
        self.episode_length_s = STEPS * 0.02
        self.scene.num_envs = N
        self.scene.env_spacing = 2.5
        self.scene.replicate_physics = False  # a different object per environment
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = False
        self.scene.packing_table = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Table", init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, EDGE_Y + 0.3, TOP - 0.02)),
                                                spawn=sim_utils.CuboidCfg(size=(1.0, 0.6, 0.04), **static))
        self.scene.pedestal = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Pedestal", init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, EDGE_Y + 0.425, (TOP - 0.04) / 2)),
                                           spawn=sim_utils.CuboidCfg(size=(0.6, 0.35, TOP - 0.04), **static))
        self.scene.object = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Object",
                                           spawn=sim_utils.MultiAssetSpawnerCfg(assets_cfg=[obj_cfg(s) for s in SPECS], random_choice=False),
                                           init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, EDGE_Y + 0.15, TOP + 0.02)))
        self.sim.device = DEV
        self.teleop_devices = None
        self.recorders = None


@configclass
class RunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = args.steps_per_env
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    save_interval = 25
    experiment_name = "grab_rl"
    policy = RslRlPpoActorCriticCfg(init_noise_std=args.init_std, actor_obs_normalization=True, critic_obs_normalization=True,
                                    actor_hidden_dims=[256, 128, 64], critic_hidden_dims=[256, 128, 64], activation="elu")
    algorithm = RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=bool(args.clip_value), clip_param=0.2, entropy_coef=0.002,
                                     num_learning_epochs=5, num_mini_batches=4, learning_rate=3.0e-4, schedule="adaptive",
                                     gamma=args.gamma, lam=args.lam, desired_kl=0.01, max_grad_norm=1.0)


torch.set_float32_matmul_precision("high")
torch.manual_seed(args.seed)
env = ManagerBasedRLEnv(cfg=GrabEnvCfg())
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
HYPER = dict(vars(args), residual=GR.describe(), steps_per_episode=STEPS, ik="batch", n_envs=N, n_scanned=int(HULL.sum()),
             n_scanned_objects=len({s["name"] for s in SPECS if s["type"] == "hull"}), **bench_spec.machine())

if TRAIN:
    agent = RunnerCfg()
    agent.max_iterations = args.iters
    agent.seed = args.seed
    venv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = OnPolicyRunner(venv, agent.to_dict(), log_dir=str(out), device=DEV)
    last = [m for m in runner.alg.policy.actor if isinstance(m, torch.nn.Linear)][-1]
    torch.nn.init.zeros_(last.weight)  # the policy starts as the open-loop grab (zero residual) plus its exploration noise
    torch.nn.init.zeros_(last.bias)
    if args.resume:
        print(f"starting from {args.resume}", flush=True)
        runner.load(args.resume, load_optimizer=False)
    (out / "hyper.json").write_text(json.dumps(dict(HYPER, runner=agent.to_dict()), indent=1, default=str))
    t0 = time.time()
    _log = runner.log
    hist = []

    def log_and_watch(locs, *a, **kw):
        _log(locs, *a, **kw)
        it = int(locs["it"])
        eps = locs.get("ep_infos", [])

        def stat(key):
            v = [float(e[key].mean()) if hasattr(e[key], "mean") else float(e[key]) for e in eps if key in e]
            return float(np.mean(v)) if v else float("nan")
        if it % 15 == 14:  # an episode is 360 steps = 7.5 iterations of 48: every 15 iterations, two rounds of episodes
            n_p, n_h = S.n_ended - S.n_ended_h, S.n_ended_h  # primitives, scanned or pooled objects
            row = dict(it=it, episodes=S.n_ended, lifted=S.n_lifted / max(1, S.n_ended), fell=S.n_fell / max(1, S.n_ended),
                       lifted_prim=(S.n_lifted - S.n_lifted_h) / n_p if n_p else float("nan"), lifted_pool=S.n_lifted_h / n_h if n_h else float("nan"),
                       reach=stat("Episode_Reward/reach"), rise=stat("Episode_Reward/rise"),
                       reward=float(np.mean(locs["rewbuffer"])) if locs.get("rewbuffer") else float("nan"),
                       noise_std=float(runner.alg.policy.action_std.mean()), elapsed_h=(time.time() - t0) / 3600)
            S.n_ended = S.n_lifted = S.n_fell = S.n_ended_h = S.n_lifted_h = 0
            hist.append(row)
            print("LEARN " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
            (out / "learn.json").write_text(json.dumps(hist))
        if args.time_limit_h > 0 and time.time() - t0 > args.time_limit_h * 3600:
            runner.save(str(out / f"model_{it}.pt"))
            (out / "learn.json").write_text(json.dumps(hist))
            print(f"TRAIN-STOP reason=time it={it}", flush=True)
            sys.stdout.flush()
            os._exit(0)
    runner.log = log_and_watch
    if args.desync:
        S.skip[:] = True  # every robot starts mid-episode (init_at_random_ep_len): no grab until its first reset
    runner.learn(num_learning_iterations=args.iters, init_at_random_ep_len=bool(args.desync))
    (out / "learn.json").write_text(json.dumps(hist))
    print(f"TRAIN-DONE {time.time() - t0:.0f} s", flush=True)
    sys.stdout.flush()
    os._exit(0)

# --- eval: one episode per env on the benchmark's layouts; the residual (or none) on the open-loop grab ----------------------
actor = GR.load(args.checkpoint, DEV) if args.checkpoint else None
obs, _ = env.reset()
done_any = torch.zeros(N, dtype=torch.bool, device=DEV)
lifted = torch.zeros(N, dtype=torch.bool, device=DEV)
rise = torch.zeros(N, device=DEV)
near = torch.zeros(N, device=DEV)
fell = torch.zeros(N, dtype=torch.bool, device=DEV)
res_abs = torch.zeros(N, GR.N_ACT, device=DEV)
t0 = time.time()
for k in range(STEPS):
    with torch.no_grad():
        a = actor(obs["policy"]) if actor is not None else torch.zeros(N, GR.N_ACT, device=DEV)
    if k == STEPS - 1:  # the last step ends the episode and resets: judge before it, on the state the step leaves
        pass
    obs, _, term, trunc, _ = env.step(a)
    live = ~done_any
    if k < STEPS - 1:
        o = env.scene["object"].data.root_pos_w
        rise = torch.where(live, o[:, 2] - S.z0, rise)
        near = torch.where(live, (grasp_w(env) - pinch_w(env)).norm(dim=1), near)
        res_abs = torch.where(live[:, None], S.res.abs(), res_abs)
        lifted = torch.where(live, is_lifted(env), lifted)
    fell |= live & env.termination_manager.get_term("fell")  # read from the term: the env has already reset a fallen robot
    done_any |= term | trunc
rows = [dict(shape=objects[i]["shape"], seed=objects[i]["seed"], **({"name": SPECS[i]["name"]} if SPECS[i]["type"] == "hull" else {}),
             lifted=bool(lifted[i]), rise=round(float(rise[i]), 4), near=round(float(near[i]), 4),
             fell=bool(fell[i]), residual=[round(float(v), 4) for v in res_abs[i]]) for i in range(N)]
name = (Path(args.checkpoint).stem if args.checkpoint else "zero") + (f"-{args.tag}" if args.tag else "")
sets = "" if args.objects == "primitives" else f"-{args.objects}"  # primitives keep the file name the earlier pilots wrote
rec = dict(kind="grab_only", split=args.split, objects=args.objects, base=args.base, residual=args.checkpoint or None, settings=HYPER, rows=rows)
(out / f"eval-{name}{sets}-{args.split}-n{N}.json").write_text(json.dumps(rec, indent=1, default=str))
by = {s: sum(r["lifted"] for r in rows if r["shape"] == s) for s in GROUPS}
print(f"GRAB-EVAL {name} base {args.base} {args.objects} {args.split}: lifted {sum(r['lifted'] for r in rows)}/{N}; by shape {by}; fell {int(fell.sum())}; "
      f"mean |residual| offset m {res_abs[:, :3].mean().item():.4f}, yaw rad {res_abs[:, 3].mean().item():.3f}, grip {res_abs[:, 4].mean().item():.3f}; "
      f"{time.time() - t0:.0f} s", flush=True)
sys.stdout.flush()
os._exit(0)
