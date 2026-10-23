"""Walk to a table and pick an object up (owner 2026-10-09: "3 for G1", walking plus grasping), scripted, on Isaac Lab's
walking G1 (isaac/locomanip_walk.py: Isaac-PickPlace-Locomanipulation-G1-Abs-v0, AGILE legs + pink IK arms + Dex3).

Scene per environment: the robot starts WALK_FROM metres from a table's near edge, facing it; one primitive object
(so101/objects.py random_objects + seeds, as the lift tests) lies 12-20 cm past the edge, within reach of the right hand.
The table is a slab on a set-back pedestal, so the feet can go under its edge.
Phases: walk at --vx until the pelvis is --stand metres from the edge (each robot stops on its own), settle, then
g1_lift.py's scripted pinch on the right hand: pinch point 8 cm above the object, down to it, close the thumb onto the
index (g1.THUMB keyframes), lift 10 cm, hold. Arm and hand: wrist poses for the pink IK, from the pinch goal and
g1.PINCH; the left wrist rides along at its start pose relative to the pelvis.
Success, as g1_lift: the object at least 9 cm above its start and within 5 cm of the pinch point at the end, robot up.

--carry (owner 2026-10-15: walk, grasp, move, put down; scripted end to end before any learning): after the hold the
right hand stays where it is relative to the pelvis (raised --carry_raise), the robot backs off --back, turns right to
face a second table B, walks to it, lowers the object onto it --place_d past its edge, opens, lifts the hand and steps
back. Per-stage counts: lifted, turned, carried (still in hand at table B), placed (resting on table B, hand clear),
upright (axis within 20 deg of the start; balls always), no fall.

  python isaac/walk_grasp.py [--n=12] [--shape=box] [--vx=0.4] [--carry] [--record]   (from the IsaacLab folder)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pinocchio  # noqa: F401  (before AppLauncher: Isaac Lab's pinocchio, not Isaac Sim's)

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=12, help="layouts per shape")
p.add_argument("--shape", default="box", help="one shape per run (every env spawns the same primitive type)")
p.add_argument("--vx", type=float, default=0.4, help="walking command, m/s")
p.add_argument("--walk_from", type=float, default=1.2, help="start distance, pelvis to table edge, m")
p.add_argument("--stand", type=float, default=0.18, help="stop when the pelvis is this far from the edge, m")
p.add_argument("--hip_height", type=float, default=0.72)
p.add_argument("--table_z", type=float, default=-0.03, help="table top relative to the standing pelvis height (g1: -0.02)")
p.add_argument("--carry_up", type=float, default=0.20,
               help="hands raised this much (pelvis frame) before walking: at their default pose they sweep the table top")
p.add_argument("--high", type=float, default=0.20, help="first waypoint height above the object, m")
p.add_argument("--carry", action="store_true", help="then turn, walk to table B and put the object down")
p.add_argument("--arms", choices=["park", "raise"], default="raise",  # 2026-10-22: park + self_collide made 1/12 (round 4 raise: 8/12 box)
               help="arms while walking: park = hanging at the sides, elbows slightly bent (owner 2026-10-15: the raised pose "
                    "crossed the left arm through the body); raise = start pose + --carry_up (rounds 1-4)")
p.add_argument("--carry_x", type=float, default=9.0, help="--carry: wrist this far ahead of the pelvis while carrying, m")
p.add_argument("--self_collide", type=int, default=0, help="robot self-collisions (the locomanip G1 has them off)")
p.add_argument("--carry_raise", type=float, default=0.05, help="hand raised this much above the lift pose for carrying, m")
p.add_argument("--back", type=float, default=0.45, help="back off this far from table A before turning, m")
p.add_argument("--b_ahead", type=float, default=0.90, help="table B's near edge, this far from the turn spot, m")
p.add_argument("--place_d", type=float, default=0.14, help="object put down this far past table B's edge, m")
p.add_argument("--yaw_rate", type=float, default=1.0,
               help="turning command, rad/s (standing still, AGILE ignores |wz| 0.5 and turns 1.16 rad/s at 1.0)")
p.add_argument("--hand_drives", choices=["env", "matched"], default="env",
               help="Dex3 finger drives: the locomanip env's (stiffness 20, damping 2) or the MuJoCo-matched ones the "
                    "fixed-base grasps used (500, 4.5, armature 0.01, g1_dexsuite.py's effort limits)")
p.add_argument("--ik", choices=["pink", "batch"], default="pink",
               help="batch: isaac/batch_ik.py, all arms in one GPU step (owner 2026-10-22: parallelizable); pink: one robot at a time on the CPU")
p.add_argument("--slow_place", type=float, default=1.0, help="put-down durations x this (balls, capsules rolled off B)")
p.add_argument("--settle_place", type=float, default=0.0, help="s holding still at the resting height before the hand opens")
p.add_argument("--roll_damp", type=float, default=20.0, help="angular damping on round or lying objects (PhysX has no rolling friction)")
p.add_argument("--rand_layout", action="store_true", help="each robot its own scene (goal 2026-10-22: varied environments): table A distance and sideways offset, table B turn angle (75-105 deg, either side) and distance")
p.add_argument("--tag", default="", help="suffix for the output file names")
p.add_argument("--out", default="runs/isaac")
p.add_argument("--record", action="store_true", help="save env motion for isaac/render_mj.py-style redraws")
args = p.parse_args()
app = AppLauncher(headless=True).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401
import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: E402,F401  (blacklisted by default)
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.utils.math import (combine_frame_transforms, matrix_from_quat, quat_apply, quat_from_matrix,  # noqa: E402
                                 subtract_frame_transforms)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from so101 import g1  # noqa: E402  (constants only: hand frame axes, pinch point, finger keyframes)
from so101.objects import layout, random_objects  # noqa: E402

TASK = "Isaac-PickPlace-Locomanipulation-G1-Abs-v0"
SEED, LIFT, STAND_Z = 960000, 0.10, 0.75  # STAND_Z: the robot's start pelvis height (G1_29DOF_CFG)
PHASES = [("settle", 1.5), ("raise", 1.0), ("approach", 1.5), ("descend", 1.0), ("close", 1.2), ("lift", 1.0), ("hold", 1.0)]
WALK_MAX = 6.0  # s: robots not at the table by then are counted as not arrived

# --- layouts (robot frame: x forward from the pelvis at the stop, y left) ---------------------------------------------
shapes = args.shape.split(",")  # several shapes: --n layouts each, mixed in one run
layouts = []
for s in shapes:
    for i in range(args.n):
        rng = np.random.default_rng(SEED + i)
        spec = random_objects(rng, 1, [s])[0]
        x, y, a = layout(rng, [spec])[0]  # lift-test layout: x 0.125-0.215 past the pelvis line, y within +-0.12
        # fixed-base scene: pelvis at x -0.08, table edge at x 0.03 (g1.PELVIS, g1.TABLE_EDGE); keep the right half
        layouts.append(dict(shape=s, seed=SEED + i, spec=spec, d=float(x) - 0.03, y=float(min(y, 0.04)), heading=float(a)))
N = len(layouts)
if args.rand_layout:  # per-env scene, from each layout's seed
    for L in layouts:
        g = np.random.default_rng(L["seed"] + 7)
        L.update(edge=float(g.uniform(0.8, 1.8)), ax=float(g.uniform(-0.3, 0.3)),
                 b_turn=float(g.choice([-1, 1]) * g.uniform(5 * np.pi / 12, 7 * np.pi / 12)), b_dist=float(g.uniform(0.9, 1.5)))  # 75-105 deg: B clears A
else:  # the fixed layout: turn right 90 deg, table B b_ahead ahead
    for L in layouts:
        L.update(edge=args.walk_from, ax=0.0, b_turn=-np.pi / 2, b_dist=args.b_ahead)

cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=N)
if args.ik == "batch":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from batch_ik import use_batch_ik  # noqa: E402

    use_batch_ik(cfg)
cfg.episode_length_s = WALK_MAX + sum(d for _, d in PHASES) + 5 + (40 if args.carry else 0)  # no auto-reset mid-run
cfg.terminations.object_dropping = None
cfg.terminations.success = None
cfg.scene.replicate_physics = False  # a different object per environment
if args.hand_drives == "matched":  # LOG 2026-10-08: stiffness 20 gave 1/60 scripted lifts vs 35/60 at 500
    cfg.scene.robot.actuators["hands"] = cfg.scene.robot.actuators["hands"].replace(
        stiffness=500.0, damping=4.5, armature=0.01, effort_limit=None,
        effort_limit_sim={".*_hand_thumb_0_joint": 2.45, ".*_hand_(thumb_[12]|index_.|middle_.)_joint": g1.THUMB_FORCE})
cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = bool(args.self_collide)
if args.carry:
    cfg.scene.env_spacing = max(cfg.scene.env_spacing, 4.0)  # table B reaches 1.5 m to the robot's right
# The robot faces +y (G1_29DOF_CFG yaw 90 deg) from the env origin; the table's near edge is walk_from ahead of it.
EDGE_Y = args.walk_from
TOP = STAND_Z + args.table_z
static = dict(collision_props=sim_utils.CollisionPropertiesCfg(),
              physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.8),
              visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.5, 0.45)))


def fixture(name, pos, size):
    """A table part: a static asset, or (--rand_layout) a kinematic rigid body each env can pose at reset."""
    if not args.rand_layout:
        return AssetBaseCfg(prim_path="{ENV_REGEX_NS}/" + name, init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
                            spawn=sim_utils.CuboidCfg(size=size, **static))
    return RigidObjectCfg(prim_path="{ENV_REGEX_NS}/" + name, init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
                          spawn=sim_utils.CuboidCfg(size=size, rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                                                    mass_props=sim_utils.MassPropertiesCfg(mass=50.0), **static))


cfg.scene.packing_table = fixture("Table", (0.0, EDGE_Y + 0.3, TOP - 0.02), (1.0, 0.6, 0.04))  # slab 60 cm deep, 1 m wide
cfg.scene.pedestal = fixture("Pedestal", (0.0, EDGE_Y + 0.425, (TOP - 0.04) / 2), (0.6, 0.35, TOP - 0.04))  # set back 25 cm
# Table B (--carry): the robot backs off --back, turns right (to face +x) and walks to B's near edge at x = B_X.
TURN_Y = EDGE_Y - args.stand - args.back
B_X = args.b_ahead
B_W = 1.2 if args.rand_layout else 1.6  # a counter: robots drift sideways ~0.3 m on the walk (1.0 m wide: objects set down past its side edge)
if args.carry:
    cfg.scene.table_b = fixture("TableB", (B_X + 0.3, TURN_Y, TOP - 0.02), (0.6, B_W, 0.04))
    cfg.scene.pedestal_b = fixture("PedestalB", (B_X + 0.425, TURN_Y, (TOP - 0.04) / 2), (0.35, 0.6, TOP - 0.04))
if args.rand_layout:
    cfg.scene.env_spacing = max(cfg.scene.env_spacing, 6.0)
# one primitive type per run; each env's size, mass and friction differ only in pose here (env_0's spawn)
spec0 = layouts[0]["spec"]


def obj_cfg(s):
    z = s["size"]
    rolls = s.get("quat_axis") is not None or s["type"] in ("sphere", "ellipsoid")
    # PhysX has no rolling resistance: lying cylinders and eggs rolled off table A from the footsteps (mix smoke
    # 2026-10-22: 11 of 16 moved during the walk). --roll_damp adds angular damping to shapes that roll.
    common = dict(rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=32, solver_velocity_iteration_count=1,
                                                               angular_damping=args.roll_damp if rolls else None),
                  mass_props=sim_utils.MassPropertiesCfg(mass=s["mass"]),
                  collision_props=sim_utils.CollisionPropertiesCfg(torsional_patch_radius=0.01, min_torsional_patch_radius=0.005),
                  physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=s["friction"], dynamic_friction=s["friction"],
                                                                  restitution=0.0, friction_combine_mode="max"),
                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(s["rgba"][:3])))
    return {"box": lambda: sim_utils.CuboidCfg(size=(2 * z[0], 2 * z[1], 2 * z[2]), **common),
            # lying shapes (quat_axis "y": lying cylinder, capsule) were built along Z, i.e. standing (fixed 2026-10-22)
            "cylinder": lambda: sim_utils.CylinderCfg(radius=z[0], height=2 * z[1], axis=(s.get("quat_axis") or "z").upper(), **common),
            "capsule": lambda: sim_utils.CapsuleCfg(radius=z[0], height=2 * z[1], axis=(s.get("quat_axis") or "z").upper(), **common),
            "sphere": lambda: sim_utils.SphereCfg(radius=z[0], **common),
            # egg: Isaac Lab has no ellipsoid; a capsule lying along x with the egg's length and mean width
            "ellipsoid": lambda: sim_utils.CapsuleCfg(radius=(z[1] + z[2]) / 2, height=max(2 * z[0] - (z[1] + z[2]), 0.002),
                                                      axis="X", **common)}[s["type"]]()


def place_of(L):
    """Object position in the env frame: d past the table edge, y to the robot's left (the robot faces +y: left is -x)."""
    return (L["ax"] - L["y"], L["edge"] + L["d"], TOP + L["spec"]["z"] + 0.001)


# one object per env, each its own size (and shape with --shape a,b,...): spawned in env order. Before 2026-10-22 every
# env spawned env_0's object, so a run of 128 was one size 128 times.
cfg.scene.object = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Object",
                                  spawn=sim_utils.MultiAssetSpawnerCfg(assets_cfg=[obj_cfg(L["spec"]) for L in layouts], random_choice=False),
                                  init_state=RigidObjectCfg.InitialStateCfg(pos=place_of(layouts[0])))
env = gym.make(TASK, cfg=cfg).unwrapped
robot, obj = env.scene["robot"], env.scene["object"]
dev, envo = env.device, env.scene.env_origins
hb = robot.body_names.index(g1.HAND_BODY)
lw = robot.body_names.index("left_wrist_yaw_link")
pinch_local = torch.tensor(g1.PINCH, dtype=torch.float32, device=dev)
H = torch.tensor(np.column_stack([g1.DOWN, g1.CLOSE, np.cross(g1.DOWN, g1.CLOSE)]), dtype=torch.float32, device=dev)
FINGERS = env.cfg.actions.upper_body_ik.hand_joint_names  # the action's 14 finger targets, in this order
open_q = dict(zip(g1.HAND, [g1.THUMB_0, *g1.THUMB[-1, 1:], *g1.INDEX, *g1.MIDDLE]))
closed_q = dict(zip(g1.HAND, [g1.THUMB_0, *g1.THUMB[0, 1:], *g1.INDEX, *g1.MIDDLE]))


def fingers(f):
    """14 finger targets: right hand f of the way from open to closed, left hand at 0."""
    return torch.tensor([float(open_q[j] + f * (closed_q[j] - open_q[j])) if j in open_q else 0.0 for j in FINGERS],
                        device=dev).expand(N, len(FINGERS))


env.reset()
EDGE = torch.tensor([L["edge"] for L in layouts], device=dev)  # table A near edge, env y
A_X = torch.tensor([L["ax"] for L in layouts], device=dev)
B_TURN = torch.tensor([L["b_turn"] for L in layouts], device=dev)  # turn after backing off (rad, + = left)
B_DIST = torch.tensor([L["b_dist"] for L in layouts], device=dev)  # turn spot to the near edge of table B
B_YAW = np.pi / 2 + B_TURN  # heading toward table B (the robot starts facing +y, yaw pi/2)
TURN_PT = torch.stack([A_X, EDGE - args.stand - args.back], 1)  # where the robot turns (nominal)
B_DIR = torch.stack([torch.cos(B_YAW), torch.sin(B_YAW)], 1)
B_SIDE = torch.stack([-B_DIR[:, 1], B_DIR[:, 0]], 1)  # along the long side of table B
B_EDGE = TURN_PT + B_DIST[:, None] * B_DIR  # centre of the near edge of table B, env xy


def pose_fixture(name, xy, yaw, z):
    p = torch.zeros(N, 7, device=dev)
    p[:, :2], p[:, 2] = xy, z
    p[:, 3], p[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
    p[:, :3] += envo
    env.scene[name].write_root_pose_to_sim(p)
    env.scene[name].write_root_velocity_to_sim(torch.zeros(N, 6, device=dev))


if args.rand_layout:  # tables per env: A facing the robot; B with its long side across the walk to it (its x = B_DIR)
    zero = torch.zeros(N, device=dev)
    pose_fixture("packing_table", torch.stack([A_X, EDGE + 0.3], 1), zero, TOP - 0.02)
    pose_fixture("pedestal", torch.stack([A_X, EDGE + 0.425], 1), zero, (TOP - 0.04) / 2)
    if args.carry:
        pose_fixture("table_b", B_EDGE + 0.3 * B_DIR, B_YAW, TOP - 0.02)
        pose_fixture("pedestal_b", B_EDGE + 0.425 * B_DIR, B_YAW, (TOP - 0.04) / 2)
# place each env's object at its layout pose
pose = torch.zeros(N, 7, device=dev)
for i, L in enumerate(layouts):
    pose[i, :3] = torch.tensor(place_of(L), device=dev)
    a = L["heading"]
    pose[i, 3:] = torch.tensor([np.cos(a / 2), 0, 0, np.sin(a / 2)], device=dev)
pose[:, :3] += envo
obj.write_root_pose_to_sim(pose)
obj.write_root_velocity_to_sim(torch.zeros(N, 6, device=dev))

rp, rq = robot.data.root_pos_w, robot.data.root_quat_w
rel = [subtract_frame_transforms(rp, rq, robot.data.body_pos_w[:, w], robot.data.body_quat_w[:, w]) for w in (lw, hb)]
start_xy = rp[:, :2].clone()
obj_start_walk = None
fell = torch.zeros(N, dtype=torch.bool, device=dev)
arrived = torch.zeros(N, dtype=torch.bool, device=dev)
arrive_t = torch.full((N,), float("nan"), device=dev)
dt = env.step_dt
rec = []


def act(right=None, f=0.0, vx=None, yaw=None):
    """One env step: left wrist (and right, unless given as env-local pos, quat) held relative to the pelvis."""
    rp, rq = robot.data.root_pos_w, robot.data.root_quat_w
    upper = []
    for k, (pos, quat) in enumerate(rel):
        if k == 1 and right is not None:
            upper += list(right)
        else:
            tp, tq = combine_frame_transforms(rp, rq, pos, quat)
            upper += [tp - envo, tq]
    upper.append(fingers(f))
    v = torch.zeros(N, 4, device=dev)
    v[:, 3] = args.hip_height
    if vx is not None:
        v[:, 0] = vx
    if yaw is not None:
        v[:, 2] = yaw
    env.step(torch.cat(upper + [v], 1))
    fell.__ior__(robot.data.root_pos_w[:, 2] < 0.5)
    if TRACK["phase"]:  # --carry: first phase in which a held object left the hand (> 5 cm from the pinch point)
        gone = (torch.linalg.norm(obj.data.root_pos_w - pinch_w(), dim=1) > 0.05) & TRACK["held"]
        for i in torch.nonzero(gone).flatten().tolist():
            TRACK["drop_in"][i] = TRACK["phase"]
        TRACK["held"] &= ~gone
    if args.record:
        rec.append(torch.cat([robot.data.root_pos_w - envo, robot.data.root_quat_w, robot.data.joint_pos,
                              obj.data.root_pos_w - envo, obj.data.root_quat_w], 1).cpu().numpy().astype(np.float32))


TRACK = dict(phase="", held=None, drop_in=[""] * N)
rel0 = [(pos.clone(), quat.clone()) for pos, quat in rel]
# Parked wrists in the pelvis frame: forward kinematics of Menagerie's G1 at shoulder pitch 0.1, roll +-0.2, elbow 1.2
# (arms hanging, slightly bent; the wrists stay 13 cm behind the table edge, the fingers below its top)
PARK = [(torch.tensor([0.036, 0.221, -0.066], device=dev), torch.tensor([0.7924, 0.0773, 0.6017, 0.0636], device=dev)),
        (torch.tensor([0.036, -0.221, -0.066], device=dev), torch.tensor([0.7924, -0.0773, 0.6017, -0.0636], device=dev))]


def to_park(k_steps, which=(0, 1), f=0.0):
    """Wrists from where they are now (relative to the pelvis) to PARK over k_steps; rel then holds them there."""
    rp, rq = robot.data.root_pos_w, robot.data.root_quat_w
    bodies = (lw, hb)
    start = [subtract_frame_transforms(rp, rq, robot.data.body_pos_w[:, bodies[w]], robot.data.body_quat_w[:, bodies[w]])
             for w in which]
    for k in range(k_steps):
        s = (k + 1) / k_steps
        for (p0, q0), w in zip(start, which):
            pp, pq = PARK[w]
            q = torch.where(((q0 * pq).sum(1, keepdim=True) < 0), -q0, q0)  # same hemisphere, then lerp + normalise
            q = (1 - s) * q + s * pq
            rel[w] = (p0 + s * (pp - p0), q / torch.linalg.norm(q, dim=1, keepdim=True))
        act(f=f, vx=0.0)


if args.arms == "park":
    to_park(75)  # 1.5 s, while the objects settle
else:
    for k in range(50):  # objects settle; both hands go up over 0.5 s (they ride at table height otherwise)
        for (pos, _), (p0, _) in zip(rel, rel0):
            pos[:] = p0
            pos[:, 2] += args.carry_up * min(1.0, (k + 1) / 50)
        act(vx=0.0)
obj_start_walk = obj.data.root_pos_w.clone()
# --- walk ---------------------------------------------------------------------------------------------------------------
t = 0.0
while t < WALK_MAX and not bool(arrived.all()):
    gap = EDGE - (robot.data.root_pos_w[:, 1] - envo[:, 1])  # pelvis to table edge along the walk
    newly = (~arrived) & (gap <= args.stand)
    arrive_t[newly] = t
    arrived |= newly
    act(vx=torch.where(arrived, 0.0, args.vx))
    t += dt
walk_s = t
gap_end = (EDGE - (robot.data.root_pos_w[:, 1] - envo[:, 1])).clone()

# --- grasp (g1_lift.py's phases), goals from where the robot and object are now ----------------------------------------
z0 = obj.data.root_pos_w[:, 2].clone()
up = torch.tensor([0.0, 0.0, 1.0], device=dev)
goal_c = obj.data.root_pos_w - envo
goal_c[:, 2] = torch.clamp(goal_c[:, 2], min=TOP + g1.Hand.MIN_Z)
R0 = matrix_from_quat(robot.data.body_quat_w[:, hb]).cpu().numpy()
goal_q = []
for i, L in enumerate(layouts):
    s, a = L["spec"], L["heading"]
    h = a if s["type"] != "box" or s["size"][0] <= s["size"][1] else a + np.pi / 2  # across the narrow side
    best = None
    for sgn in (1, -1):
        cw = sgn * np.array([np.cos(h), np.sin(h), 0.0])
        W = np.column_stack([[0, 0, -1.0], cw, np.cross([0, 0, -1.0], cw)])
        R = W @ H.cpu().numpy().T
        sc = np.trace(R0[i].T @ R)
        if best is None or sc > best[0]:
            best = (sc, R)
    goal_q.append(quat_from_matrix(torch.tensor(best[1], dtype=torch.float32)[None])[0].tolist())
goal_q = torch.tensor(goal_q, device=dev)


def wrist_for(pinch, q=None):  # env-local wrist pose that puts the pinch point at `pinch` (env-local), hand turned q
    q = goal_q if q is None else q
    return pinch - quat_apply(q, pinch_local.expand(N, 3)), q


edges = np.cumsum([0] + [d for _, d in PHASES])
names = [n for n, _ in PHASES]
track = []
hand0 = robot.data.body_pos_w[:, hb] + quat_apply(robot.data.body_quat_w[:, hb], pinch_local.expand(N, 3)) - envo
start_obj = obj.data.root_pos_w.clone()
start_quat = obj.data.root_quat_w.clone()
moved_in = ["" for _ in range(N)]  # first phase in which the object moved > 2 cm (before the close): what knocked it
for k in range(int(round(edges[-1] / dt))):
    tt = k * dt
    j = int(np.searchsorted(edges, tt, side="right") - 1)
    ph, frac = names[j], (tt - edges[j]) / PHASES[j][1]
    s = min(1.0, frac * 1.5)
    if ph == "settle":
        act()
        if k == int(round(PHASES[0][1] / dt)) - 1:  # moved while walking or settling (before the hand comes)
            m = (torch.linalg.norm(obj.data.root_pos_w - obj_start_walk, dim=1) > 0.02).cpu().numpy()
            for i in np.flatnonzero(m):
                moved_in[i] = moved_in[i] or "walk"
        continue
    pinch, f = {"raise": (hand0 + s * (goal_c + args.high * up - hand0), 0.0),  # up and over first: keeps the hand off the table
                "approach": (goal_c + args.high * up + s * (0.08 - args.high) * up, 0.0), "descend": (goal_c + (1 - s) * 0.08 * up, 0.0),
                "close": (goal_c, min(1.0, frac)), "lift": (goal_c + s * LIFT * up, 1.0),
                "hold": (goal_c + LIFT * up, 1.0)}[ph]
    act(right=wrist_for(pinch), f=f)
    if ph in ("raise", "approach", "descend"):
        m = (torch.linalg.norm(obj.data.root_pos_w - start_obj, dim=1) > 0.02).cpu().numpy()
        for i in np.flatnonzero(m):
            moved_in[i] = moved_in[i] or ph
    if ph == "descend" and frac > 0.95:
        pw = robot.data.body_pos_w[:, hb] + quat_apply(robot.data.body_quat_w[:, hb], pinch_local.expand(N, 3))
        track.append(torch.linalg.norm(pw - envo - goal_c, dim=1))

# --- carry and put down (--carry) ----------------------------------------------------------------------------------------
def yaw_of(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def pinch_w():
    return robot.data.body_pos_w[:, hb] + quat_apply(robot.data.body_quat_w[:, hb], pinch_local.expand(N, 3))


def heading_err():  # to each env's heading toward table B (fixed layout: yaw 0, facing +x)
    return torch.remainder(B_YAW - yaw_of(robot.data.root_quat_w) + np.pi, 2 * np.pi) - np.pi


def carry_phases():
    """Back off, turn right, walk to table B, put the object down, step back. Returns per-env stage results."""
    fell0 = fell.clone()
    TRACK.update(phase="raise", held=torch.tensor(lifted, device=dev))
    # right hand held relative to the pelvis at its commanded hold pose, then raised
    wp, wq = wrist_for(goal_c + LIFT * up)
    hold = list(subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, wp + envo, wq))
    h0 = hold[0].clone()
    rel[1] = hold
    for k in range(50):  # up --carry_raise and in to --carry_x (closer to the body), hand turned as it held
        s = (k + 1) / 50
        hold[0][:] = h0
        hold[0][:, 2] += args.carry_raise * s
        if args.arms == "park":
            hold[0][:, 0] = h0[:, 0] + s * (torch.clamp(h0[:, 0], max=args.carry_x) - h0[:, 0])
        act(f=1.0, vx=0.0)
    TRACK["phase"] = "back off"
    y0 = robot.data.root_pos_w[:, 1].clone()
    t, done = 0.0, torch.zeros(N, dtype=torch.bool, device=dev)
    while t < 4.0 and not bool(done.all()):
        done |= (y0 - robot.data.root_pos_w[:, 1]) >= args.back
        act(f=1.0, vx=torch.where(done, 0.0, -0.25))
        t += dt
    TRACK["phase"] = "turn"
    t, done = 0.0, torch.zeros(N, dtype=torch.bool, device=dev)
    turn_t = torch.full((N,), float("nan"), device=dev)
    while t < 8.0 and not bool(done.all()):
        err = heading_err()
        newly = (~done) & (err.abs() < 0.2)  # stops ~0.15 rad later; the walk's heading hold takes out the rest
        turn_t[newly] = t
        done |= newly
        act(f=1.0, vx=0.0, yaw=torch.where(done, 0.0, torch.sign(err) * args.yaw_rate))  # bang-bang: below ~1 rad/s it stands
        t += dt
    turned = done & ~fell
    for _ in range(int(0.5 / dt)):
        act(f=1.0, vx=0.0)
    TRACK["phase"] = "walk to B"
    t, done = 0.0, torch.zeros(N, dtype=torch.bool, device=dev)
    walk_t = torch.full((N,), float("nan"), device=dev)
    while t < 6.0 and not bool(done.all()):
        gap = ((B_EDGE - (robot.data.root_pos_w[:, :2] - envo[:, :2])) * B_DIR).sum(1)  # to the edge, along the heading
        newly = (~done) & (gap <= args.stand)
        walk_t[newly] = t
        done |= newly
        act(f=1.0, vx=torch.where(done, 0.0, args.vx), yaw=torch.where(done, 0.0, torch.clamp(1.5 * heading_err(), -0.3, 0.3)))
        t += dt
    at_b = done.clone()
    err_b = heading_err()
    for _ in range(int(1.0 / dt)):
        act(f=1.0, vx=0.0)
    TRACK["phase"] = "put down"
    carried = (torch.linalg.norm(obj.data.root_pos_w - pinch_w(), dim=1) < 0.05) & at_b & ~fell
    # put down: the object's centre to its resting height on B, place_d past the edge, at its current sideways spot
    o = obj.data.root_pos_w - envo
    off = o - (pinch_w() - envo)  # object centre relative to the pinch point, as held
    rest_z = torch.tensor([L["spec"]["z"] for L in layouts], device=dev)
    lat = ((o[:, :2] - B_EDGE) * B_SIDE).sum(1, keepdim=True)  # keep the sideways spot along table B
    tgt = torch.cat([B_EDGE + args.place_d * B_DIR + lat * B_SIDE, (TOP + rest_z + 0.005)[:, None]], 1) - off
    p_start, q_hold = pinch_w() - envo, robot.data.body_quat_w[:, hb].clone()
    above = tgt + 0.05 * up
    for name, dur in [("move", 1.5), ("lower", 1.0), ("rest", args.settle_place), ("open", 0.8), ("clear", 1.0)]:
        n = int(round(dur * (args.slow_place if name != "rest" else 1.0) / dt))
        if n == 0:
            continue
        for k in range(n):
            s = min(1.0, (k + 1) / n * 1.5)
            pinch, f = {"move": (p_start + s * (above - p_start), 1.0), "lower": (above + s * (tgt - above), 1.0),
                        "rest": (tgt, 1.0), "open": (tgt, 1.0 - (k + 1) / n), "clear": (tgt + s * 0.10 * up, 0.0)}[name]
            if name == "open":
                TRACK["phase"] = ""  # the release is not a drop
            act(right=wrist_for(pinch, q_hold), f=f, vx=0.0)
    TRACK["phase"] = ""
    # step back, the hand held where it ended relative to the pelvis (park: back to the side first)
    rel[1] = list(subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w,
                                            robot.data.body_pos_w[:, hb], robot.data.body_quat_w[:, hb]))
    if args.arms == "park":
        to_park(50, which=(1,))
    for _ in range(int(1.5 / dt)):
        act(vx=-0.2)
    for _ in range(int(1.0 / dt)):
        act(vx=0.0)
    o = obj.data.root_pos_w - envo
    along, across = ((o[:, :2] - B_EDGE) * B_DIR).sum(1), ((o[:, :2] - B_EDGE) * B_SIDE).sum(1)
    in_b = (along > 0) & (along < 0.6) & (across.abs() < B_W / 2) & (o[:, 2] > TOP) & (o[:, 2] < TOP + 0.12)
    still = torch.linalg.norm(obj.data.root_lin_vel_w, dim=1) < 0.05
    free = torch.linalg.norm(obj.data.root_pos_w - pinch_w(), dim=1) > 0.04
    on_b = (in_b
            & (torch.linalg.norm(obj.data.root_lin_vel_w, dim=1) < 0.05)
            & (torch.linalg.norm(obj.data.root_pos_w - pinch_w(), dim=1) > 0.04))
    zq = quat_apply(obj.data.root_quat_w, up.expand(N, 3))
    z0 = quat_apply(start_quat, up.expand(N, 3))
    tilt = torch.rad2deg(torch.arccos(torch.clamp((zq * z0).sum(1), -1, 1)))
    ball = torch.tensor([L["spec"]["type"] == "sphere" for L in layouts], device=dev)
    # lying shapes (axis along y or x: lying cylinder, capsule, egg) roll about their own axis and stay the same shape on
    # the table: for them "upright" = their long axis still horizontal (start axis tilt measured about z is meaningless)
    axis = [L["spec"].get("quat_axis") or ("x" if L["spec"]["type"] == "ellipsoid" else None) for L in layouts]
    lying = torch.tensor([a is not None for a in axis], device=dev)
    ax = torch.tensor([[1.0, 0, 0] if a == "x" else [0, 1.0, 0] for a in axis], device=dev)
    flat = (quat_apply(obj.data.root_quat_w, ax).abs() * up).sum(1) < 0.34  # long axis within 20 deg of horizontal
    c = lambda x: x.cpu().numpy()  # noqa: E731
    return dict(turned=[bool(v) for v in c(turned)], carried=[bool(v) for v in c(carried)],
                placed=[bool(v) for v in c(on_b)], upright=[bool(v) for v in c((ball | (lying & flat) | (~lying & (tilt < 20))) & on_b)],
                in_b=[bool(v) for v in c(in_b)], still=[bool(v) for v in c(still)], free=[bool(v) for v in c(free)],
                standing=[bool(v) for v in c(~fell)], heading_at_b=[round(float(v), 3) for v in c(err_b)], fell_carry=[bool(v) for v in c(fell & ~fell0)],
                turn_s=[round(float(v), 2) for v in c(turn_t)], walk_b_s=[round(float(v), 2) for v in c(walk_t)],
                tilt=[round(float(v), 1) for v in c(tilt)], drop_in=TRACK["drop_in"], end_obj=[[round(float(a), 3) for a in v] for v in c(o)])


# --- outcome -------------------------------------------------------------------------------------------------------------
pw = robot.data.body_pos_w[:, hb] + quat_apply(robot.data.body_quat_w[:, hb], pinch_local.expand(N, 3))
rise = (obj.data.root_pos_w[:, 2] - z0).cpu().numpy()
near = torch.linalg.norm(obj.data.root_pos_w - pw, dim=1).cpu().numpy()
reach_err = (torch.stack(track).mean(0) if track else torch.full((N,), float("nan"), device=dev)).cpu().numpy()
fell_grasp = fell.clone()
lifted = [bool(rise[i] > LIFT - 0.01 and near[i] < 0.05 and not fell[i]) for i in range(N)]
carry = carry_phases() if args.carry else None
rows = [dict(shape=L["shape"], seed=L["seed"], fell=bool(fell_grasp[i]), arrived=bool(arrived[i]),
             arrive_s=round(float(arrive_t[i]), 2), stop_gap=round(float(gap_end[i]), 3),
             success=bool(rise[i] > LIFT - 0.01 and near[i] < 0.05 and not fell_grasp[i]),
             held=bool(rise[i] > 0.05 and near[i] < 0.05 and not fell_grasp[i]),
             moved_in=moved_in[i], rise=round(float(rise[i]), 4), near=round(float(near[i]), 4), reach_err=round(float(reach_err[i]), 4),
             **({k: v[i] for k, v in carry.items()} if carry else {}))
        for i, L in enumerate(layouts)]
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
path = out / f"walk-{'carry' if args.carry else 'grasp'}-n{args.n}-{'-'.join(shapes)}{'-hm' if args.hand_drives == 'matched' else ''}{args.tag}.json"
path.write_text(json.dumps(dict(args=vars(args), rows=rows), indent=1))
print(f"{path}: {N} robots; walked {walk_s:.1f} s; arrived {int(arrived.sum())}; fell {int(fell.sum())}; "
      f"stop gap median {np.median([r['stop_gap'] for r in rows]) * 100:.1f} cm (target {args.stand * 100:.0f})")
from collections import Counter  # noqa: E402
print(f"  objects moved > 2 cm before the close, first in: {dict(Counter(r['moved_in'] for r in rows if r['moved_in']))}")
print(f"  pinch-to-object error at the end of the descent: median {np.nanmedian(reach_err) * 100:.1f} cm")
for s in shapes:
    r = [x for x in rows if x["shape"] == s]
    print(f"  {s:10s} lifted {sum(x['success'] for x in r)}/{len(r)}  held {sum(x['held'] for x in r)}/{len(r)}")
print(f"  all        lifted {sum(x['success'] for x in rows)}/{N}; held {sum(x['held'] for x in rows)}/{N}", flush=True)
if carry:
    ok = [r["success"] for r in rows]
    for k in ("turned", "carried", "placed", "upright", "standing"):
        ok = [a and r[k] for a, r in zip(ok, rows)]
        print(f"  carry: lifted and ... {k:9s} {sum(ok)}/{N}   ({k} alone {sum(r[k] for r in rows)}/{N})")
    print(f"  carry: robots fell after the grasp {sum(r['fell_carry'] for r in rows)}; "
          f"turn s median {np.nanmedian([r['turn_s'] for r in rows]):.1f}; "
          f"walk to B s median {np.nanmedian([r['walk_b_s'] for r in rows]):.1f}", flush=True)
    print(f"  carry: lifted objects dropped first in: {dict(Counter(r['drop_in'] for r in rows if r['success'] and r['drop_in']))}", flush=True)
if args.record:
    np.savez_compressed(path.with_suffix(".npz"), frames=np.stack(rec), joint_names=np.array(robot.joint_names), dt=dt,
                        specs=json.dumps([dict(shape=L["shape"], type=L["spec"]["type"], size=list(map(float, L["spec"]["size"])),
                                               rgba=list(map(float, L["spec"]["rgba"])), quat_axis=L["spec"].get("quat_axis")) for L in layouts]),
                        table=np.array([0.0, EDGE_Y + 0.3, TOP]),
                        layout=json.dumps([dict(edge=L["edge"], ax=L["ax"], b_turn=L["b_turn"], b_dist=L["b_dist"]) for L in layouts]),
                        table_b=np.array([B_X + 0.3, TURN_Y, TOP] if args.carry else [np.nan] * 3))
    print("recorded", path.with_suffix(".npz"))
sys.stdout.flush()
os._exit(0)
