"""The G1 lift test in Isaac Lab, Isaac only (owner 2026-10-07: "just test the g1 in isaac, we are not going to use mujoco
for it anymore"): the scripted top-down pinch planned and run in Isaac, the baseline later RL work compares against.

Layouts: the held-out lift test's (6 shapes x 60 layouts, seeds 900000+, so101/objects.py random_objects + layout: one
primitive on the table in front of the robot), all in parallel, one per environment. Robot: Unitree's fixed-base
G1-29dof + Dex3 USD (unitree_sim_isaaclab), pelvis behind the table with the table top at hip height (the old scene's
geometry: g1.PELVIS, TABLE_EDGE), gravity off on the robot (stands in for a humanoid controller's gravity feed-forward).
Changed from Unitree's config: waist free to move (their velocity limit 0 froze it); hand drives as in the MuJoCo model
(kp 500, kv 4.5, armature 0.01, 1.4 N m, thumb_0 2.45; Unitree's 300 N m is far above the real Dex3's, and soft drives,
stiffness 20, dropped the lift from 35 to 1 of 60).
PhysX contact settings for the objects (2026-10-07 sweep, n=10): torsional friction patch 1 cm (MuJoCo used condim 6),
friction combine "max", 32 solver iterations; without them the pinch squeezes objects out.

Grasp (scripted, per environment, all at once): the pinch point (between thumb and index pads, g1.PINCH in the hand frame)
goes 8 cm above the object's centre, then down to it (at least 1.2 cm above the table), the hand pointing down with its
closing direction across the object's narrow side (of its two signs, the smaller turn from the start orientation);
close over 1.2 s (thumb to closed, torque-limited), lift 10 cm, hold 1 s. Arm and
waist by Isaac Lab's differential IK (damped least squares) on the hand frame. Success, as before: the object at least
9 cm above its start and within 5 cm of the pinch point at the end.

  python isaac/g1_lift.py [--n=60] [--shapes=box,ball] [--video]     -> runs/isaac/g1-lift-n<n>.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=60)
p.add_argument("--shapes", default="box,cylinder,lying cylinder,ball,capsule,egg")
p.add_argument("--unitree", default="/opt/dlami/nvme/w/unitree_sim_isaaclab")
p.add_argument("--out", default="runs/isaac")
p.add_argument("--no-self-collide", action="store_true",
               help="Unitree's default: the arms can pass through the body (on by default since 2026-10-07)")
p.add_argument("--record", action="store_true", help="also save every layout's motion (joints, object pose per 10 ms) for isaac/render_mj.py")
p.add_argument("--friction", type=float, default=0.0, help="override every object's friction (0: the layout's own)")
p.add_argument("--camera-collision", action="store_true",
               help="keep the hand camera mount's collider (Unitree's 145 mm box overlaps the palm: with self-collision on it fights the hand)")
p.add_argument("--torsional", type=float, default=0.01,
               help="objects' torsional friction patch radius, m (PhysX; stands in for MuJoCo's condim 6)")
p.add_argument("--iters", type=int, default=32, help="objects' solver position iterations (0: PhysX default)")
p.add_argument("--robot-iters", type=int, default=0, help="robot's solver position iterations (0: Unitree's USD)")
p.add_argument("--dt", type=float, default=0.005, help="physics step, s (control every 10 ms regardless)")
p.add_argument("--close", type=float, default=1.2, help="close phase, s (fingers ramp open -> closed over all of it)")
p.add_argument("--depen", type=float, default=0.0, help="objects' max depenetration velocity, m/s (0: PhysX default)")
p.add_argument("--soft-hand", action="store_true", help="old hand drives (stiffness 20, damping 1) instead of MuJoCo's")
p.add_argument("--combine", default="max", help="PhysX friction combine mode for the objects: average | max | multiply")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import ContactSensorCfg  # noqa: E402
from isaaclab.utils.math import quat_apply, matrix_from_quat, quat_apply_inverse, quat_from_matrix, quat_mul, subtract_frame_transforms  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["PROJECT_ROOT"] = args.unitree
sys.path.insert(0, args.unitree)
from robots.unitree import G129_CFG_WITH_DEX3_BASE_FIX  # noqa: E402
from so101 import g1  # noqa: E402  (constants only: pelvis, table edge, hand frame axes, finger poses)
from so101.objects import layout, random_objects  # noqa: E402

DT = args.dt  # physics step; control every 10 ms
DECIM = max(1, round(0.01 / DT))
PHASES = [("approach", 1.5), ("descend", 1.0), ("close", args.close), ("lift", 1.0), ("hold", 1.0)]
TEST_SEED, LIFT = 900000, 0.10
dev = "cuda:0"

# --- layouts -------------------------------------------------------------------------------------------------------------
shapes = args.shapes.split(",")
layouts = []
for s in shapes:
    for i in range(args.n):
        rng = np.random.default_rng(TEST_SEED + i)
        spec = random_objects(rng, 1, [s])[0]
        x, y, a = layout(rng, [spec])[0]
        layouts.append(dict(shape=s, seed=TEST_SEED + i, spec=spec, x=x, y=y, heading=a))
N = len(layouts)

# --- robot ---------------------------------------------------------------------------------------------------------------
ARM, HAND = list(g1.ARM), list(g1.HAND)
open_q = dict(zip(HAND, [g1.THUMB_0, *g1.THUMB[-1, 1:], *g1.INDEX, *g1.MIDDLE]))  # opening 1.0
closed_q = dict(zip(HAND, [g1.THUMB_0, *g1.THUMB[0, 1:], *g1.INDEX, *g1.MIDDLE]))  # opening 0: thumb to the index pad
start_q = {k: float(v) for k, v in (dict(zip(ARM, g1.READY.tolist())) | dict(g1.LEFT_ARM) | open_q).items()}
cfg = G129_CFG_WITH_DEX3_BASE_FIX.replace(prim_path="/World/envs/env_.*/Robot")
cfg.spawn.rigid_props.disable_gravity = True
cfg.spawn.articulation_props.enabled_self_collisions = not args.no_self_collide  # Unitree's USD config: off
if args.robot_iters:
    cfg.spawn.articulation_props.solver_position_iteration_count = args.robot_iters
cfg.init_state = cfg.init_state.replace(pos=tuple(g1.PELVIS.tolist()), joint_pos=cfg.init_state.joint_pos | start_q)
cfg.actuators["waist"] = cfg.actuators["waist"].replace(velocity_limit=None, velocity_limit_sim=None, effort_limit=None,
                                                        effort_limit_sim=1000.0, stiffness=3000.0, damping=60.0)
if args.soft_hand:
    cfg.actuators["hands"] = cfg.actuators["hands"].replace(effort_limit=None, effort_limit_sim=g1.THUMB_FORCE,
                                                            stiffness=20.0, damping=1.0)
else:  # the MuJoCo model's hand drives (Menagerie g1_with_hands + g1.prepare): kp 500, kv 4.5, armature 0.01
    cfg.actuators["hands"] = cfg.actuators["hands"].replace(
        effort_limit=None, effort_limit_sim={".*_hand_thumb_0_joint": 2.45, ".*_hand_(thumb_[12]|index_.|middle_.)_joint": g1.THUMB_FORCE},
        stiffness=500.0, damping=4.5, armature=0.01)

scene_cfg = InteractiveSceneCfg(num_envs=N, env_spacing=2.0, replicate_physics=False)
scene_cfg.robot = cfg
cfg.spawn.activate_contact_sensors = True
scene_cfg.fingers = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/right_hand_(palm|thumb|index|middle).*", update_period=0.0, history_length=1)
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=dev))
scene = InteractiveScene(scene_cfg)
origins = scene.env_origins.cpu().numpy()

TABLE_POS = np.array([g1.PELVIS[0] + g1.TABLE_EDGE + 0.4, 0.0, -0.5])
table = sim_utils.CuboidCfg(size=(0.8, 1.2, 1.0), collision_props=sim_utils.CollisionPropertiesCfg(),
                            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.8),
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.5, 0.45)))
shelf = sim_utils.CuboidCfg(size=(0.08, 0.12, 0.06), collision_props=sim_utils.CollisionPropertiesCfg(),
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.65, 0.5)))


def yaw_quat(a):
    return [np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)]


LIE = [np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0]  # objects.build: lying shapes' long axis along the heading
for i, L in enumerate(layouts):
    # translations are in the env prim's frame (it already sits at its origin)
    table.func(f"/World/envs/env_{i}/Table", table, translation=tuple(TABLE_POS.tolist()))
    shelf.func(f"/World/envs/env_{i}/Shelf", shelf, translation=(0.32, 0.0, 0.03))
    s = L["spec"]
    common = dict(rigid_props=sim_utils.RigidBodyPropertiesCfg(**({"solver_position_iteration_count": args.iters,
                                                                    "solver_velocity_iteration_count": 1} if args.iters else {}),
                                                                 **({"max_depenetration_velocity": args.depen} if args.depen else {})),
                  mass_props=sim_utils.MassPropertiesCfg(mass=s["mass"]),
                  collision_props=sim_utils.CollisionPropertiesCfg(**({"torsional_patch_radius": args.torsional,
                                                                       "min_torsional_patch_radius": args.torsional / 2}
                                                                      if args.torsional else {})),
                  physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=args.friction or s["friction"],
                                                                  dynamic_friction=args.friction or s["friction"],
                                                                  restitution=0.0, friction_combine_mode=args.combine),
                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(s["rgba"][:3])))
    z = s["size"]
    obj = {"box": lambda: sim_utils.CuboidCfg(size=(2 * z[0], 2 * z[1], 2 * z[2]), **common),
           "cylinder": lambda: sim_utils.CylinderCfg(radius=z[0], height=2 * z[1], axis="Z", **common),
           "capsule": lambda: sim_utils.CapsuleCfg(radius=z[0], height=2 * z[1], axis="Z", **common),
           "sphere": lambda: sim_utils.SphereCfg(radius=z[0], **common),
           "ellipsoid": lambda: sim_utils.MeshSphereCfg(radius=1.0, **common)}[s["type"]]()
    q = torch.tensor(yaw_quat(L["heading"]))
    if s["quat_axis"] == "y":
        q = quat_mul(q[None], torch.tensor(LIE)[None])[0]
    prim = obj.func(f"/World/envs/env_{i}/Object", obj, translation=(L["x"], L["y"], s["z"] + 0.001),
                    orientation=tuple(q.tolist()))
    if s["type"] == "ellipsoid":  # unit mesh sphere scaled per axis (the spawner already wrote a scale op)
        from pxr import Gf
        at = prim.GetAttribute("xformOp:scale")
        at.Set(Gf.Vec3d(*z) if "double" in str(at.GetTypeName()) else Gf.Vec3f(*z))

if not args.camera_collision:  # default since 2026-10-07: off (MuJoCo's model has no camera mount)
    from pxr import UsdPhysics
    import omni.usd
    off = 0
    stage = omni.usd.get_context().get_stage()
    for i in range(N):  # deactivate the mount's collider subtree (instanced, so its shapes cannot be edited one by one)
        for side in ("left", "right"):
            prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/Robot/{side}_hand_camera_base_link/collisions")
            if prim.IsValid():
                prim.SetActive(False)
                off += 1
    print(f"camera mount colliders off: {off}")

obj = RigidObject(RigidObjectCfg(prim_path="/World/envs/env_.*/Object", spawn=None))
scene.rigid_objects["object"] = obj
robot: Articulation = scene["robot"]
sim.reset()
scene.update(DT)
# the object view's order need not be env_0, env_1, ... (prim paths sort as text): index it back into environment order
paths = list(obj.root_physx_view.prim_paths)
perm = torch.tensor([paths.index(f"/World/envs/env_{i}/Object") for i in range(N)], device=dev)


def obj_pos():
    return obj.data.root_pos_w[perm]

# --- grasp targets (env frame) -------------------------------------------------------------------------------------------
jn = robot.joint_names
arm_ids = [jn.index(j) for j in ARM]
hand_ids = [jn.index(j) for j in HAND]
hb = robot.body_names.index(g1.HAND_BODY)
jac_idx = hb - 1 if robot.is_fixed_base else hb
ik = DifferentialIKController(DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
                              num_envs=N, device=dev)
pinch_local = torch.tensor(g1.PINCH, dtype=torch.float32, device=dev)
H = torch.tensor(np.column_stack([g1.DOWN, g1.CLOSE, np.cross(g1.DOWN, g1.CLOSE)]), dtype=torch.float32, device=dev)


def narrow_heading(L):
    """Direction (radians) across the object's narrowest horizontal width: the pinch closes along it."""
    s, a = L["spec"], L["heading"]
    if s["type"] == "box":
        return a if s["size"][0] <= s["size"][1] else a + np.pi / 2
    if s["quat_axis"] == "y" or s["type"] == "ellipsoid":  # long axis along the heading
        return a + np.pi / 2
    return a  # round in plan: any


goal_q, goal_c, goal_cw = [], [], []
R0 = matrix_from_quat(robot.data.body_quat_w[:, hb]).cpu().numpy()  # each hand's start orientation (world)
for i, L in enumerate(layouts):
    c = np.array([L["x"], L["y"], max(L["spec"]["z"], g1.Hand.MIN_Z)])
    h = narrow_heading(L)
    best = None
    for sgn in (1, -1):  # the closing direction's two signs: keep the one closer to the hand's start orientation
        cw = sgn * np.array([np.cos(h), np.sin(h), 0.0])
        dw = np.array([0.0, 0.0, -1.0])
        W = np.column_stack([dw, cw, np.cross(dw, cw)])
        R = W @ H.cpu().numpy().T
        score = np.trace(R0[i].T @ R)  # cos of the rotation from the start orientation, up to scale: smaller turn wins
        if best is None or score > best[0]:
            best = (score, R, cw)
    goal_cw.append(best[2].tolist())
    goal_q.append(quat_from_matrix(torch.tensor(best[1], dtype=torch.float32)[None])[0].tolist())
    goal_c.append(c.tolist())
goal_q = torch.tensor(goal_q, device=dev)
goal_c = torch.tensor(goal_c, device=dev)
env_o = scene.env_origins


def hand_goal(pinch_env):
    """Hand-body pose (world) that puts the pinch point at `pinch_env` (env frame) with the goal orientation."""
    pos = pinch_env + env_o - quat_apply(goal_q, pinch_local.expand(N, 3))
    return pos, goal_q


def step_ik(target_pos, target_q):
    root_p, root_q = robot.data.root_pos_w, robot.data.root_quat_w
    ee_p, ee_q = robot.data.body_pos_w[:, hb], robot.data.body_quat_w[:, hb]
    tp, tq = subtract_frame_transforms(root_p, root_q, target_pos, target_q)
    ep, eq = subtract_frame_transforms(root_p, root_q, ee_p, ee_q)
    ik.set_command(torch.cat([tp, tq], 1))
    jac = robot.root_physx_view.get_jacobians()[:, jac_idx, :, arm_ids]
    return ik.compute(ep, eq, jac, robot.data.joint_pos[:, arm_ids])


def hand_target(f):
    """Fingers at fraction f of the way from open (0) to closed (1)."""
    o = torch.tensor([float(open_q[j]) for j in HAND], dtype=torch.float32, device=dev)
    c = torch.tensor([float(closed_q[j]) for j in HAND], dtype=torch.float32, device=dev)
    return (o + f * (c - o)).expand(N, len(HAND))


z0 = obj_pos()[:, 2].clone()
sensor = scene["fingers"]
fnames = sensor.body_names
touch = torch.zeros(N, len(fnames), device=dev)  # largest contact force on each right-hand link during close + lift, N
start_err = torch.linalg.norm(obj_pos()[:, :2] - env_o[:, :2] - goal_c[:, :2], dim=1).cpu().numpy()  # spawn check
trace = {}  # per phase end: object and pinch point, env frame
up = torch.tensor([0.0, 0.0, 1.0], device=dev)
t_total = sum(d for _, d in PHASES)
steps = int(round(t_total / (DT * DECIM)))
t = 0.0
rec_q, rec_o = [], []
for k in range(steps):
    t = k * DT * DECIM
    edges = np.cumsum([0] + [d for _, d in PHASES])
    ph = next(n for (n, _), lo, hi in zip(PHASES, edges[:-1], edges[1:]) if lo <= t < hi)
    frac = (t - edges[[n for n, _ in PHASES].index(ph)]) / dict(PHASES)[ph]
    s = min(1.0, frac * 1.5)  # reach each waypoint in the first two thirds of its phase
    if ph == "approach":
        pinch, f = goal_c + 0.08 * up, 0.0
    elif ph == "descend":
        pinch, f = goal_c + (1 - s) * 0.08 * up, 0.0
    elif ph == "close":
        pinch, f = goal_c, min(1.0, frac * (1.2 if args.close == 0.6 else 1.0))
    elif ph == "lift":
        pinch, f = goal_c + s * LIFT * up, 1.0
    else:
        pinch, f = goal_c + LIFT * up, 1.0
    tp, tq = hand_goal(pinch)
    q_arm = step_ik(tp, tq)
    robot.set_joint_position_target(q_arm, joint_ids=arm_ids)
    robot.set_joint_position_target(hand_target(f), joint_ids=hand_ids)
    for _ in range(DECIM):
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(DT)
    if args.record:
        rec_q.append(robot.data.joint_pos.half().cpu().numpy())
        rec_o.append(torch.cat([obj_pos() - env_o, obj.data.root_quat_w[perm]], 1).cpu().numpy())
    if ph in ("close", "lift"):
        touch = torch.maximum(touch, torch.linalg.norm(sensor.data.net_forces_w, dim=-1))
    if k + 1 == steps or ph != next(n for (n, _), lo, hi in zip(PHASES, edges[:-1], edges[1:]) if lo <= t + DT * DECIM < hi):
        pw = robot.data.body_pos_w[:, hb] + quat_apply(robot.data.body_quat_w[:, hb], pinch_local.expand(N, 3))
        hp, hq = robot.data.body_pos_w[:, hb], robot.data.body_quat_w[:, hb]
        trace[ph] = dict(obj=(obj_pos() - env_o).cpu().numpy(), pinch=(pw - env_o).cpu().numpy(),
                         hand=robot.data.joint_pos[:, hand_ids].cpu().numpy(),
                         obj_in_hand=quat_apply_inverse(hq, obj_pos() - hp).cpu().numpy(),  # compare g1.PINCH
                         down=quat_apply(hq, H[:, 0].expand(N, 3)).cpu().numpy(),  # want (0, 0, -1)
                         close=quat_apply(hq, H[:, 1].expand(N, 3)).cpu().numpy())  # want goal_cw

# --- outcome -------------------------------------------------------------------------------------------------------------
op = obj_pos()
pinch_w = robot.data.body_pos_w[:, hb] + quat_apply(robot.data.body_quat_w[:, hb], pinch_local.expand(N, 3))
rise = (op[:, 2] - z0).cpu().numpy()
near = torch.linalg.norm(op - pinch_w, dim=1).cpu().numpy()
goal_err = torch.linalg.norm(pinch_w - (goal_c + LIFT * up + env_o), dim=1).cpu().numpy()  # IK tracking check
rows = [dict(shape=L["shape"], seed=L["seed"], success=bool(rise[i] > LIFT - 0.01 and near[i] < 0.05),
             held=bool(rise[i] > 0.05 and near[i] < 0.05),  # in the hand, at least 5 cm up (success needs 9 cm)
             rise=round(float(rise[i]), 4), near=round(float(near[i]), 4), pinch_err=round(float(goal_err[i]), 4),
             start_err=round(float(start_err[i]), 4), goal_cw=np.round(goal_cw[i], 3).tolist(),
             touch={n: round(float(touch[i, j]), 2) for j, n in enumerate(fnames)},
             trace={ph: {k: np.round(v[i], 3).tolist() for k, v in d.items()} for ph, d in trace.items()})
        for i, L in enumerate(layouts)]
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
path = out / f"g1-lift-n{args.n}{'-f%g' % args.friction if args.friction else ''}{'-' + args.combine if args.combine != 'max' else ''}{'-cam' if args.camera_collision else ''}{'-t%g' % args.torsional if args.torsional != 0.01 else ''}{'-i%d' % args.iters if args.iters != 32 else ''}{'-ri%d' % args.robot_iters if args.robot_iters else ''}{'-dt%g' % args.dt if args.dt != 0.005 else ''}{'-c%g' % args.close if args.close != 1.2 else ''}{'-d%g' % args.depen if args.depen else ''}{'-soft' if args.soft_hand else ''}{'-noself' if args.no_self_collide else ''}.json"
path.write_text(json.dumps(dict(phases=PHASES, rows=rows), indent=1))
print(f"{path}: {N} layouts")
if args.record:
    np.savez_compressed(path.with_suffix(".npz"), q=np.stack(rec_q), obj=np.stack(rec_o).astype(np.float32),
                        joint_names=np.array(robot.joint_names), dt=DT * DECIM, pelvis=g1.PELVIS,
                        table_pos=TABLE_POS, table_size=np.array([0.8, 1.2, 1.0]),
                        specs=json.dumps([dict(shape=L["shape"], seed=L["seed"], type=L["spec"]["type"],
                                               size=list(map(float, L["spec"]["size"])),
                                               rgba=list(map(float, L["spec"]["rgba"])), success=r["success"], held=r["held"])
                                          for L, r in zip(layouts, rows)]))
    print("recorded", path.with_suffix(".npz"))
for s in shapes:
    r = [x for x in rows if x["shape"] == s]
    print(f"  {s:15s} lifted {sum(x['success'] for x in r):3d}/{len(r)}  median pinch error at the end "
          f"{np.median([x['pinch_err'] for x in r]) * 100:.1f} cm")
print(f"  all             lifted {sum(x['success'] for x in rows)}/{N}; held (5 cm up, in the hand) {sum(x['held'] for x in rows)}/{N}")
sys.stdout.flush()
os._exit(0)  # app.close() hangs in Isaac Sim 5.1 headless
