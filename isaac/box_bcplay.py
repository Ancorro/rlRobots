"""Box BC (behaviour cloning) rollouts that say how far each one gets (owner 2026-10-21: the standing grab-and-hold
network scored 0/20 with Isaac Lab's play.py, which prints only success). Same loop as play.py (robomimic policy,
actions unnormalised with the run's min/max, the env's success term checked each step), plus per rollout: how close
the wrists got to the box sides, how high the box rose, how long it stayed up, where it ended.

  /isaac-sim/python.sh box_tasks.py box_bcplay.py --checkpoint m.pth --norm_factor_min a --norm_factor_max b --out r.json
      --task Isaac-PickPlace-Locomanipulation-G1-Box-Abs-v0 --headless --enable_pinocchio
"""
import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--norm_factor_min", type=float, required=True)
parser.add_argument("--norm_factor_max", type=float, required=True)
parser.add_argument("--num_rollouts", type=int, default=20)
parser.add_argument("--horizon", type=int, default=800)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--out", required=True)
parser.add_argument("--record", action="store_true", help="save each rollout's motion next to --out (.npz) for\n"
                    "isaac/render_mj.py boxstand: per step the pelvis pose, every joint, the box pose")
parser.add_argument("--handoff", type=int, default=0, help="> 0: the script takes over (moves the box over the goal, lowers it,\n"
                    "opens, backs the hands off, as box_scripted.py) once the box is up 5 cm and either within --near of the goal or\n"
                    "up for this many steps")
parser.add_argument("--clean_reset", type=int, default=20, help="> 0: after each reset, hold the arms where they are this many\n"
                    "steps, then put the box back at its start at rest (half the rollouts flung the box in the first 5 steps without it)")
parser.add_argument("--jitter", type=float, default=0.0, help="box start shift, +- m in x and y (needs --clean_reset)")
parser.add_argument("--near", type=float, default=0.04, help="handoff distance, box centre to goal centre in x-y (m)")
parser.add_argument("--slow", type=float, default=2.0, help="script moves take this many times box_scripted.py's step counts")
parser.add_argument("--dataset", required=True, help="the wheel demos, for the open finger angles")
parser.add_argument("--ik", default="pink", help="batch: isaac/batch_ik.py on the GPU (--device cuda:0), as batch-IK demos were recorded")
parser.add_argument("--enable_pinocchio", default=False, action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.enable_pinocchio:
    import pinocchio  # noqa: F401
simulation_app = AppLauncher(args).app

import copy  # noqa: E402

import gymnasium as gym  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import robomimic.utils.file_utils as FileUtils  # noqa: E402
import robomimic.utils.torch_utils as TorchUtils  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: E402,F401
import bench_spec  # noqa: E402  (the shared success tests: upright, released)
from box_cfgs import BELOW, SIZE  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

DEV = args.device if args.ik == "batch" else "cpu"  # Pink needs the CPU
cfg = parse_env_cfg(args.task, device=DEV, num_envs=1)
if args.ik == "batch":
    from batch_ik import use_batch_ik

    use_batch_ik(cfg)
cfg.observations.policy.concatenate_terms = False
cfg.terminations.time_out = None
cfg.recorders = None
success = cfg.terminations.success
cfg.terminations.success = None
env = gym.make(args.task, cfg=cfg).unwrapped
torch.manual_seed(args.seed)
np.random.seed(args.seed)
env.seed(args.seed)
robot, box, o = env.scene["robot"], env.scene["object"], env.scene.env_origins[0]
device = TorchUtils.get_torch_device(try_to_use_cuda=True)
GOAL = np.array(env.cfg.box_goal_centre)
with h5py.File(args.dataset, "r") as f:
    OPEN = f["data"]["demo_0"]["actions"][0, 14:28].copy()  # box_scripted.py's open fingers


def pose(name):  # a link's position and orientation (w, x, y, z), env frame: an action's wrist target
    i = robot.data.body_names.index(name)
    return np.concatenate([(robot.data.body_pos_w[0, i] - o).cpu().numpy(), robot.data.body_quat_w[0, i].cpu().numpy()])


def nearest():  # the robot link closest to the box centre, and how far (m)
    d = np.linalg.norm((robot.data.body_pos_w[0] - o).cpu().numpy() - centre(), axis=1)
    return robot.data.body_names[int(d.argmin())], round(float(d.min()), 3)


def centre():
    return (box.data.root_pos_w[0] - o).cpu().numpy() + np.array([0, 0, SIZE[2] / 2 - BELOW])


def wrist(name):
    return (robot.data.body_pos_w[0, robot.data.body_names.index(name)] - o).cpu().numpy()


def box_state():
    """The box's tilt from vertical (deg), upright and released by bench_spec's rules (released: the nearer wrist clear of
    the box's side faces; no contact sensor, so within 1 cm counts as touching)."""
    q = box.data.root_quat_w[0].cpu().numpy()
    tilt = bench_spec.tilt_deg([2 * (q[1] * q[3] + q[0] * q[2]), 2 * (q[2] * q[3] - q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)],
                               [0.0, 0.0, 1.0])
    c = centre()
    gap = min(np.linalg.norm((wrist(f"{s}_wrist_yaw_link") - c)[:2]) for s in ("left", "right")) - SIZE[0] / 2 - 0.01
    return tilt, bench_spec.upright(tilt, "standing"), bench_spec.released(gap, gap < 0.01)


def script_tail(a, c, rest):
    """Actions from the network's last one: move both wrists so the box is over the goal (as high as it is), lower it
    to its resting height, open, move the hands 10 cm out, back to the start pose, wait (box_scripted.py's carry..wait)."""
    a = a.copy()
    L, R, hand = a[0:3].copy(), a[7:10].copy(), a[14:28].copy()
    moves = []

    def go(n, dL, dR, h=None):  # straight-line moves from where the last one ended
        nonlocal L, R, hand
        n = int(n * args.slow)
        L0, R0, h0 = L.copy(), R.copy(), hand.copy()
        for k in range(1, n + 1):
            f = k / n
            b = a.copy()
            b[0:3], b[7:10] = L0 + f * dL, R0 + f * dR
            b[14:28] = h0 if h is None else h0 + f * (h - h0)
            moves.append(b)
        L, R = L0 + dL, R0 + dR
        if h is not None:
            hand = h

    over = np.array([GOAL[0] - c[0], GOAL[1] - c[1], 0.0])
    go(60, over, over)
    down = np.array([0, 0, -(c[2] - GOAL[2] - 0.01)])
    go(40, down, down)
    go(15, np.zeros(3), np.zeros(3), OPEN)
    go(30, np.array([-0.10, 0, 0]), np.array([0.10, 0, 0]))
    go(40, rest[0:3] - L, rest[7:10] - R)
    go(60, np.zeros(3), np.zeros(3))
    return moves


rows, recs = [], []
for t in range(args.num_rollouts):
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=args.checkpoint, device=device)
    policy.start_episode()
    obs_dict, _ = env.reset()
    if args.clean_reset:  # arms settle at their reset pose (wrist targets = where the wrists are), then the box goes
        # back to its start at rest: the first network step flung the box in half the rollouts without it
        hold = np.concatenate([pose("left_wrist_yaw_link"), pose("right_wrist_yaw_link"), OPEN, [0, 0, 0, 0.72]])
        for _ in range(args.clean_reset):
            env.step(torch.tensor(hold, dtype=torch.float32, device=DEV).view(1, -1))
        st = box.data.default_root_state.clone()
        st[:, :3] += env.scene.env_origins
        st[:, :2] += torch.tensor(np.random.uniform(-args.jitter, args.jitter, 2), dtype=torch.float32, device=DEV)
        box.write_root_pose_to_sim(st[:, :7])
        box.write_root_velocity_to_sim(torch.zeros(1, 6, device=DEV))
        env.sim.forward()
        env.scene.update(dt=0.0)
        obs_dict = {"policy": env.observation_manager.compute()["policy"]}
    c0 = centre()
    w0 = [wrist("left_wrist_yaw_link").round(3).tolist(), wrist("right_wrist_yaw_link").round(3).tolist()]
    r = dict(wrists_at_reset=w0, trial=t, success=False, steps=0, max_rise=0.0, steps_up=0, ended="horizon",
             closest_wrist=9.0, start=c0.round(3).tolist(), handoff=None)
    plan, a, rest = [], None, None
    frames = []
    for i in range(args.horizon):
        if plan:  # the script's moves, after the handoff
            a = plan.pop(0)
        else:
            obs = {k: torch.squeeze(v) for k, v in copy.deepcopy(obs_dict["policy"]).items()}
            a = policy(obs)
            a = (a + 1) * (args.norm_factor_max - args.norm_factor_min) / 2 + args.norm_factor_min
            if rest is None:
                rest = a.copy()  # the network's first wrist targets: the arms' start pose
        obs_dict, _, term, trunc, _ = env.step(torch.from_numpy(np.asarray(a)).view(1, -1).float().to(DEV))
        r["steps"] = i + 1
        if args.record:
            frames.append(torch.cat([robot.data.root_pos_w[0] - o, robot.data.root_quat_w[0], robot.data.joint_pos[0],
                                     box.data.root_pos_w[0] - o, box.data.root_quat_w[0]]).cpu().numpy().astype(np.float32))
        c = centre()
        if i in (0, 1, 2, 5, 20):  # the first steps: where the box and wrists are right after the reset
            r.setdefault("early", []).append([i, c.round(3).tolist(), wrist("left_wrist_yaw_link").round(3).tolist(),
                                              wrist("right_wrist_yaw_link").round(3).tolist(), nearest()])
        rise = c[2] - c0[2]
        r["max_rise"] = max(r["max_rise"], float(rise))
        r["steps_up"] += int(rise > 0.05)
        # wrist distance to where the script puts it: beside the box side, so 0 = in grip position
        d = [abs(abs(wrist(f"{s}_wrist_yaw_link")[0] - c[0]) - SIZE[0] / 2) for s in ("left", "right")]
        r["closest_wrist"] = min(r["closest_wrist"], float(max(d)))
        if args.handoff and r["handoff"] is None and rise > 0.05:
            near = np.linalg.norm(c[:2] - GOAL[:2]) < args.near
            if near or r["steps_up"] >= args.handoff:
                r["handoff"], r["handoff_by"], r["handoff_box"] = i + 1, "near" if near else "steps", c.round(3).tolist()
                plan = script_tail(a, c, rest)
        if bool(success.func(env, **success.params)[0]):
            r["success"], r["ended"] = True, "success"
            break
        if bool(term[0] or trunc[0]):
            tm = env.termination_manager
            r["ended"] = [n for n in tm.active_terms if bool(tm.get_term(n)[0])] or "truncated"
            break
    r["end"] = centre().round(3).tolist()
    r["tilt"], r["upright"], r["released"] = box_state()  # the env's success term alone passed tipped or still-held boxes
    r["tilt"] = round(float(r["tilt"]), 1)
    r["env_success"], r["success"] = r["success"], bool(r["success"] and r["upright"] and r["released"])
    r["fell"] = bool(float(robot.data.root_pos_w[0, 2] - o[2]) < 0.4)
    r["max_rise"] = round(r["max_rise"], 3)
    r["closest_wrist"] = round(r["closest_wrist"], 3)
    rows.append(r)
    if args.record:
        recs.append(np.stack(frames))
    print(f"[box_bcplay] {r}", flush=True)
summary = dict(n=len(rows), success=sum(r["success"] for r in rows), lifted=sum(r["max_rise"] > 0.05 for r in rows),
               held_100=sum(r["steps_up"] >= 100 for r in rows), gripped=sum(r["closest_wrist"] < 0.03 for r in rows),
               handoffs=sum(r["handoff"] is not None for r in rows))
print(f"[box_bcplay] summary {summary}", flush=True)
json.dump(dict(summary=summary, args=vars(args), rows=rows), open(args.out, "w"), indent=1)
if args.record:  # rollouts differ in length: one array each
    np.savez_compressed(args.out.replace(".json", ".npz"), joint_names=np.array(robot.joint_names), dt=env.step_dt,
                        size=np.array(SIZE), below=BELOW, goal=GOAL, **{f"r{i}": f for i, f in enumerate(recs)})
env.close()
simulation_app.close()
