"""Scripted two-hand box pick and place for the G1 (owner 2026-10-21: "Scripted grasp", after Mimic copies of the wheel's
one-hand rim grip never lifted a box). Both palms squeeze the box's left and right sides, lift, carry it sideways,
set it down, open and back off. The script reads the box's real pose each step (closed loop), so its successful runs
are demos for the same BC (behaviour cloning) training as the wheel; --record writes them as an Isaac Lab hdf5.

--envs N runs N robots side by side in one simulation (owner 2026-10-21: concurrent batches): every copy follows the
same schedule of moves, each aimed at its own box, and a copy's run is saved the step it succeeds.

Actions (the stock task's): left wrist pos + quat (w, x, y, z), right wrist pos + quat, 14 hand joints, lower body
(vx, vy, yaw rate, height); the robot stands still. Wrist targets are in the env frame, the robot at the origin
facing +y. Box from isaac/box_cfgs.py (BOX_SIZE, BOX_POS, BOX_GOAL), its frame 0.48 cm above its bottom.

  /isaac-sim/python.sh box_scripted.py --trials 64 --envs 16 --headless --enable_pinocchio [--record out.hdf5]
"""
import argparse
import json
import math
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--trials", type=int, default=4, help="trials per wrist roll")
parser.add_argument("--envs", type=int, default=1, help="robots simulated side by side")
parser.add_argument("--roll", default="0", help="comma list: degrees both wrists turn about the forward axis from rest")
parser.add_argument("--palm", type=float, default=0.012, help="wrist centre to finger surface, sideways (m)")
parser.add_argument("--thumb", type=float, default=0.075, help="wrist sits this far in front of the box's near face (m): the\n"
                    "open thumb sticks 6 cm inward 6.4 cm ahead of the wrist, so it goes on the near face, fingers on the sides")
parser.add_argument("--squeeze", type=float, default=0.02, help="targets this far inside the box sides (m)")
parser.add_argument("--close", type=float, default=0.4, help="fingers, 0 open .. 1 the demos' closed grip")
parser.add_argument("--jitter", type=float, default=0.03, help="box start shift, +- m in x and y")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--ik", default="pink", help="pink: the stock arm IK, one robot at a time on the CPU; batch: isaac/batch_ik.py,\n"
                    "all robots at once, so the sim can run on the GPU (--device cuda:0) with many more robots")
parser.add_argument("--slow", type=float, default=2.0, help="every move takes this many times its base step count")
parser.add_argument("--grip_below", type=float, default=0.05, help="hands grip this far below the box top (m)")
parser.add_argument("--phases", action="store_true", help="print box and wrists of robot 0 after each move")
parser.add_argument("--dataset", default="", help="the wheel demos, for the open and closed finger angles")
parser.add_argument("--record", default="", help="write successful runs to this hdf5")
parser.add_argument("--out", default="")
parser.add_argument("--enable_pinocchio", default=False, action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.enable_pinocchio:
    import pinocchio  # noqa: F401
simulation_app = AppLauncher(args).app

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg  # noqa: E402
from isaaclab.managers import DatasetExportMode  # noqa: E402
from isaaclab.utils.math import quat_apply_inverse, quat_from_angle_axis, quat_mul  # noqa: E402

import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: E402,F401
from batch_ik import use_batch_ik  # noqa: E402
from box_cfgs import BELOW, SIZE, BoxEnvCfg, BoxSizedEnvCfg  # noqa: E402

L_IDX, R_IDX = [0, 1, 2, 6, 7, 8, 12], [3, 4, 5, 9, 10, 11, 13]  # hand joints of each hand in the action's 14
N = args.envs
cfg = BoxSizedEnvCfg() if os.environ.get("BOX_SIZED") else BoxEnvCfg()  # BOX_SIZED=1: demos also record object_size
cfg.scene.num_envs = N
cfg.sim.device = args.device if args.ik == "batch" else "cpu"  # Pink needs the CPU
DEV = cfg.sim.device
if args.ik == "batch":
    use_batch_ik(cfg)
SUCCESS = cfg.terminations.success  # checked here each step, as Isaac Lab's record_demos.py does, so a success
cfg.terminations.success = None     # does not reset the env before the recorder has the episode
cfg.terminations.time_out = None        # no term may end one copy mid-schedule (the env would reset it alone);
cfg.terminations.object_dropping = None  # a dropped box is checked here instead
if args.record:
    cfg.recorders = ActionStateRecorderManagerCfg()
    cfg.recorders.dataset_export_dir_path = os.path.dirname(os.path.abspath(args.record))
    cfg.recorders.dataset_filename = os.path.splitext(os.path.basename(args.record))[0]
    cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
env = ManagerBasedRLEnv(cfg=cfg)
robot, box = env.scene["robot"], env.scene["object"]
origins = env.scene.env_origins
rng = np.random.default_rng(args.seed)

# finger angles: open from the wheel demos' first step, closed from where each hand holds the wheel
with h5py.File(args.dataset, "r") as f:
    act = f["data"]["demo_0"]["actions"][:]
OPEN = act[0, 14:28].copy()
SHUT = OPEN.copy()
SHUT[L_IDX] = act[105, 14:28][L_IDX]
SHUT[R_IDX] = act[150, 14:28][R_IDX]
HAND = OPEN + args.close * (SHUT - OPEN)


def body(name):  # every copy's link position (env frame) and orientation, (N, 3) and (N, 4)
    i = robot.data.body_names.index(name)
    return (robot.data.body_pos_w[:, i] - origins), robot.data.body_quat_w[:, i]


def show_hands():
    """Where each hand's links sit in its wrist frame: tells which way the palm faces."""
    for side in ("left", "right"):
        wp, wq = (v[0] for v in body(f"{side}_wrist_yaw_link"))
        for n in robot.data.body_names:
            if n.startswith(f"{side}_hand"):
                p = body(n)[0][0]
                print(f"BOX_SCRIPT hand {n}: in wrist frame {quat_apply_inverse(wq, p - wp).cpu().numpy().round(3).tolist()}, "
                      f"in env frame from wrist {(p - wp).cpu().numpy().round(3).tolist()}")


def turned(q, deg, axis=(0.0, 1.0, 0.0)):  # rotate wrist orientations about an env-frame axis (y = forward)
    r = quat_from_angle_axis(torch.tensor([np.radians(deg)]), torch.tensor([axis]))
    return quat_mul(r.expand(len(q), 4), torch.as_tensor(q, dtype=torch.float32)).cpu().numpy()


def box_centre():  # (N, 3), env frame
    return (box.data.root_pos_w - origins).cpu().numpy() + np.array([0, 0, SIZE[2] / 2 - BELOW])


def run(roll):
    """One round: every copy does the whole schedule once. Returns one trace per copy."""
    env.reset()
    if args.record:
        env.recorder_manager.reset()
    start = box.data.default_root_state.clone()
    start[:, :3] += origins
    start[:, 0] += torch.tensor(rng.uniform(-args.jitter, args.jitter, N), dtype=torch.float32, device=DEV)
    start[:, 1] += torch.tensor(rng.uniform(-args.jitter, args.jitter, N), dtype=torch.float32, device=DEV)
    box.write_root_pose_to_sim(start[:, :7])
    box.write_root_velocity_to_sim(torch.zeros(N, 6, device=DEV))
    lp, lq = (v.cpu().numpy() for v in body("left_wrist_yaw_link"))
    rp, rq = (v.cpu().numpy() for v in body("right_wrist_yaw_link"))
    lq, rq = turned(lq, roll), turned(rq, -roll)
    goal = np.array(cfg.box_goal_centre)
    c0 = box_centre()
    side = SIZE[0] / 2 + args.palm  # wrist distance from the box centre at contact
    pose = dict(L=lp.copy(), R=rp.copy(), hand=np.tile(OPEN, (N, 1)))
    rise, steps = np.zeros(N), 0
    won = np.zeros(N, dtype=bool)          # succeeded (and saved) at some step
    won_at = np.full(N, -1)
    dropped = np.zeros(N, dtype=bool)
    lower_body = np.tile([0, 0, 0, 0.72], (N, 1))

    def go(n, L=None, R=None, hand=None):
        nonlocal steps
        aL, aR, ah = pose["L"].copy(), pose["R"].copy(), pose["hand"].copy()
        n = int(n * args.slow)
        for k in range(1, n + 1):
            t = k / n
            if L is not None:
                pose["L"] = aL + t * (L - aL)
            if R is not None:
                pose["R"] = aR + t * (R - aR)
            if hand is not None:
                pose["hand"] = ah + t * (hand - ah)
            a = np.concatenate([pose["L"], lq, pose["R"], rq, pose["hand"], lower_body], axis=1)
            env.step(torch.tensor(a, dtype=torch.float32, device=DEV))
            steps += 1
            c = box_centre()
            rise[:] = np.maximum(rise, c[:, 2] - c0[:, 2])
            dropped[:] |= c[:, 2] < 0.5  # the stock object_dropping height
            new = SUCCESS.func(env, **SUCCESS.params).cpu().numpy() & ~won & ~dropped
            if new.any():
                ids = np.flatnonzero(new).tolist()
                won[ids], won_at[ids] = True, steps
                if args.record:  # save each copy's run the step it succeeds; what it does after is never saved
                    env.recorder_manager.record_pre_reset(ids, force_export_or_skip=False)
                    env.recorder_manager.set_success_to_episodes(ids, torch.ones(len(ids), 1, dtype=torch.bool, device=DEV))
                    env.recorder_manager.export_episodes(ids)
            if (won | dropped).all():
                return True
        return False

    def at(c, dx, dz=0.0):  # wrist targets dx outside each side of boxes centred at c, (N, 3) each
        c = np.broadcast_to(c, (N, 3))
        y = c[:, 1] - SIZE[1] / 2 - args.thumb
        z = c[:, 2] + SIZE[2] / 2 - args.grip_below + dz
        return np.stack([c[:, 0] - side - dx, y, z], 1), np.stack([c[:, 0] + side + dx, y, z], 1)

    # the approach moves re-aim at where each box is when the move starts (None: filled in then)
    steps_plan = [
        ("above", 40, None, None, None),        # above and outside the sides
        ("beside", 30, None, None, None),       # down beside them
        ("squeeze", 30, None, None, None),      # palms in, past the sides: the IK pulls against the box
        ("fingers", 20, None, None, HAND),      # fingers
        ("lift", 40, None, None, None),         # lift (aimed from the box after the squeeze)
        ("carry", 60, *at(goal, -args.squeeze, 0.12), None),
        ("lower", 40, *at(goal, -args.squeeze, 0.01), None),
        ("open", 15, None, None, OPEN),         # let go
        ("out", 30, *at(goal, 0.08, 0.01), None),
        ("back", 40, lp, rp, None),             # back to where the arms started
        ("wait", 60, None, None, None),         # wait for the success check
    ]
    aim = dict(above=(0.07, 0.10), beside=(0.07, 0.0), squeeze=(-args.squeeze, 0.0), lift=(-args.squeeze, 0.12))
    for name, n, L, R, hand in steps_plan:
        if name in aim:
            L, R = at(box_centre(), *aim[name])
        stop = go(n, L, R, None if hand is None else np.tile(hand, (N, 1)))
        if args.phases:  # robot 0: where its box and real wrists are after each move, against the wrist targets
            wl, wr, b = body("left_wrist_yaw_link")[0][0].cpu().numpy(), body("right_wrist_yaw_link")[0][0].cpu().numpy(), box_centre()[0]
            print(f"BOX_PHASE {name}: box {b.round(3).tolist()} | L {wl.round(3).tolist()} want "
                  f"{pose['L'][0].round(3).tolist()} | R {wr.round(3).tolist()} want {pose['R'][0].round(3).tolist()} | "
                  f"wrist gap {wr[0] - wl[0]:.3f}", flush=True)
        if stop:
            break
    end = box_centre()
    fell = (robot.data.root_pos_w[:, 2] - origins[:, 2]).cpu().numpy() < 0.5  # pelvis down: the robot fell
    return [dict(roll=roll, success=bool(won[i]), rise=float(rise[i]), steps=int(won_at[i]) if won[i] else steps,
                 why="success" if won[i] else "dropped" if dropped[i] else "ran out", fell=bool(fell[i]),
                 start=c0[i].round(3).tolist(), end=end[i].round(3).tolist(), goal=goal.round(3).tolist()) for i in range(N)]


torch.set_grad_enabled(False)  # the recorder cannot save tensors that track gradients (IK action terms make some)
env.reset()
show_hands()
results = []
t0 = time.time()
for roll in [float(r) for r in args.roll.split(",")]:
    for k in range(math.ceil(args.trials / N)):
        for i, r in enumerate(run(roll)[: args.trials - k * N]):
            results.append(r)
            print(f"BOX_SCRIPT roll {roll} trial {k * N + i}: success {r['success']}, rise {r['rise']:.3f} m, start {r['start']} "
                  f"end {r['end']} goal {r['goal']}, {r['steps']} steps, ended by {r['why']}", flush=True)
wall = time.time() - t0
for roll in sorted({r["roll"] for r in results}):
    rs = [r for r in results if r["roll"] == roll]
    print(f"BOX_SCRIPT roll {roll}: {sum(r['success'] for r in rs)}/{len(rs)} succeed, "
          f"{sum(r['rise'] > 0.05 for r in rs)}/{len(rs)} lifted 5 cm, {sum(r['fell'] for r in rs)}/{len(rs)} fell")
print(f"BOX_SCRIPT time: {len(results)} trials, {N} robots side by side, {wall:.0f} s, {wall / max(1, len(results)):.1f} s per trial")
if args.out:
    json.dump(dict(args=vars(args), results=results, seconds=wall), open(args.out, "w"), indent=1)
env.close()
simulation_app.close()
