"""Does Isaac Lab's ready-made walk-and-grasp G1 walk? Isaac-PickPlace-Locomanipulation-G1-Abs-v0: Unitree's 29-dof G1 with
Dex3 hands (43 joints, 34.2 kg), legs driven by NVIDIA's pretrained AGILE walking policy (actions: vx, vy, yaw rate, hip
height), arms by pink IK (actions: both wrist poses, --frame base or world, then 14 finger targets). The wrists are held at
their start pose relative to the pelvis, so the arms ride along; fingers open. For each forward-speed command: falls
(pelvis below 0.5 m), achieved forward speed over the second half, and wrist drift from the held pose.

  python isaac/locomanip_walk.py [--num_envs=16] [--seconds=10] [--speeds=0,0.3,0.6]     (from the IsaacLab folder)
"""
import argparse
import os
import sys

import pinocchio  # noqa: F401  (before AppLauncher: Isaac Lab's pinocchio, not Isaac Sim's)

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=16)
p.add_argument("--seconds", type=float, default=10.0)
p.add_argument("--speeds", default="0,0.5,-0.3")
p.add_argument("--cmds", default="", help="vx:vy:wz commands instead of --speeds, comma separated (turning check)")
p.add_argument("--frame", default="world", help="wrist targets: base (pelvis frame) or world")
p.add_argument("--record", default="", help="save env 0's motion (pelvis pose, joints) per step to this .npz, for isaac/render_mj.py")
p.add_argument("--hip_height", type=float, default=0.72)  # the teleop retargeters' default
args = p.parse_args()
app = AppLauncher(headless=True).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: E402,F401  (blacklisted by default)
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "Isaac-PickPlace-Locomanipulation-G1-Abs-v0"
cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=args.num_envs)
cfg.episode_length_s = args.seconds + 5
cfg.terminations.object_dropping = None  # only the time limit: falls are counted here
cfg.terminations.success = None
cfg.scene.packing_table = None  # nothing to walk into (the object stays: the observations read it; it falls to the floor)
env = gym.make(TASK, cfg=cfg).unwrapped
robot = env.scene["robot"]
wrists = [robot.body_names.index(n) for n in ("left_wrist_yaw_link", "right_wrist_yaw_link")]
N, dev = env.num_envs, env.device
steps = int(args.seconds / env.step_dt)

cmds = ([tuple(float(x) for x in c.split(":")) for c in args.cmds.split(",")] if args.cmds
        else [(float(v), 0.0, 0.0) for v in args.speeds.split(",")])
for vx, vy, wz in cmds:
    env.reset()
    rp, rq = robot.data.root_pos_w, robot.data.root_quat_w
    rel = [subtract_frame_transforms(rp, rq, robot.data.body_pos_w[:, w], robot.data.body_quat_w[:, w]) for w in wrists]
    start = robot.data.root_pos_w[:, :2].clone()
    fell = torch.zeros(N, dtype=torch.bool, device=dev)
    speed, drift, rec, yawr = [], [], [], []
    for k in range(steps):
        rp, rq = robot.data.root_pos_w, robot.data.root_quat_w
        upper = []
        for pos, quat in rel:
            tp, tq = combine_frame_transforms(rp, rq, pos, quat)  # world, then env-local: the IK subtracts env origins
            upper += [tp - env.scene.env_origins, tq] if args.frame == "world" else [pos, quat]
        upper.append(torch.zeros(N, 14, device=dev))  # fingers at 0 (open)
        lower = torch.tensor([vx, vy, wz, args.hip_height], device=dev).expand(N, 4)
        env.step(torch.cat(upper + [lower], 1))
        fell |= robot.data.root_pos_w[:, 2] < 0.5
        if args.record:
            rec.append(torch.cat([robot.data.root_pos_w[0] - env.scene.env_origins[0], robot.data.root_quat_w[0],
                                  robot.data.joint_pos[0]]).cpu().numpy())
        if k >= steps // 2:
            speed.append(robot.data.root_lin_vel_b[:, 0].clone())
            yawr.append(robot.data.root_ang_vel_b[:, 2].clone())
            for (pos, quat), w in zip(rel, wrists):
                now, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w,
                                                   robot.data.body_pos_w[:, w], robot.data.body_quat_w[:, w])
                drift.append(torch.linalg.norm(now - pos, dim=1))
    up = ~fell
    sp = torch.stack(speed).mean(0)
    yr = torch.stack(yawr).mean(0)
    print(f"[vx {vx:.2f} vy {vy:.2f} wz {wz:.2f}] achieved yaw rate {float(yr[up].mean()) if up.any() else float('nan'):.2f} rad/s")
    print(f"[vx {vx:.1f} m/s] {N} robots, {args.seconds:.0f} s: fell {int(fell.sum())}; achieved forward speed "
          f"{float(sp[up].mean()) if up.any() else float('nan'):.2f} m/s; moved {float(torch.linalg.norm(robot.data.root_pos_w[:, :2] - start, dim=1).mean()):.2f} m; wrist drift {float(torch.stack(drift).mean()) * 100:.1f} cm",
          flush=True)
    if args.record:
        import numpy as np
        out = args.record.replace(".npz", f"-vx{vx:g}.npz")
        np.savez_compressed(out, frames=np.stack(rec), joint_names=np.array(robot.joint_names), dt=env.step_dt,
                            label=f"walk command {vx:g} m/s")
        print("recorded", out, flush=True)
sys.stdout.flush()
os._exit(0)
