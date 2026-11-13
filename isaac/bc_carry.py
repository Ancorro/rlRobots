"""BC pick, scripted walk while holding, BC again to place: our own walking step around the Plan #2 BC policy.

Stock task Isaac-PickPlace-Locomanipulation-G1-Abs-v0 (steering wheel, one table). Each rollout:
  bc     the robomimic policy runs until the object is LIFT m above its start height;
  carry  the wrist targets and finger targets are frozen relative to the pelvis (the IK action takes wrist poses in
         the env frame, so holding the raw action would pin the hands in space while the body walks away), and the
         AGILE legs get the --walk velocity segments, hip height kept from BC;
  resume BC again (RNN state kept, or reset with --reset_rnn) until the task's success check or the horizon.
Action layout: left wrist pos+quat (7), right wrist (7), 14 hand joints, lower body [vx, vy, wz, hip_height] (4).

  /isaac-sim/python.sh .../bc_carry.py --checkpoint ... --norm_factor_min ... --norm_factor_max ... --out x.json
"""
import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-PickPlace-Locomanipulation-G1-Abs-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--norm_factor_min", type=float, required=True)
parser.add_argument("--norm_factor_max", type=float, required=True)
parser.add_argument("--num_rollouts", type=int, default=10)
parser.add_argument("--seed", type=int, default=101)
parser.add_argument("--lift", type=float, default=0.06, help="hand-off height above the start, m")
parser.add_argument("--bc_max", type=int, default=400, help="steps allowed to reach the lift")
parser.add_argument("--resume_max", type=int, default=400, help="steps allowed to place after the walk")
parser.add_argument("--walk", default="-0.2,0,0,2;0,0,0,1;0.2,0,0,2;0,0,0,1",
                    help="segments vx,vy,wz,seconds separated by ';' (default: back 0.4 m and return)")
parser.add_argument("--reset_rnn", action="store_true", help="restart the BC policy's RNN state after the walk")
parser.add_argument("--out", required=True)
parser.add_argument("--enable_pinocchio", default=False, action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.enable_pinocchio:
    import pinocchio  # noqa: F401  (Isaac Lab's pinocchio before the app's)
simulation_app = AppLauncher(args).app

import copy  # noqa: E402
import random  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import robomimic.utils.file_utils as FileUtils  # noqa: E402
import robomimic.utils.torch_utils as TorchUtils  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as mu  # noqa: E402
import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

UP, HANDS = 14, 14  # wrist pose block, hand joints; lower body is the last 4


def bc_action(policy, obs_dict, env):
    obs = copy.deepcopy(obs_dict["policy"])
    for k in obs:
        obs[k] = torch.squeeze(obs[k])
    a = policy(obs)
    a = (a + 1) * (args.norm_factor_max - args.norm_factor_min) / 2 + args.norm_factor_min
    return torch.from_numpy(a).to(env.device).view(1, -1).float()


def pelvis(env, idx):
    s = env.scene["robot"].data.body_link_state_w[:, idx, :7]
    return s[:, :3] - env.scene.env_origins, s[:, 3:7]


def rollout(policy, env, success_term, walk, dt, idx):
    policy.start_episode()
    obs_dict, _ = env.reset()
    obj = env.scene["object"]
    z0 = float(obj.data.root_pos_w[0, 2])
    r = dict(lifted=False, held_after_walk=False, success=False, dropped=False, bc_steps=0, walked_m=0.0,
             min_obj_z_walk=None, success_step=None)

    def step(a):
        nonlocal obs_dict
        obs_dict, _, term, trunc, _ = env.step(a)
        return bool(term[0] or trunc[0])

    a = None
    for i in range(args.bc_max):  # bc until lifted
        a = bc_action(policy, obs_dict, env)
        if step(a):
            r["dropped"] = True
            return r
        if float(obj.data.root_pos_w[0, 2]) - z0 > args.lift:
            r["lifted"], r["bc_steps"] = True, i + 1
            break
    if not r["lifted"]:
        return r

    # carry: wrist targets relative to the pelvis, fingers and hip height frozen
    pp, pq = pelvis(env, idx)
    rel = []
    for k in (0, 7):
        rel.append(mu.subtract_frame_transforms(pp, pq, a[:, k:k + 3], a[:, k + 3:k + 7]))
    hands, hip = a[:, UP:UP + HANDS].clone(), a[:, -1:].clone()
    start, minz = pp[0, :2].clone(), 9.0
    for vx, vy, wz, sec in walk:
        for _ in range(max(1, round(sec / dt))):
            pp, pq = pelvis(env, idx)
            wr = [torch.cat(mu.combine_frame_transforms(pp, pq, p, q), dim=-1) for p, q in rel]
            cmd = torch.tensor([[vx, vy, wz]], device=env.device)
            a = torch.cat([wr[0], wr[1], hands, cmd, hip], dim=-1)
            done = step(a)
            minz = min(minz, float(obj.data.root_pos_w[0, 2]))
            r["walked_m"] = max(r["walked_m"], float(torch.linalg.norm(pp[0, :2] - start)))
            if done:
                r["dropped"], r["min_obj_z_walk"] = True, minz
                return r
    r["min_obj_z_walk"] = minz
    r["held_after_walk"] = float(obj.data.root_pos_w[0, 2]) - z0 > args.lift / 2

    if args.reset_rnn:
        policy.start_episode()
    for i in range(args.resume_max):  # bc places
        if step(bc_action(policy, obs_dict, env)):
            r["dropped"] = True
            return r
        if bool(success_term.func(env, **success_term.params)[0]):
            r["success"], r["success_step"] = True, i + 1
            return r
    return r


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    cfg.observations.policy.concatenate_terms = False
    cfg.terminations.time_out = None
    cfg.recorders = None
    success_term = cfg.terminations.success
    cfg.terminations.success = None
    env = gym.make(args.task, cfg=cfg).unwrapped
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    env.seed(args.seed)
    walk = [tuple(float(x) for x in seg.split(",")) for seg in args.walk.split(";")]
    idx = env.scene["robot"].find_bodies("pelvis")[0][0]
    dt = env.step_dt
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    rows = []
    for t in range(args.num_rollouts):
        policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=args.checkpoint, device=device)
        r = rollout(policy, env, success_term, walk, dt, idx)
        rows.append(r)
        print(f"[bc_carry] trial {t}: {r}", flush=True)
    keys = ("lifted", "held_after_walk", "success", "dropped")
    summary = {k: sum(r[k] for r in rows) for k in keys}
    summary.update(n=len(rows), walk=args.walk, reset_rnn=args.reset_rnn)
    print(f"[bc_carry] summary {summary}", flush=True)
    with open(args.out, "w") as f:
        json.dump(dict(summary=summary, rows=rows, args=vars(args)), f, indent=1, default=str)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
