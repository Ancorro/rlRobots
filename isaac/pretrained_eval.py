"""Does Isaac Lab's published G1 walking checkpoint walk? Loads it (play.py's --use_pretrained_checkpoint source) into an
rsl_rl runner, runs the task's play config for --seconds, and counts falls (termination before the time limit). Isaac
Lab's G1 asset carries three-finger hands, so a pass here is a walking start that already has hands.

  python isaac/pretrained_eval.py --task Isaac-Velocity-Flat-G1-Play-v0 [--num_envs 512] [--seconds 20]
  (from the IsaacLab folder)
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", default="Isaac-Velocity-Flat-G1-Play-v0")
p.add_argument("--num_envs", type=int, default=512)
p.add_argument("--seconds", type=float, default=20.0)
p.add_argument("--add_mass", type=float, default=0.0, help="kg added to each body matching --mass_bodies at startup")
p.add_argument("--mass_bodies", default=".*_wrist_yaw_link")
p.add_argument("--checkpoint", default="", help="a local .pt instead of the published one")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

train_task = args.task.replace("-Play", "")
env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
env_cfg.episode_length_s = args.seconds
if args.add_mass:
    from isaaclab.managers import EventTermCfg, SceneEntityCfg
    import isaaclab.envs.mdp as mdp
    env_cfg.events.extra_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=args.mass_bodies),
                "mass_distribution_params": (args.add_mass, args.add_mass), "operation": "add"})
agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
ckpt = args.checkpoint or get_published_pretrained_checkpoint("rsl_rl", train_task)
print(f"checkpoint: {ckpt}")
env = gym.make(args.task, cfg=env_cfg)
robot = env.unwrapped.scene["robot"]
print(f"robot: {robot.num_joints} joints ({sum('hand' in n or 'zero' in n or 'one' in n or 'two' in n for n in robot.joint_names)} "
      f"look like hand joints), mass {float(robot.root_physx_view.get_masses()[0].sum()):.1f} kg")
venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(venv, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
runner.load(ckpt)
policy = runner.get_inference_policy(device="cuda:0")
obs = venv.get_observations()
steps = int(args.seconds / env.unwrapped.step_dt)
fresh = torch.ones(args.num_envs, dtype=torch.bool, device="cuda:0")
fell = torch.zeros_like(fresh)
up = torch.zeros(args.num_envs, device="cuda:0")
for _ in range(steps):
    with torch.inference_mode():
        obs, _, dones, extras = venv.step(policy(obs))
    timeouts = extras.get("time_outs", torch.zeros_like(dones)).bool()
    d = dones.bool()
    fell |= d & ~timeouts & fresh
    up += fresh.float()
    fresh &= ~d
print(f"[{args.task} +{args.add_mass} kg on {args.mass_bodies}] {args.num_envs} robots, {args.seconds:.0f} s: fell {int(fell.sum())} ({100 * fell.float().mean():.1f}%), "
      f"mean time up {float(up.mean()) * env.unwrapped.step_dt:.1f} s")
sys.stdout.flush()
os._exit(0)
