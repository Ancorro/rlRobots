"""Train a robot controller with RL (reinforcement learning).

Usage:  uv run python train.py [ENV_ID] [TIMESTEPS] [--no-gui]
Default task: InvertedPendulum-v5 -- slide a cart to keep a pole upright (MuJoCo physics).
Training shows live in a MuJoCo viewer window (full speed, not real time) unless --no-gui.
Saves the trained model, a learning-curve chart, and before/after scores to runs/<ENV_ID>/.
"""
import sys
from pathlib import Path

import gymnasium as gym
import matplotlib
import mujoco.viewer

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

import so101  # noqa: F401  registers the SO-101 arm tasks (SO101Reach-v0)
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor, load_results

GUI = "--no-gui" not in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
ENV_ID = args[0] if args else "InvertedPendulum-v5"
TIMESTEPS = int(args[1]) if len(args) > 1 else 50_000
OUT = Path("runs") / ENV_ID
OUT.mkdir(parents=True, exist_ok=True)

# Environment = the simulated world. Monitor logs every episode's total reward to a CSV.
env = Monitor(gym.make(ENV_ID), str(OUT / "train"))

# Agent = PPO (Proximal Policy Optimization), with a small neural network as the policy.
model = PPO("MlpPolicy", env, seed=0, verbose=0)


def score():
    """Average episode reward, plus success rate (share of episodes ending in success) if the task reports one."""
    finals = []
    log_success = lambda local, _: local["done"] and finals.append(local["info"].get("is_success"))
    mean, _ = evaluate_policy(model, gym.make(ENV_ID), n_eval_episodes=20, callback=log_success)
    if None in finals:
        return f"{mean:.0f}"
    return f"{mean:.1f} (success {sum(finals) / len(finals):.0%})"


class SyncViewer(BaseCallback):
    """Push the training simulation's state to the viewer after every step."""

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer

    def _on_step(self):
        if self.viewer.is_running():
            self.viewer.sync()
        return True


before = score()
if GUI:
    sim = env.unwrapped  # the MuJoCo model/data being trained on
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        model.learn(total_timesteps=TIMESTEPS, callback=SyncViewer(viewer))
else:
    model.learn(total_timesteps=TIMESTEPS)
after = score()
model.save(OUT / "model")

# Learning curve: reward per episode as training goes on.
log = load_results(str(OUT))
steps = log["l"].cumsum()
plt.figure(figsize=(8, 4))
plt.plot(steps, log["r"], alpha=0.3, label="each episode")
plt.plot(steps, log["r"].rolling(20, min_periods=1).mean(), label="20-episode average")
plt.xlabel("simulation steps of experience")
plt.ylabel("episode reward (higher = better)")
plt.title(f"PPO learning {ENV_ID}")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "learning_curve.png", dpi=120)

summary = f"{ENV_ID}: avg reward before training {before} -> after {after} ({TIMESTEPS:,} steps)"
(OUT / "summary.txt").write_text(summary + "\n")
print(summary)
