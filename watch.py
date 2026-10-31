"""Watch a trained agent in MuJoCo's own viewer window.  Usage: uv run python watch.py [ENV_ID]

Uses mujoco.viewer rather than Gymnasium's render_mode="human", whose viewer crashes on
camera moves with MuJoCo 3.14 (mjv_moveCamera argument mismatch).
"""
import sys
import time

import gymnasium as gym
import mujoco.viewer
from stable_baselines3 import PPO

import so101  # noqa: F401  registers the SO-101 arm tasks (SO101Reach-v0)

ENV_ID = sys.argv[1] if len(sys.argv) > 1 else "InvertedPendulum-v5"
model = PPO.load(f"runs/{ENV_ID}/model")
env = gym.make(ENV_ID)
sim = env.unwrapped  # underlying MuJoCo model/data
dt = sim.dt  # simulated seconds per environment step, for real-time playback

with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
    for episode in range(5):
        obs, _ = env.reset()
        total, done = 0.0, False
        while not done and viewer.is_running():
            start = time.perf_counter()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
            viewer.sync()
            time.sleep(max(0.0, dt - (time.perf_counter() - start)))
        print(f"episode {episode + 1}: reward {total:.0f}", flush=True)
        if not viewer.is_running():
            break
env.close()
