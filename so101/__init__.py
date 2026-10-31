"""SO-101 robot arm tasks. Importing this package registers them with Gymnasium."""
try:
    import gymnasium as gym
except ImportError:  # the vLLM container (plan_bench.py) has no Gymnasium; only the RL tasks need it
    gym = None
if gym is not None:
    gym.register("SO101Reach-v0", entry_point="so101.envs:SO101Reach", max_episode_steps=100)
