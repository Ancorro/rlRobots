"""SO-101 tasks with camera images, in LeRobot's format, for VLA (vision-language-action) policies.

The "System 1" policy sees what the real arm's policy sees:
  obs["pixels"]["front"], obs["pixels"]["wrist"]: 256x256 RGB (red-green-blue) camera images
  obs["agent_pos"]: 6 joint readings in LeRobot SO-101 units
and returns 6 absolute joint targets in the same units.

LeRobot units (so101_follower default): arm joints scaled so their range maps to -100..100,
gripper 0..100. On the real arm that range comes from each robot's calibration, so sim and
real units match in shape, not exactly in angle -- a known sim-to-real gap to handle later.

Headless machines (cluster, WSL) need MUJOCO_GL=egl for rendering.
"""
import gymnasium as gym
import mujoco
import numpy as np

from so101.envs import FRAME_SKIP, SUCCESS_DIST, SO101Reach

IMAGE_SIZE = 256


class SO101PixelReach(gym.Env):
    """SO101Reach with cameras and absolute joint-target actions."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": round(1 / (0.005 * FRAME_SKIP))}
    task_description = "move the gripper to the red ball"

    def __init__(self, max_steps=100, render_mode="rgb_array"):
        self.base = SO101Reach()
        self.model, self.data = self.base.model, self.base.data
        self.lo, self.hi = self.base.lo, self.base.hi
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.renderer = mujoco.Renderer(self.model, IMAGE_SIZE, IMAGE_SIZE)
        img = gym.spaces.Box(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)
        self.observation_space = gym.spaces.Dict({
            "pixels": gym.spaces.Dict({"front": img, "wrist": img}),
            "agent_pos": gym.spaces.Box(-np.inf, np.inf, (6,), np.float32),
        })
        self.action_space = gym.spaces.Box(np.array([-100] * 5 + [0], np.float32), np.full(6, 100, np.float32))

    # LeRobot units <-> radians
    def to_lerobot(self, q):
        u = (q - self.lo) / (self.hi - self.lo)  # 0..1 across each joint's range
        return np.concatenate([u[:5] * 200 - 100, u[5:] * 100]).astype(np.float32)

    def to_radians(self, a):
        u = np.concatenate([(a[:5] + 100) / 200, a[5:] / 100])
        return self.lo + np.clip(u, 0, 1) * (self.hi - self.lo)

    def _camera(self, name):
        self.renderer.update_scene(self.data, camera=name)
        return self.renderer.render().copy()

    def _obs(self):
        return {
            "pixels": {"front": self._camera("front"), "wrist": self._camera("wrist_cam")},
            "agent_pos": self.to_lerobot(self.data.qpos[:6].copy()),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.base.reset(seed=seed)
        self.steps = 0
        return self._obs(), {"task": self.task_description}

    def step(self, action):
        self.data.ctrl[:] = self.to_radians(np.asarray(action, np.float64))
        mujoco.mj_step(self.model, self.data, nstep=FRAME_SKIP)
        self.steps += 1
        dist = float(np.linalg.norm(self.model.site_pos[self.base.target] - self.base._tip()))
        success = dist < SUCCESS_DIST
        return self._obs(), float(success), success, self.steps >= self.max_steps, {"distance": dist, "is_success": success}

    def render(self):
        return self._camera("front")

    def close(self):
        self.renderer.close()
