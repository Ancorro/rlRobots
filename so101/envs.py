"""SO-101 arm tasks as Gymnasium environments (MuJoCo physics).

SO101Reach-v0: move the gripper tip to a random red target. Smallest useful arm task;
it checks the robot model, action scaling, and reward before anything bigger.

Action: 5 numbers in [-1, 1], each nudging one arm joint's position target by up to
MAX_DELTA radians per step -- the same "go to this joint angle" interface the real
SO-101 servos use. The gripper is held open.
Observation: joint angles and speeds, gripper-tip position, target position, and the
gap between them (21 numbers).
"""
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

SCENE = Path(__file__).parent / "assets" / "scene_reach.xml"
FRAME_SKIP = 7  # 7 x 5 ms physics steps = 35 ms per action, about 29 Hz (real arm runs ~30 Hz)
MAX_DELTA = 0.05  # radians a joint target may move per action
EPISODE_STEPS = 100  # 3.5 seconds of simulated time
SUCCESS_DIST = 0.02  # metres: tip within 2 cm of the target counts as reached
ARM_JOINTS = 5  # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll (gripper excluded)


class SO101Reach(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.tip = self.model.site("gripperframe").id
        self.target = self.model.site("target").id
        self.lo, self.hi = self.model.actuator_ctrlrange.T.copy()
        self.dt = self.model.opt.timestep * FRAME_SKIP
        self.action_space = gym.spaces.Box(-1.0, 1.0, (ARM_JOINTS,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (21,), np.float32)

    def _tip(self):
        return self.data.site_xpos[self.tip].copy()

    def _sample_target(self):
        # Pose the arm at random joint angles and use where the tip lands: every target is reachable.
        for _ in range(100):
            q = self.np_random.uniform(0.6 * self.lo[:ARM_JOINTS], 0.6 * self.hi[:ARM_JOINTS])
            self.data.qpos[:ARM_JOINTS] = q
            mujoco.mj_kinematics(self.model, self.data)
            p = self._tip()
            if p[2] > 0.03:  # above the floor
                return p
        return np.array([0.25, 0.0, 0.1])

    def _obs(self):
        tip, goal = self._tip(), self.model.site_pos[self.target]
        d = self.data
        return np.concatenate([d.qpos[:6], d.qvel[:6], tip, goal, goal - tip]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.model.site_pos[self.target] = self._sample_target()
        mujoco.mj_resetData(self.model, self.data)  # arm back to the zero pose
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        a = np.clip(action, -1.0, 1.0)
        ctrl = self.data.ctrl
        ctrl[:ARM_JOINTS] = np.clip(ctrl[:ARM_JOINTS] + MAX_DELTA * a, self.lo[:ARM_JOINTS], self.hi[:ARM_JOINTS])
        mujoco.mj_step(self.model, self.data, nstep=FRAME_SKIP)
        self.steps += 1
        dist = float(np.linalg.norm(self.model.site_pos[self.target] - self._tip()))
        reward = -dist - 0.01 * float(np.square(a).sum())  # get close, without jerky moves
        truncated = self.steps >= EPISODE_STEPS
        return self._obs(), reward, False, truncated, {"distance": dist, "is_success": dist < SUCCESS_DIST}
