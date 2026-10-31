"""SO-101 pick task with camera images, in LeRobot's format, for evaluating a fine-tuned pick policy.

Same scene, layouts and start pose as the sort task (so101/skill_env.py) and the pick demos
(vla/record_picks.py): the policy gets "pick up the green cube" (or red), front + wrist images and
joint readings, and sends 6 absolute joint targets in LeRobot SO-101 units, one per 30 ms tick.
Success = the named cube is lifted at least 3 cm and is at the gripper tip.

Layout seeds: pick demos use 100000+, sort evals 1000+; evaluate picks on 900000+ (held out).
layouts="train" replays the demo layouts instead (seed w -> 100000 + w, same cube), to separate fitting from generalising.
Diagnostics: distractor=False moves the other cube out of view; oracle=True appends the target cube's
start position (x, y in cm) to agent_pos, so a policy trained that way needs no vision to find it.
oracle_noise=CM adds one random offset per episode (normal, CM per axis) to that position, like a detector's
error: the hint is off by a fixed amount, so only the cameras can correct the last centimetre. true_cm keeps the truth.
"""
import gymnasium as gym
import mujoco
import numpy as np

from so101.pixel_env import IMAGE_SIZE
from so101.skill_env import CUBES, SO101Sort
from so101.skills import CONTROL_STEPS

LIFT = 0.03  # metres above the resting height
AWAY = [1.5, -1.5, 0.015]  # behind the front camera, out of every view
NEAR = 0.03  # cube centre to gripper tip, metres


class SO101PixelPick(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": round(1 / (0.005 * CONTROL_STEPS))}

    def __init__(self, max_steps=250, layouts="heldout", distractor=True, oracle=False, oracle_noise=0.0):
        self.layouts, self.distractor, self.oracle, self.oracle_noise = layouts, distractor, oracle, oracle_noise
        self.base = SO101Sort()
        self.model, self.data = self.base.model, self.base.data
        self.lo, self.hi = self.model.actuator_ctrlrange.T.copy()
        self.max_steps = max_steps
        self.renderer = mujoco.Renderer(self.model, IMAGE_SIZE, IMAGE_SIZE)
        self.task_description = ""

    def to_lerobot(self, q):  # radians -> LeRobot units (as SO101PixelReach)
        u = (q - self.lo) / (self.hi - self.lo)
        return np.concatenate([u[:5] * 200 - 100, u[5:] * 100]).astype(np.float32)

    def to_radians(self, a):
        u = np.concatenate([(a[:5] + 100) / 200, a[5:] / 100])
        return self.lo + np.clip(u, 0, 1) * (self.hi - self.lo)

    def _camera(self, name):
        self.renderer.update_scene(self.data, camera=name)
        return self.renderer.render().copy()

    def _obs(self):
        return {"pixels": {"front": self._camera("front"), "wrist": self._camera("wrist_cam")},
                "agent_pos": self.agent_pos()}

    def agent_pos(self):
        q = self.to_lerobot(self.data.qpos[:6].copy())
        return np.concatenate([q, self.target_cm]).astype(np.float32) if self.oracle else q

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.layouts == "train":  # as vla/record_picks.py batch 0: seeds 100000+w, cube per world from default_rng(100000)
            self.base.reset(seed=100_000 + seed)
            self.cube = CUBES[np.random.default_rng(100_000).integers(0, len(CUBES), 256)[seed]]
        else:
            self.base.reset(seed=None if seed is None else 900_000 + seed)
            self.cube = CUBES[self.np_random.integers(len(CUBES))]
        other = next(c for c in CUBES if c != self.cube)
        self.other_start = self.base.pos(other)[:2].copy()  # kept for diagnostics even when moved away
        if not self.distractor:
            adr = self.model.jnt_qposadr[self.model.body(other).jntadr[0]]
            self.data.qpos[adr:adr + 7] = [*AWAY, 1, 0, 0, 0]
            mujoco.mj_forward(self.model, self.data)
        self.true_cm = self.base.pos(self.cube)[:2] * 100
        self.target_cm = self.true_cm + self.np_random.normal(0, self.oracle_noise, 2) if self.oracle_noise else self.true_cm
        self.task_description = f"pick up the {self.cube.replace('_', ' ')}"
        self.z0, self.steps = self.base.pos(self.cube)[2], 0
        return self._obs(), {"task": self.task_description}

    def step(self, action):
        self.data.ctrl[:] = self.to_radians(np.asarray(action, np.float64))
        mujoco.mj_step(self.model, self.data, nstep=CONTROL_STEPS)
        self.steps += 1
        c = self.base.pos(self.cube)
        dist = float(np.linalg.norm(c - self.base.arm.tip_pos()))
        success = bool(c[2] - self.z0 > LIFT and dist < NEAR)
        return self._obs(), float(success), success, self.steps >= self.max_steps, {"distance": dist, "is_success": success}

    def render(self):
        return self._camera("front")

    def close(self):
        self.renderer.close()
