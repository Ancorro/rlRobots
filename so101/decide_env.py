"""SO-101 reach task for "System One" decision models (Jev-style: state text + fixed options in,
probabilities out), following EmbodiedJev's step-wise XYZ design (github.com/FBddcz/embodied-jev).

Each step the decision model answers three questions -- move the gripper tip along X, Y, Z:
negative / hold / positive. The program, not the model, turns that into motion: the tip target
moves STEP_MM along each chosen axis and IK (inverse kinematics: joint angles that put the tip at
a point) drives the arm there. Success = tip within SUCCESS_MM of the red ball.

Gymnasium action: MultiDiscrete([3, 3, 3]), 0 = negative, 1 = hold, 2 = positive per axis.
"""
import gymnasium as gym
import mujoco
import numpy as np

from so101.envs import ARM_JOINTS, SO101Reach

STEP_MM = 15.0  # how far one "positive"/"negative" moves the tip target
SUCCESS_MM = 20.0
MAX_DECISIONS = 40
SETTLE_STEPS = 20  # 20 x 5 ms physics steps for the servos to follow each decision
AXES = "xyz"
OPTIONS = ("negative", "hold", "positive")

# The questions every decision model gets, in the TypeSafe Jev / Laya request schema.
QUESTIONS = {
    ax: {
        "type": "choice",
        "instructions": f"Which way should the gripper tip move along the {ax.upper()} axis to get closer to the target?",
        "criteria": {"negative": f"decrease {ax}", "hold": f"keep {ax} the same", "positive": f"increase {ax}"},
    }
    for ax in AXES
}


class SO101DecideReach(gym.Env):
    metadata = {"render_modes": []}
    questions = QUESTIONS

    def __init__(self):
        self.base = SO101Reach()
        self.model, self.data = self.base.model, self.base.data
        self.scratch = mujoco.MjData(self.model)  # IK works on a copy so it never disturbs the physics
        self.lo, self.hi = self.base.lo, self.base.hi
        self.action_space = gym.spaces.MultiDiscrete([3, 3, 3])
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (9,), np.float32)

    # --- what the decision model sees ---------------------------------------------------------
    def tip_pos(self):
        return self.base._tip()

    def tip_mm(self):
        return self.base._tip() * 1000

    def target_mm(self):
        return self.model.site_pos[self.base.target] * 1000

    def state_text(self):
        t, g = self.tip_mm(), self.target_mm()
        d = g - t
        return (f"Gripper tip at x={t[0]:.0f} y={t[1]:.0f} z={t[2]:.0f} mm. "
                f"Red target ball at x={g[0]:.0f} y={g[1]:.0f} z={g[2]:.0f} mm. "
                f"Offset, target minus tip: dx={d[0]:+.0f} dy={d[1]:+.0f} dz={d[2]:+.0f} mm. "
                f"Each move is {STEP_MM:.0f} mm.")

    def _obs(self):
        t, g = self.tip_mm(), self.target_mm()
        return np.concatenate([t, g, g - t]).astype(np.float32) / 100

    def distance_mm(self):
        return float(np.linalg.norm(self.target_mm() - self.tip_mm()))

    # --- execution ------------------------------------------------------------------------------
    def _ik(self, goal_m, iters=30, damping=0.02):
        """Damped least squares: nudge the 5 arm joints until the tip reaches goal_m."""
        s, site = self.scratch, self.base.tip
        s.qpos[:] = self.data.qpos
        jac = np.zeros((3, self.model.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(self.model, s)
            mujoco.mj_comPos(self.model, s)
            err = goal_m - s.site_xpos[site]
            if np.linalg.norm(err) < 1e-4:
                break
            mujoco.mj_jacSite(self.model, s, jac, None, site)
            J = jac[:, :ARM_JOINTS]
            dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(3), err)
            s.qpos[:ARM_JOINTS] = np.clip(s.qpos[:ARM_JOINTS] + dq, self.lo[:ARM_JOINTS], self.hi[:ARM_JOINTS])
        return s.qpos[:ARM_JOINTS].copy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.base.reset(seed=seed)
        self.goal = self.base._tip()  # where the tip is being sent, metres
        self.steps = 0
        return self._obs(), {"state_text": self.state_text()}

    def step(self, action):
        move = (np.asarray(action) - 1) * STEP_MM / 1000  # -1/0/+1 per axis, in metres
        self.goal = self.goal + move
        self.goal[2] = max(self.goal[2], 0.01)  # never aim below the floor
        self.data.ctrl[:ARM_JOINTS] = self._ik(self.goal)
        before = self.distance_mm()
        mujoco.mj_step(self.model, self.data, nstep=SETTLE_STEPS)
        self.steps += 1
        dist = self.distance_mm()
        success = dist < SUCCESS_MM
        reward = (before - dist) / STEP_MM + (1.0 if success else 0.0)  # progress, plus a bonus for arriving
        info = {"distance_mm": dist, "is_success": success, "state_text": self.state_text()}
        return self._obs(), reward, success, self.steps >= MAX_DECISIONS, info

    def oracle_action(self, deadband_mm=STEP_MM / 2):
        """Rule-based baseline: move toward the target on each axis unless already close on it."""
        d = self.target_mm() - self.tip_mm()
        return np.where(d > deadband_mm, 2, np.where(d < -deadband_mm, 0, 1))
