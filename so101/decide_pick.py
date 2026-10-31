"""SO-101 pick-and-place for decision models: put the green cube in the blue tray.

Four questions per step, in the Jev / Laya schema: move X, Y, Z (negative / hold / positive) and
gripper (open / hold / close). The program executes: the tip target moves STEP_MM per axis and IK
(inverse kinematics) drives the arm there while keeping the jaw pointing down.
Gymnasium action: MultiDiscrete([3, 3, 3, 3]); for the gripper 0 = open, 1 = hold, 2 = close.
Success = cube resting in the tray and released.
"""
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from so101.arm import GRIP_CLOSED, GRIP_OPEN, ik_jaw_down
from so101.decide_env import QUESTIONS, SETTLE_STEPS, STEP_MM
from so101.envs import ARM_JOINTS

PICK_QUESTIONS = {
    **QUESTIONS,
    "gripper": {
        "type": "choice",
        "instructions": "What should the gripper do? Close it on the cube to grab it; open it over the tray to release.",
        "criteria": {"open": "open the jaw", "hold": "keep the jaw as it is", "close": "close the jaw"},
    },
}
MAX_DECISIONS = 80
HOVER = 0.05  # metres: approach and carry height. 0.08 was out of the jaw-down workspace within ~15 cm of the base.
TRAY_HALF = 0.04  # metres
SCENE = Path(__file__).parent / "assets" / "scene_pick.xml"


class SO101DecidePick(gym.Env):
    metadata = {"render_modes": []}
    questions = PICK_QUESTIONS

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data, self.scratch = mujoco.MjData(self.model), mujoco.MjData(self.model)
        self.tip = self.model.site("gripperframe").id
        self.wrist = self.model.body("gripper").id
        self.cube = self.model.body("cube").id
        self.cube_qpos = self.model.jnt_qposadr[self.model.body_jntadr[self.cube]]
        self.tray = self.model.body("tray").id
        self.lo, self.hi = self.model.actuator_ctrlrange.T.copy()
        self.action_space = gym.spaces.MultiDiscrete([3, 3, 3, 3])
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (10,), np.float32)

    # --- what the decision model sees (metres internally, millimetres in text) -------------------
    def tip_pos(self):
        return self.data.site_xpos[self.tip].copy()

    def cube_pos(self):
        return self.data.xpos[self.cube].copy()

    def tray_pos(self):
        return self.model.body_pos[self.tray].copy()

    def jaw_closed(self):
        return self.data.ctrl[5] < 0.5

    def holding(self):
        """Grasp = jaw told to close but stopped early (blocked by the cube), cube at the tip.
        Real servo grippers detect a grasp the same way, from position error."""
        blocked = self.data.qpos[5] > GRIP_CLOSED + 0.15
        near = np.linalg.norm(self.cube_pos() - self.tip_pos()) < 0.03
        return bool(self.jaw_closed() and blocked and near)

    def in_tray(self):
        c, t = self.cube_pos(), self.tray_pos()
        return bool(np.all(np.abs(c[:2] - t[:2]) < TRAY_HALF) and c[2] < 0.03 and not self.jaw_closed())

    def state_text(self):
        def mm(p):
            return f"x={p[0] * 1000:.0f} y={p[1] * 1000:.0f} z={p[2] * 1000:.0f} mm"
        tip, cube, tray = self.tip_pos(), self.cube_pos(), self.tray_pos()
        to_cube, to_tray = (cube - tip) * 1000, (tray - cube) * 1000
        return (f"Gripper tip at {mm(tip)}, jaw {'closed' if self.jaw_closed() else 'open'}, "
                f"{'holding the cube' if self.holding() else 'not holding the cube'}. "
                f"Green cube at {mm(cube)}. Blue tray at {mm(tray)}. "
                f"Cube minus tip: dx={to_cube[0]:+.0f} dy={to_cube[1]:+.0f} dz={to_cube[2]:+.0f} mm. "
                f"Tray minus cube: dx={to_tray[0]:+.0f} dy={to_tray[1]:+.0f} mm. Each move is {STEP_MM:.0f} mm.")

    def _obs(self):
        return np.concatenate([self.tip_pos(), self.cube_pos(), self.tray_pos()[:2],
                               [float(self.jaw_closed()), float(self.holding())]]).astype(np.float32)

    def distance_mm(self):
        return self._progress() * 1000

    # --- execution --------------------------------------------------------------------------------
    def _ik(self, goal, w=None):
        return ik_jaw_down(self.model, self.data, self.scratch, self.tip, self.wrist, goal, self.lo, self.hi, w)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        rng = self.np_random
        while True:  # cube and tray in front of the arm, apart from each other
            cube = np.array([rng.uniform(0.13, 0.26), rng.uniform(-0.12, 0.12)])
            tray = np.array([rng.uniform(0.13, 0.26), rng.uniform(-0.15, 0.15)])
            if np.linalg.norm(cube - tray) > 0.09:
                break
        self.data.qpos[self.cube_qpos:self.cube_qpos + 3] = [*cube, 0.015]
        self.model.body_pos[self.tray] = [*tray, 0.002]
        self.data.ctrl[5] = GRIP_OPEN
        mujoco.mj_forward(self.model, self.data)
        self.goal = np.array([0.18, 0.0, HOVER])  # start above the workspace, jaw down
        self.data.ctrl[:ARM_JOINTS] = self._ik(self.goal)
        mujoco.mj_step(self.model, self.data, nstep=200)
        self.steps = 0
        return self._obs(), {"state_text": self.state_text()}

    def _progress(self):
        """Distance still to cover, metres: reach the cube, then carry it to the tray."""
        if self.in_tray():
            return 0.0
        if self.holding():
            return float(np.linalg.norm(self.tray_pos()[:2] - self.cube_pos()[:2]))
        return 0.3 + float(np.linalg.norm(self.cube_pos() - self.tip_pos()))

    def step(self, action):
        a = np.asarray(action)
        self.goal = np.clip(self.goal + (a[:3] - 1) * STEP_MM / 1000, [0.05, -0.3, 0.01], [0.35, 0.3, 0.25])
        jaw_cmd = None if a[3] == 1 else (GRIP_OPEN if a[3] == 0 else GRIP_CLOSED)
        jaw_changed = jaw_cmd is not None and jaw_cmd != self.data.ctrl[5]
        if jaw_cmd is not None:
            self.data.ctrl[5] = jaw_cmd
        before = self._progress()
        self.data.ctrl[:ARM_JOINTS] = self._ik(self.goal)
        mujoco.mj_step(self.model, self.data, nstep=SETTLE_STEPS * 2)
        if jaw_changed:  # the jaw takes longer than one decision to close: let it stop (up to 0.5 s more) before the
            for _ in range(10):  # state is read, else a half-closed jaw reads as a grasp and the arm lifts too early
                if abs(self.data.qvel[5]) < 0.05:
                    break
                mujoco.mj_step(self.model, self.data, nstep=10)
        self.steps += 1
        success = self.in_tray()
        reward = (before - self._progress()) * 10 + (5.0 if success else 0.0)
        info = {"distance_mm": self.distance_mm(), "is_success": success, "holding": self.holding(),
                "state_text": self.state_text()}
        return self._obs(), reward, success, self.steps >= MAX_DECISIONS, info

    def oracle_action(self):
        """Rule-based baseline. Uses only what the state text states, so a decision model could too."""
        tip, cube, tray = self.tip_pos(), self.cube_pos(), self.tray_pos()
        db = STEP_MM / 2 / 1000

        def toward(delta):
            return np.where(delta > db, 2, np.where(delta < -db, 0, 1))

        low = tip[2] < HOVER - 0.005
        tol = db + 0.003 if low else db  # once descending, accept a little more offset: with exactly half a step the
        # target flipped in and out of line as the arm came down and the tip dithered sideways for ever
        if self.holding():
            if np.any(np.abs(tray[:2] - cube[:2]) > tol):  # carry: rise first, then across
                return np.array([*toward(tray[:2] - cube[:2]), 2 if tip[2] < HOVER - 0.01 else 1, 1])
            if cube[2] > 0.03:
                return np.array([1, 1, 0, 1])  # lower into the tray
            return np.array([1, 1, 1, 0])  # release
        if self.jaw_closed():
            return np.array([1, 1, 2, 0])  # empty jaw: open and back off upward
        d = cube - tip
        if np.any(np.abs(d[:2]) > tol):  # line up above the cube
            return np.array([*toward(d[:2]), 2 if tip[2] < HOVER - 0.01 else 1, 1])
        if d[2] < -0.004:
            return np.array([1, 1, 0, 1])  # descend onto it
        return np.array([1, 1, 1, 2])  # close on the cube
