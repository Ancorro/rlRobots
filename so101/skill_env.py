"""SO-101 sorting task at the skill level: the decision model picks the next skill and its object.

Two cubes (green, red) and two trays (green, red); each cube goes in the tray of its colour.
Each step the decision model answers two questions, in the TypeSafe Jev / Laya schema:
which skill (pick / place / done) and which object (one option per scene item). The scripted
skills in so101/skills.py carry it out and report done or failed with a reason, which the next
state text shows. Invalid pairs (pick a tray, place with an empty gripper) count as failed skills.

Gymnasium action: MultiDiscrete([3, 4]) = (skill index, object index).
`drop_prob` > 0 makes the gripper slip during some carries, to test recovery from a drop.
Set `env.on_tick` to a function to watch skills run (decide.py syncs the viewer with it).
"""
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from so101.arm import GRIP_OPEN
from so101.skills import CONTROL_STEPS, HOVER_Z, Arm, pick, place

SKILLS = ("pick", "place", "done")
OBJECTS = ("green_cube", "red_cube", "green_tray", "red_tray")
CUBES, TRAYS = OBJECTS[:2], OBJECTS[2:]
MAX_SKILLS = 8  # the rule baseline needs 4 (+ 2 per failure)
TRAY_HALF = 0.04  # metres
SCENE = Path(__file__).parent / "assets" / "scene_sort.xml"
GOAL_TEXT = "Put each cube in the tray of the same colour."

QUESTIONS = {
    "skill": {
        "type": "choice",
        "instructions": "Which skill should the robot run next? Pick grasps a cube and lifts it; "
                        "place puts the held cube into a tray; done ends the task.",
        "criteria": {"pick": "grasp a cube and lift it", "place": "put the held cube into a tray",
                     "done": "the task is finished, stop"},
    },
    "object": {
        "type": "choice",
        "instructions": "Which object should the skill act on? Pick needs a cube; place needs a tray.",
        "criteria": {o: f"the {o.replace('_', ' ')}" for o in OBJECTS},
    },
}


class SO101Sort(gym.Env):
    metadata = {"render_modes": []}
    questions = QUESTIONS
    valid_targets = {"pick": CUBES, "place": TRAYS}  # what each skill accepts (deciders may restrict to it)

    def __init__(self, drop_prob=0.0):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data, self.scratch = mujoco.MjData(self.model), mujoco.MjData(self.model)
        self.drop_prob = drop_prob
        self.on_tick = None
        self.action_space = gym.spaces.MultiDiscrete([len(SKILLS), len(OBJECTS)])
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (3 * len(OBJECTS) + len(CUBES) + 1,), np.float32)

    # --- what the decision model sees (metres internally, millimetres in text) -------------------
    def pos(self, name):
        return self.data.xpos[self.model.body(name).id].copy()

    def held(self):
        return next((c for c in CUBES if self.arm.holding(c)), None)

    def in_tray(self, cube, tray):
        c, t = self.pos(cube), self.pos(tray)
        return bool(np.all(np.abs(c[:2] - t[:2]) < TRAY_HALF) and c[2] < 0.03 and self.held() != cube)

    def where(self, cube):
        if self.held() == cube:
            return "in the gripper"
        tray = next((t for t in TRAYS if self.in_tray(cube, t)), None)
        return f"in the {tray.replace('_', ' ')}" if tray else "on the table"

    def n_sorted(self):
        return sum(self.in_tray(c, c.replace("cube", "tray")) for c in CUBES)

    def state_text(self):
        def mm(p):
            return f"x={p[0] * 1000:.0f} y={p[1] * 1000:.0f} z={p[2] * 1000:.0f} mm"
        held = self.held()
        lines = [f"Task: {GOAL_TEXT}",
                 f"Gripper: {'holding the ' + held.replace('_', ' ') if held else 'empty'}."]
        lines += [f"{c.replace('_', ' ')}: {mm(self.pos(c))}, {self.where(c)}." for c in CUBES]
        lines += [f"{t.replace('_', ' ')}: {mm(self.pos(t))}." for t in TRAYS]
        lines.append(f"Last skill: {self.last}. Skills used: {self.steps} of {MAX_SKILLS}.")
        return " ".join(lines)

    def status(self):
        return f"sorted {self.n_sorted()} / {len(CUBES)}   last: {self.last}"

    def _obs(self):
        held = self.held()
        return np.concatenate([*(self.pos(o) for o in OBJECTS), [held == c for c in CUBES],
                               [self.steps / MAX_SKILLS]]).astype(np.float32)

    # --- execution --------------------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        rng = self.np_random
        while True:  # four items in front of the arm, apart from each other
            xy = np.column_stack([rng.uniform(0.12, 0.27, 4), rng.uniform(-0.16, 0.16, 4)])
            gaps = [np.linalg.norm(xy[i] - xy[j]) for i in range(4) for j in range(i + 1, 4)]
            if min(gaps) > 0.09:
                break
        for name, (x, y) in zip(OBJECTS, xy):
            body = self.model.body(name)
            if name in CUBES:
                adr = self.model.jnt_qposadr[body.jntadr[0]]
                self.data.qpos[adr:adr + 7] = [x, y, 0.015, 1, 0, 0, 0]
            else:
                self.data.mocap_pos[self.model.body_mocapid[body.id]] = [x, y, 0.002]
        self.arm = Arm(self.model, self.data, self.scratch)
        self.arm.goal = np.array([0.18, 0.0, HOVER_Z])  # start above the workspace, jaw down
        self.data.ctrl[:] = self.arm.target(self.arm.goal)
        mujoco.mj_step(self.model, self.data, nstep=200)
        self.steps, self.last = 0, "none yet"
        return self._obs(), {"state_text": self.state_text(), "status": self.status()}

    def _run(self, skill):
        """Drive the arm with a skill generator until it returns (ok, reason)."""
        slip_at = self.np_random.integers(10, 40) if self.drop_prob and self.np_random.random() < self.drop_prob else -1
        tick = 0
        while True:
            try:
                self.data.ctrl[:] = next(skill)
            except StopIteration as stop:
                return stop.value
            if tick == slip_at and self.held():  # simulated slip: the jaw lets go mid-carry
                self.arm.grip = GRIP_OPEN
            mujoco.mj_step(self.model, self.data, nstep=CONTROL_STEPS)
            tick += 1
            if self.on_tick:
                self.on_tick()

    def step(self, action):
        skill, obj = SKILLS[action[0]], OBJECTS[action[1]]
        before = self.n_sorted()
        if skill == "done":
            ok, why = self.n_sorted() == len(CUBES), "stopped"
        elif skill == "pick" and obj not in CUBES:
            ok, why = False, "pick needs a cube"
        elif skill == "pick":
            ok, why = self._run(pick(self.arm, obj))
        elif obj not in TRAYS:
            ok, why = False, "place needs a tray"
        elif self.held() is None:
            ok, why = False, "not holding anything to place"
        else:
            ok, why = self._run(place(self.arm, self.held(), obj))
        self.steps += 1
        self.last = f"{skill} {obj} -> {'done' if ok else 'failed'}: {why}"
        success = self.n_sorted() == len(CUBES)
        reward = self.n_sorted() - before - 0.05 + (1.0 if success else 0.0)
        if skill == "done" and not success:
            reward -= 1.0  # quitting early must cost more than any amount of trying
        info = {"is_success": success, "status": self.status(), "state_text": self.state_text(),
                "skill_ok": ok, "skill": skill}
        return self._obs(), reward, success or skill == "done", self.steps >= MAX_SKILLS, info

    def oracle_action(self):
        """Rule baseline: place what is held in its tray; else pick the first unsorted cube; else done."""
        held = self.held()
        if held:
            return np.array([SKILLS.index("place"), OBJECTS.index(held.replace("cube", "tray"))])
        for c in CUBES:
            if not self.in_tray(c, c.replace("cube", "tray")):
                return np.array([SKILLS.index("pick"), OBJECTS.index(c)])
        return np.array([SKILLS.index("done"), 0])
