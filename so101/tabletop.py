"""SO-101 tabletop task at the skill level: move objects of different shapes to named places.

Objects: red cube, blue ball, yellow cylinder (standing), green bar (lying, any heading); cube and bar get
a random heading. Places (fixed, like a calibrated real workspace): red tray, blue tray, shelf (6 cm up).
Each episode sets a goal of two "object -> place" moves, e.g. "Put the blue ball on the shelf and the
green bar in the red tray."

Layers (docs/plan.md): perception says where each object is (`perception="exact"` reads the simulator;
"detector" uses so101/perception.py on the front camera) -> the decision model picks skill + object ->
the scripted skills (so101/skills.py) get the perceived position, grasp height and grip heading.
Same interface as so101/skill_env.py (questions, state_text, oracle_action, on_tick), so decide.py runs it.

Gymnasium action: MultiDiscrete([3, 7]) = (skill, object or place).
"""
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from so101.skills import CONTROL_STEPS, Arm, pick, place

SKILLS = ("pick", "place", "done")
OBJECTS = ("red_cube", "blue_ball", "yellow_cylinder", "green_bar")
PLACES = ("red_tray", "blue_tray", "shelf")
TARGETS = OBJECTS + PLACES
HALF_HEIGHT = {"red_cube": 0.012, "blue_ball": 0.015, "yellow_cylinder": 0.02, "green_bar": 0.01}  # metres
# grip heading relative to the object's heading: across the bar; cube faces every 90 degrees; any for round ones
GRIP = {"red_cube": [0, np.pi / 2], "green_bar": [np.pi / 2], "blue_ball": None, "yellow_cylinder": None}
TRAY_HALF, SHELF_HALF, SHELF_TOP = 0.04, (0.035, 0.05), 0.06
GRIP_OPEN_TABLETOP = 0.6  # radians, jaw gap ~6 cm: wide open (10 cm) hit the shelf and the arm's own shoulder
HOVER = 0.11  # travel height for the tip: a carried object clears the shelf and the standing cylinder
MAX_SKILLS = 8  # the rule needs 4 (+ 2 per failure)
SCENE = Path(__file__).parent / "assets" / "scene_tabletop.xml"


def name(o):
    return o.replace("_", " ")


QUESTIONS = {
    "skill": {
        "type": "choice",
        "instructions": "Which skill should the robot run next? Pick grasps an object and lifts it; "
                        "place puts the held object on a tray or the shelf; done ends the task.",
        "criteria": {"pick": "grasp an object and lift it", "place": "put the held object on a place",
                     "done": "the task is finished, stop"},
    },
    "object": {
        "type": "choice",
        "instructions": "Which object or place should the skill act on? Pick needs an object; place needs a tray or the shelf.",
        "criteria": {o: f"the {name(o)}" for o in TARGETS},
    },
}


class SO101Tabletop(gym.Env):
    metadata = {"render_modes": []}
    questions = QUESTIONS
    valid_targets = {"pick": OBJECTS, "place": PLACES}  # what each skill accepts (deciders may restrict to it)

    def __init__(self, perception="exact"):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data, self.scratch = mujoco.MjData(self.model), mujoco.MjData(self.model)
        self.on_tick = None
        if perception == "exact":
            self.perceive = self.exact
        else:
            from so101.perception import Detector
            self.detector = Detector(self.model, self.data)
            self.perceive = lambda: self.detector.locate(OBJECTS)
        self.action_space = gym.spaces.MultiDiscrete([len(SKILLS), len(TARGETS)])
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (3 * len(OBJECTS) + len(OBJECTS) + 1,), np.float32)

    # --- the scene ------------------------------------------------------------------------------
    def pos(self, o):
        return self.data.xpos[self.model.body(o).id].copy()

    def heading(self, o):
        R = self.data.xmat[self.model.body(o).id].reshape(3, 3)
        return float(np.arctan2(R[1, 0], R[0, 0]))  # the body's x axis (the bar's long axis)

    def exact(self):
        """Perception from the simulator: {object: (x, y, grasp height, heading)}."""
        return {o: (*self.pos(o)[:2], self.pos(o)[2], self.heading(o)) for o in OBJECTS}

    def held(self):
        return next((o for o in OBJECTS if self.arm.holding(o)), None)

    def at(self, o, p):
        c, t = self.pos(o), self.pos(p)
        if self.held() == o:
            return False
        if p == "shelf":
            return bool(np.all(np.abs(c[:2] - t[:2]) < SHELF_HALF) and c[2] > SHELF_TOP)
        return bool(np.all(np.abs(c[:2] - t[:2]) < TRAY_HALF) and c[2] < HALF_HEIGHT[o] + 0.02)

    def where(self, o):
        if self.held() == o:
            return "in the gripper"
        p = next((p for p in PLACES if self.at(o, p)), None)
        return ("on the shelf" if p == "shelf" else f"in the {name(p)}") if p else "on the table"

    def n_done(self):
        return sum(self.at(o, p) for o, p in self.goal)

    def goal_text(self):
        parts = [f"the {name(o)} {'on the shelf' if p == 'shelf' else 'in the ' + name(p)}" for o, p in self.goal]
        return "Put " + " and ".join(parts) + "."

    def state_text(self):
        def mm(p):
            return f"x={p[0] * 1000:.0f} y={p[1] * 1000:.0f} z={p[2] * 1000:.0f} mm"
        held = self.held()
        seen = self.seen
        lines = [f"Task: {self.goal_text()}", f"Gripper: {'holding the ' + name(held) if held else 'empty'}."]
        lines += [f"{name(o)}: {mm(seen[o][:3])}, {self.where(o)}." for o in OBJECTS]
        lines += [f"{name(p)}: {mm(self.pos(p))}." for p in PLACES]
        lines.append(f"Last skill: {self.last}. Skills used: {self.steps} of {MAX_SKILLS}.")
        return " ".join(lines)

    def status(self):
        return f"done {self.n_done()} / {len(self.goal)}   last: {self.last}"

    def _obs(self):
        held = self.held()
        return np.concatenate([*(self.pos(o) for o in OBJECTS), [held == o for o in OBJECTS],
                               [self.steps / MAX_SKILLS]]).astype(np.float32)

    # --- execution ------------------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        rng = self.np_random
        while True:  # objects in front of the arm, clear of each other, the trays and the shelf
            xy = np.column_stack([rng.uniform(0.14, 0.21, 4), rng.uniform(-0.11, 0.11, 4)])
            xy[3] = [rng.uniform(0.16, 0.19), rng.uniform(-0.08, 0.08)]  # the bar is long: keep it further in
            if min(np.linalg.norm(xy, axis=1)) > 0.15 and min(np.linalg.norm(xy[i] - xy[j]) for i in range(4) for j in range(i + 1, 4)) > 0.065:
                break
        for o, (x, y) in zip(OBJECTS, xy):
            a = rng.uniform(-np.pi, np.pi) if GRIP[o] else 0.0
            adr = self.model.jnt_qposadr[self.model.body(o).jntadr[0]]
            self.data.qpos[adr:adr + 7] = [x, y, HALF_HEIGHT[o], np.cos(a / 2), 0, 0, np.sin(a / 2)]
        objs = rng.choice(len(OBJECTS), 2, replace=False)
        places = rng.choice(len(PLACES), 2, replace=False)
        self.goal = [(OBJECTS[o], PLACES[p]) for o, p in zip(objs, places)]
        self.arm = Arm(self.model, self.data, self.scratch)
        self.arm.open = self.arm.grip = GRIP_OPEN_TABLETOP
        self.arm.goal = np.array([0.18, 0.0, HOVER])
        self.data.ctrl[:] = self.arm.target(self.arm.goal)
        mujoco.mj_step(self.model, self.data, nstep=200)
        self.steps, self.last = 0, "none yet"
        self.seen = {o: (np.nan,) * 4 for o in OBJECTS}
        self.look()
        return self._obs(), {"state_text": self.state_text(), "status": self.status()}

    def _run(self, skill):
        while True:
            try:
                self.data.ctrl[:] = next(skill)
            except StopIteration as stop:
                return stop.value
            mujoco.mj_step(self.model, self.data, nstep=CONTROL_STEPS)
            if self.on_tick:
                self.on_tick()

    def look(self):
        """Update where objects are; one the camera misses (e.g. in the gripper) keeps its last position."""
        self.seen.update(self.perceive())

    def pick_skill(self, o):
        x, y, z, a = self.seen[o]
        if np.isnan(x):
            return iter(())  # never seen: the skill fails at once
        yaw = None if GRIP[o] is None else [a + g for g in GRIP[o]]
        return pick(self.arm, o, at=(x, y), grasp_z=z, yaw=yaw, hover=HOVER)

    def place_skill(self, o, p):
        top = SHELF_TOP if p == "shelf" else 0.004
        return place(self.arm, o, p, release_z=top + HALF_HEIGHT[o] + 0.01, hover=HOVER)

    def step(self, action):
        skill, target = SKILLS[action[0]], TARGETS[action[1]]
        before = self.n_done()
        if skill == "done":
            ok, why = self.n_done() == len(self.goal), "stopped"
        elif skill == "pick" and target not in OBJECTS:
            ok, why = False, "pick needs an object"
        elif skill == "pick":
            ok, why = self._run(self.pick_skill(target)) or (False, f"cannot see the {name(target)}")
        elif target not in PLACES:
            ok, why = False, "place needs a tray or the shelf"
        elif self.held() is None:
            ok, why = False, "not holding anything to place"
        else:
            ok, why = self._run(self.place_skill(self.held(), target))
        self.steps += 1
        self.last = f"{skill} {target} -> {'done' if ok else 'failed'}: {why}"
        self.look()  # perception re-reads the scene after each skill
        success = self.n_done() == len(self.goal)
        reward = self.n_done() - before - 0.05 + (1.0 if success else 0.0)
        if skill == "done" and not success:
            reward -= 1.0
        info = {"is_success": success, "status": self.status(), "state_text": self.state_text(),
                "skill_ok": ok, "skill": skill}
        return self._obs(), reward, success or skill == "done", self.steps >= MAX_SKILLS, info

    def oracle_action(self):
        """Rule baseline: place what is held at its goal place; else pick the first goal object not there; else done."""
        held = self.held()
        goal = dict(self.goal)
        if held:
            return np.array([SKILLS.index("place"), TARGETS.index(goal.get(held, "red_tray"))])
        for o, p in self.goal:
            if not self.at(o, p):
                return np.array([SKILLS.index("pick"), TARGETS.index(o)])
        return np.array([SKILLS.index("done"), 0])
