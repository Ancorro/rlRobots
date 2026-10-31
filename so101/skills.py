"""Scripted SO-101 skills: the teachers that SmolVLA skill policies will later imitate and replace.

A skill is a generator. Each `next()` gives the 6 servo targets (5 arm joints + gripper, radians)
for one control tick (CONTROL_STEPS physics steps = 30 ms, ~33 Hz). When the skill ends it returns
(ok, reason), so skills report failures honestly instead of failing silently (plan: docs/plan.md).
A learned skill only has to keep the same contract: yield targets, return (ok, reason).
"""
import mujoco
import numpy as np

from so101.arm import GRIP_CLOSED, GRIP_OPEN, ik_jaw_down, roll_for
from so101.envs import ARM_JOINTS

CONTROL_STEPS = 6  # physics steps (5 ms each) per control tick
SPEED = 0.15  # tip speed, metres per second
HOVER_Z = 0.08  # travel height for the tip, metres
GRIP_TICKS = 15  # ticks to wait for the jaw to open or close (~0.45 s)


class Arm:
    """The robot plus what the skills need to know about the scene (body ids by name).
    robot: an arm plug-in (so101/robot.py: SimArm or RealArm, behind a Guard); the skills then see only its joint
    readings, kept in `state` (a copy of the model posed from them), and step() drives it. robot=None: older
    tasks, which set data.ctrl and step the physics themselves; `state` is then the simulator itself.
    This class is the SO-101; so101/g1.py G1Arm overrides the robot-specific parts (the class attributes below, the
    IK in target/pose, turn_jaw, can_grasp and the jaw's direction and axis)."""
    tip_site, wrist_body, root = "gripperframe", "gripper", "base"  # root: the robot is this body and all under it
    n_arm = ARM_JOINTS  # read()/send() carry the arm joints, then the hand's
    grip_open, grip_closed = GRIP_OPEN, GRIP_CLOSED  # hand commands (SO-101: the gripper joint's target, radians)
    grip_joint, grip_joint_closed = "gripper", GRIP_CLOSED  # the joint whose stop short of closed tells a grasp
    hand = None  # grasp geometry for so101/grasp.py; None = its SO-101 constants
    ramp = 0  # ticks to move the hand command gradually (0 = at once)

    def __init__(self, model, data, scratch, robot=None):
        self.model, self.data, self.scratch = model, data, scratch
        self.robot = robot
        self.state = data if robot is None else mujoco.MjData(model)
        self.q = self.prev = None  # last two joint readings (robot only)
        self.assume_holding = False  # dry run: no objects in a rehearsal, so every grasp counts as made
        self.tip = model.site(self.tip_site).id
        self.wrist = model.body(self.wrist_body).id
        self.grip_adr = model.jnt_qposadr[model.joint(self.grip_joint).id]
        self.lo, self.hi = model.actuator_ctrlrange.T.copy()
        self.goal = None  # where the tip is being sent; None until the first move
        self.grip = self.grip_open
        self.open = self.grip_open  # how wide the jaw opens; less avoids hitting nearby things (tabletop)
        self.roll = None  # wrist_roll held for a turned grip; None = free (G1: the grip heading)
        self.blocked_margin = 0.15  # radians short of closed that counts as blocked (general task: 0.06, thin objects)
        if robot is not None:
            self.sync()

    def sync(self):
        """Read the joints and pose `state` from them (after every tick)."""
        self.prev, self.q = self.q, self.robot.read()
        self.state.qpos[:len(self.q)] = self.q
        mujoco.mj_kinematics(self.model, self.state)

    def step(self, q):
        """Send one tick's targets, wait the tick, read the joints."""
        self.robot.send(q)
        self.robot.tick()
        self.sync()

    def still(self, goal_q, tol=0.02, rest=0.002):
        """Arm joints at the target (within tol radians) or no longer moving (rest radians per tick)."""
        if self.prev is None:
            return False
        a = slice(0, self.n_arm)
        return bool(np.max(np.abs(self.q[a] - goal_q[a])) < tol or np.max(np.abs(self.q[a] - self.prev[a])) < rest)

    def tip_pos(self):
        return self.state.site_xpos[self.tip].copy()

    def body_pos(self, name):
        return self.data.xpos[self.model.body(name).id].copy()

    def jaw_closed(self):
        return self.grip == self.grip_closed

    def holding(self, name):
        """Grasp = jaw told to close but blocked part-way, object at the tip. Real servo grippers
        detect a grasp the same way, from position error. name=None: the jaw position alone (no sim truth)."""
        if self.assume_holding:
            return self.jaw_closed()
        blocked = self.state.qpos[self.grip_adr] > self.grip_joint_closed + self.blocked_margin
        near = name is None or np.linalg.norm(self.body_pos(name) - self.tip_pos()) < 0.03
        return bool(self.jaw_closed() and blocked and near)

    def target(self, goal):
        q = ik_jaw_down(self.model, self.state, self.scratch, self.tip, self.wrist, goal, self.lo, self.hi, roll=self.roll)
        return np.append(q, self.grip)

    def with_grip(self, q):
        """Targets q with the hand part set from the present grip command."""
        return np.append(q[:self.n_arm], self.grip)

    def pose(self, goal):
        """Pose `scratch` at the IK solution for `goal` (with the held roll)."""
        q = ik_jaw_down(self.model, self.state, self.scratch, self.tip, self.wrist, goal, self.lo, self.hi, roll=self.roll)
        self.scratch.qpos[:len(q)] = q
        mujoco.mj_kinematics(self.model, self.scratch)

    def jaw_direction(self):
        """Unit xy vector from the fixed jaw to the moving jaw, for the pose in scratch."""
        m, s = self.model, self.scratch
        f = np.mean([s.geom_xpos[m.geom(f"fixed_jaw_sph_tip{i}").id] for i in (1, 2, 3)], 0)
        mv = np.mean([s.geom_xpos[m.geom(f"moving_jaw_sph_tip{i}").id] for i in (1, 2, 3)], 0)
        n = (mv - f)[:2]
        return n / np.linalg.norm(n)

    def jaw_axis(self, goal):
        """Unit vector along the jaw (wrist to tip), for the IK pose at `goal`. Not straight down at far reach."""
        self.pose(goal)
        a = self.scratch.site_xpos[self.tip] - self.scratch.xpos[self.wrist]
        return a / np.linalg.norm(a)

    def carry_turn(self, xy, heading):
        """How far a held object turns carrying it to `xy` (the release point; x and y used) from where it was picked
        (base heading `heading`): the held wrist_roll stays, so the jaw turns with the base."""
        return np.arctan2(xy[1], xy[0]) - heading

    def can_grasp(self, g):
        """Whether the arm can do grasp g (so101/grasp.py plan's check). The SO-101 takes the planner's best."""
        return True

    def turn_jaw(self, goal, yaws):
        """Hold wrist_roll so that, at `goal`, the jaws close along one of the headings `yaws` (radians;
        the one needing the least wrist turn)."""
        self.roll = None
        q = ik_jaw_down(self.model, self.state, self.scratch, self.tip, self.wrist, goal, self.lo, self.hi)
        rolls = [roll_for(self.model, self.scratch, q, y, self.lo, self.hi) for y in np.atleast_1d(yaws)]
        yaw = np.atleast_1d(yaws)[int(np.argmin(np.abs(rolls)))]
        for _ in range(2):  # the roll moves the tip a little; re-solve with it held
            self.roll = roll_for(self.model, self.scratch, q, yaw, self.lo, self.hi)
            q = ik_jaw_down(self.model, self.state, self.scratch, self.tip, self.wrist, goal, self.lo, self.hi, roll=self.roll)


def run(arm, skill, each=None):
    """Drive a skill on arm.robot to the end; returns its (ok, reason). each(): called after every tick (viewers)."""
    while True:
        try:
            q = next(skill)
        except StopIteration as stop:
            return stop.value
        arm.step(q)
        if each:
            each()


def settle(arm, q, least, most):
    """Hold target q for `least` ticks, then (robot only) until the arm is there or stops moving, up to `most`."""
    for i in range(most if arm.robot is not None else least):
        if i >= least and arm.still(q):
            return
        yield q


def move_to(arm, goal, speed=SPEED, smooth=False):
    """Straight line from the current goal to `goal` at `speed`, then let the servos settle.
    smooth: ease in and out (minimum-jerk timing, same peak speed), so a held object is not jerked loose."""
    start = arm.tip_pos() if arm.goal is None else arm.goal
    goal = np.asarray(goal, float)
    n = max(1, int(np.ceil(np.linalg.norm(goal - start) / (speed * CONTROL_STEPS * 0.005) * (1.875 if smooth else 1))))
    for i in range(1, n + 1):
        t = i / n
        arm.goal = start + (goal - start) * ((10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5) if smooth else t)
        yield arm.target(arm.goal)
    yield from settle(arm, arm.target(arm.goal), 5, 35)  # real servos lag: wait until the arm gets there


def set_grip(arm, value):
    if arm.ramp:  # a hand with several joints: move the command along its path, so the joints follow the path (G1)
        q, start = arm.target(arm.goal), arm.grip
        for i in range(1, arm.ramp + 1):
            arm.grip = start + (value - start) * i / arm.ramp
            q = arm.with_grip(q)
            yield q
        arm.grip = value  # exact, so jaw_closed() sees it (the stepped sum can end ~1e-17 off)
        yield from settle(arm, q, GRIP_TICKS - arm.ramp, GRIP_TICKS - arm.ramp)
        return
    arm.grip = value
    yield from settle(arm, arm.target(arm.goal), GRIP_TICKS, GRIP_TICKS)


def pick(arm, obj, at=None, grasp_z=None, yaw=None, hover=HOVER_Z):
    """Grasp `obj` (a body name) from above and lift it to travel height.
    Defaults read the object's true position (sort task). From perception: `at` = (x, y) to grasp at,
    `grasp_z` = tip height for the grasp, `yaw` = heading (radians) the jaws should close along (long objects)."""
    if arm.jaw_closed():
        return False, "already holding something; place it first"
    p = arm.body_pos(obj)
    x, y = p[:2] if at is None else at
    z = p[2] if grasp_z is None else grasp_z
    if yaw is not None:
        arm.turn_jaw(np.array([x, y, z]), yaw)
    yield from move_to(arm, [x, y, hover])
    if at is None:
        p = arm.body_pos(obj)  # re-read: it may have been nudged
        x, y = p[:2]
        z = p[2] if grasp_z is None else grasp_z
    yield from move_to(arm, [x, y, z + 0.04])
    yield from move_to(arm, [x, y, z], speed=SPEED / 4)  # slow last 4 cm: fast ones tip the cube
    yield from set_grip(arm, GRIP_CLOSED)
    if not arm.holding(obj):
        yield from set_grip(arm, arm.open)
        yield from move_to(arm, [*arm.goal[:2], hover])
        arm.roll = None
        return False, f"missed {obj}"
    yield from move_to(arm, [*arm.goal[:2], hover])
    if not arm.holding(obj):
        yield from set_grip(arm, arm.open)
        arm.roll = None
        return False, f"dropped {obj} while lifting"
    return True, f"holding {obj}"


def place(arm, obj, place_at, release_z=0.035, hover=HOVER_Z):
    """Carry the held `obj` over `place_at` (a body name), lower the tip to `release_z` and let go."""
    if not arm.holding(obj):
        return False, "not holding anything to place"
    p = arm.body_pos(place_at)
    yield from move_to(arm, [p[0], p[1], hover])
    if not arm.holding(obj):
        yield from set_grip(arm, arm.open)
        arm.roll = None
        return False, f"dropped {obj} while carrying"
    yield from move_to(arm, [p[0], p[1], release_z])
    yield from set_grip(arm, arm.open)
    yield from move_to(arm, [p[0], p[1], hover])
    arm.roll = None
    return True, f"released {obj} over {place_at}"
