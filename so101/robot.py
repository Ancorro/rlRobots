"""The arm as a plug-in, like the cameras (so101/camera.py): the planning code reads joint angles and sends joint
targets, and does not know whether the simulator or a real SO-101 is on the other end.

Angles are radians in the simulator's convention (the MuJoCo model's joints; zero = the model's zero pose, arm
straight out forward), 6 numbers: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.

  SimArm(model, data)   the simulated arm, optionally with real-servo sloppiness for testing
  RealArm(port)         a real SO-101 through LeRobot (UNTESTED: no arm yet)
  Guard(arm, model)     safety layer in front of either one: joint limits, speed cap, jaws above the table,
                        reach limit, and stop (freeze in place)
  KinematicArm(q)       weightless stand-in that reaches every target at once (dry runs, so101/dryrun.py)

Each has read() -> 6 angles, send(q), tick() (wait one control period, 30 ms), stop() (hold still where it is),
relax() (motors off: the arm falls limp) and close().

Real arm, before first use (python -m so101.robot ... --port=COM5, the arm's USB serial port):
  1. LeRobot's own setup: motor ids (lerobot-setup-motors) and joint ranges (lerobot-calibrate), see the
     LeRobot SO-101 guide. Use the id "so101".
  2. zero    match LeRobot's angles to the simulator's: pose the arm like the picture in the viewer, then move
             each joint the way the viewer shows. Writes runs/calib/arm.json.
  3. check   motors off; move the arm by hand and the viewer copy must follow it.
  selftest   checks the zero maths against a pretend LeRobot arm with random offsets and directions.
"""
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
TICK = 0.03  # control period, seconds (~33 Hz)
CONTROL_STEPS = 6  # physics steps (5 ms) per tick
CALIB = Path("runs/calib/arm.json")
JAW_TIPS = [f"{j}_jaw_sph_tip{i}" for j in ("fixed", "moving") for i in (1, 2, 3)]


@dataclass(frozen=True)
class Robot:
    """What the plug-ins and the guard need to know about a robot: which joints read()/send() carry and how to check
    them. SO101 below; so101/g1.py G1 (a humanoid's waist and right arm, and its hand)."""
    name: str
    joints: tuple  # in read()/send() order: arm joints, then the hand's (actuators have the same names)
    arm: int  # how many of `joints` are arm joints (speed cap, calibration error); the rest are the hand
    root: str  # the robot is this body and everything under it (known pixels, the dry-run ghost)
    tip: str  # site at the grasp point
    floor_geoms: tuple  # the hand's lowest points (floor rule)
    reach_from: tuple = (0.0, 0.0)  # xy the reach rule measures from
    reach: float = 0.42
    ceiling: float = 0.40
    steps: int = CONTROL_STEPS  # physics steps per 30 ms tick


SO101 = Robot("so101", JOINTS, 5, "base", "gripperframe", tuple(JAW_TIPS))


class Stopped(Exception):
    """The guard froze the arm: a rule was broken or stop was pressed."""


class SimArm:
    """The simulated arm: send() sets the position servos' targets, tick() runs 30 ms of physics.

    Real-servo sloppiness, for testing (all off by default):
      joint_err  calibration error, degrees (standard deviation; one random offset per joint, drawn from `seed`).
                 The joint sits that far from where the software thinks, and its reading is off by the same
                 amount, so the software cannot tell; only the cameras could. Arm joints only: the gripper's
                 closed reading is measured on the real arm (close it on nothing).
      play       dead band, degrees: a servo stops correcting once within `play` of its target (gear play and the
                 servo's dead zone). The reading shows where it really stopped.
      quantise   readings in whole encoder steps, 4096 per turn, as the real servos report them.
    The servo lag itself is already in the physics (the model's gains are derived from LeRobot's servo settings)."""

    def __init__(self, model, data, joint_err=0.0, play=0.0, quantise=False, seed=0, steps=None, robot=SO101):
        self.model, self.data, self.steps = model, data, steps or robot.steps
        self.qadr = [model.jnt_qposadr[model.joint(j).id] for j in robot.joints]
        self.act = [model.actuator(j).id for j in robot.joints]
        self.err = np.zeros(len(robot.joints))
        self.err[:robot.arm] = np.radians(np.random.default_rng(seed).normal(0.0, joint_err, robot.arm))  # true - believed
        self.play, self.quantise = np.radians(play), quantise
        self.servo = None  # servo targets after the dead band (true angles)

    def read(self):
        q = self.data.qpos[self.qadr] - self.err
        return np.round(q / (2 * np.pi / 4096)) * (2 * np.pi / 4096) if self.quantise else q.copy()

    def send(self, q):
        target = np.asarray(q, float) + self.err
        if self.servo is None or not self.play:
            self.servo = target
        else:  # moves only when the target leaves the dead band, then trails it by `play`
            self.servo = np.clip(self.servo, target - self.play, target + self.play)
        self.data.ctrl[self.act] = self.servo

    def tick(self):
        mujoco.mj_step(self.model, self.data, nstep=self.steps)

    def stop(self):
        self.send(self.read())

    def relax(self):
        pass  # the simulator has no limp mode; stop() holds it

    def close(self):
        pass


class KinematicArm:
    """Weightless stand-in: every target is reached at once, nothing is touched. For dry runs."""

    def __init__(self, q):
        self.q = np.array(q, float)

    def read(self):
        return self.q.copy()

    def send(self, q):
        self.q = np.array(q, float)

    def tick(self):
        pass

    def stop(self):
        pass

    def relax(self):
        pass

    def close(self):
        pass


class Guard:
    """Safety layer between the planning code and an arm. Every target is checked before it is sent:
      limits  joint angles inside their ranges, `margin` radians in from the ends (clipped)
      speed   no arm joint moves more than `max_speed` rad/s from the last target (clipped: the arm just goes
              slower). 4 rad/s is about an STS3215 servo's top speed, so in the simulator this also stops the
              arm from moving faster than a real one could. The planner's targets do jump: e.g. turning the
              wrist for a grasp re-poses the whole arm in one tick (measured: up to 0.7 rad, wrist 4.8 rad).
      floor   the jaw tips stay above `floor` metres (the table is at 0)
      reach   the gripper tip stays within `reach` metres of the base, below `ceiling`
    Clipping is always on. A floor or reach breach: mode "stop" freezes the arm where it is and raises Stopped;
    mode "warn" sends it anyway and counts it (simulator benchmarks: how often a rule would fire).
    stop() freezes the arm (it holds its position) and every later send raises Stopped until resume().
    relax() turns the motors off: the arm drops onto the table and lets go of what it holds. Freeze first."""

    def __init__(self, arm, model, mode="stop", max_speed=4.0, floor=-0.003, reach=None, ceiling=None, margin=0.0,
                 robot=SO101):
        self.arm, self.model, self.mode = arm, model, mode
        hand = len(robot.joints) - robot.arm
        self.max_step = np.r_[np.full(robot.arm, max_speed * TICK), np.full(hand, np.inf)]  # the hand moves at its own top speed
        self.floor, self.origin = floor, np.array(robot.reach_from)
        self.reach = robot.reach if reach is None else reach
        self.ceiling = robot.ceiling if ceiling is None else ceiling
        lo, hi = model.actuator_ctrlrange[[model.actuator(j).id for j in robot.joints]].T
        self.lo, self.hi = lo + margin, hi - margin
        self.qadr = [model.jnt_qposadr[model.joint(j).id] for j in robot.joints]
        self.fk = mujoco.MjData(model)
        self.tip = model.site(robot.tip).id
        self.jaw = [model.geom(n).id for n in robot.floor_geoms]
        self.last = None
        self.frozen = False
        self.pressed = False  # the stop key (set from a viewer's key handler; acted on at the next send)
        self.counts = {"limits": 0, "speed": 0, "floor": 0, "reach": 0}
        self.lowest = np.inf  # lowest jaw tip of any target sent, metres

    def read(self):
        return self.arm.read()

    def check(self, q):
        """Rules a target breaks (floor, reach), without sending it."""
        self.fk.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.fk)
        low = float(min(self.fk.geom_xpos[g][2] for g in self.jaw))
        tip = self.fk.site_xpos[self.tip]
        broken = []
        if low < self.floor:
            broken.append(f"floor: jaw tip at {low * 1000:.0f} mm")
        if np.linalg.norm(tip[:2] - self.origin) > self.reach or tip[2] > self.ceiling:
            broken.append(f"reach: tip at {np.round(tip * 100, 1)} cm")
        return broken, low

    def send(self, q):
        if self.pressed:
            self.pressed = False
            self.stop()
            raise Stopped("stop pressed")
        if self.frozen:
            raise Stopped("arm is stopped")
        q = np.asarray(q, float)
        c = np.clip(q, self.lo, self.hi)
        self.counts["limits"] += bool(np.any(np.abs(c - q) > 1e-4))  # readings at an end stop overshoot ~3e-6
        if self.last is not None:
            step = np.clip(c - self.last, -self.max_step, self.max_step)
            self.counts["speed"] += bool(np.any(np.abs(c - self.last) > self.max_step + 1e-9))
            c = self.last + step
        broken, low = self.check(c)
        self.lowest = min(self.lowest, low)
        for b in broken:
            self.counts[b.split(":")[0]] += 1
        if broken and self.mode == "stop":
            self.stop()
            raise Stopped("; ".join(broken))
        self.last = c
        self.arm.send(c)

    def tick(self):
        self.arm.tick()

    def stop(self):
        """Freeze: hold the present position. Safe: the arm keeps holding whatever is in the jaw."""
        self.frozen = True
        self.arm.stop()
        self.last = self.arm.read()

    def resume(self):
        self.frozen = False
        self.last = self.arm.read()

    def relax(self):
        self.frozen = True
        self.arm.relax()

    def close(self):
        self.arm.close()


class RealArm:
    """A real SO-101 follower arm through LeRobot. UNTESTED: written before the arm arrived.

    LeRobot reports degrees from the middle of each joint's calibrated range, and the gripper as 0-100 % of its
    range. runs/calib/arm.json (written by `python -m so101.robot zero`) converts: radians = sign * radians(deg) +
    offset for the arm joints; the gripper maps its closed and open readings (percent) to the simulator's limits.
    tick() keeps the 30 ms rhythm (waits out the rest of the period) and reads the joints once per tick.
    max_step_deg: LeRobot's own cap on how far a target may be from where the joint is (a second safety layer)."""

    def __init__(self, port, calib=CALIB, robot_id="so101", max_step_deg=10.0):
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        self.cal = json.loads(Path(calib).read_text()) if calib else None
        self.bot = SO101Follower(SO101FollowerConfig(port=port, id=robot_id, use_degrees=True,
                                                     max_relative_target=max_step_deg))
        self.bot.connect()
        self.next = time.perf_counter()
        self.late = 0  # ticks that overran the period
        self.q = None

    def raw(self):
        obs = self.bot.get_observation()
        return np.array([obs[f"{j}.pos"] for j in JOINTS], float)

    def read(self):
        if self.q is None:
            self.q = to_sim(self.raw(), self.cal)
        return self.q.copy()

    def send(self, q):
        a = to_lerobot(np.asarray(q, float), self.cal)
        self.bot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, a)})

    def tick(self):
        self.next += TICK
        wait = self.next - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        else:
            self.late += 1
            self.next = time.perf_counter()
        self.q = to_sim(self.raw(), self.cal)

    def stop(self):
        self.q = to_sim(self.raw(), self.cal)
        self.send(self.q)

    def relax(self):
        self.bot.bus.disable_torque()

    def close(self):
        self.bot.disconnect()


def to_sim(raw, cal):
    """LeRobot readings (degrees x5, gripper percent) -> simulator radians."""
    sign, off = np.array(cal["sign"]), np.array(cal["offset"])
    q = np.empty(6)
    q[:5] = sign * np.radians(raw[:5]) + off
    (pc, po), (qc, qo) = cal["gripper_pct"], cal["gripper_rad"]
    q[5] = qc + (raw[5] - pc) / (po - pc) * (qo - qc)
    return q


def to_lerobot(q, cal):
    sign, off = np.array(cal["sign"]), np.array(cal["offset"])
    raw = np.empty(6)
    raw[:5] = np.degrees((q[:5] - off) * sign)
    (pc, po), (qc, qo) = cal["gripper_pct"], cal["gripper_rad"]
    raw[5] = pc + (q[5] - qc) / (qo - qc) * (po - pc)
    return raw


# --- commands (real arm) ------------------------------------------------------------------------------
def zero(raw_read, show, ask, model, window=2.0):
    """Fill the angle conversion. raw_read() -> LeRobot readings; show(q) puts the viewer's arm in pose q;
    ask(text) waits for the person. Readings are averaged over `window` seconds after Enter, while the person keeps
    holding the arm still (with the motors off, the elbow and wrist sag if let go). Returns the calibration dict.
    Offsets: the arm posed like the simulator's zero pose. Directions: each joint moved the way the viewer shows.
    Gripper: readings fully closed and fully open, mapped to the simulator's joint limits."""
    lo, hi = model.jnt_range[:6].T
    q0 = np.zeros(6)
    q0[5] = lo[5]
    def held():
        out = []
        for _ in range(10):
            out.append(raw_read())
            time.sleep(window / 10)
        return np.mean(out, 0)

    show(q0)
    ask("Motors are off. Pose the arm like the viewer: arm straight out forward, level, jaw closed. Keep one hand "
        "holding it there and press Enter with the other; hold still 2 more seconds.")
    r0 = held()
    sign = np.ones(5)
    for j in range(5):
        q = q0.copy()
        q[j] = 0.5 if hi[j] > 0.5 else hi[j]
        show(q)
        ask(f"Move only the {JOINTS[j].replace('_', ' ')} about 30 degrees, the way the viewer shows. Enter.")
        d = held()[j] - r0[j]
        if abs(d) < 5:
            raise RuntimeError(f"{JOINTS[j]} barely moved ({d:.1f} deg): check the joint and try again")
        sign[j] = np.sign(d)
        show(q0)
        ask(f"Put the {JOINTS[j].replace('_', ' ')} back. Enter.")
    closed = r0[5]
    show(np.r_[q0[:5], hi[5]])
    ask("Open the jaw all the way and hold it. Enter.")
    opened = held()[5]
    return {"sign": sign.tolist(), "offset": (-sign * np.radians(r0[:5])).tolist(),
            "gripper_pct": [float(closed), float(opened)], "gripper_rad": [float(lo[5]), float(hi[5])]}


def _arg(name, default):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


def _viewer(model):
    import mujoco.viewer
    data = mujoco.MjData(model)
    viewer = mujoco.viewer.launch_passive(model, data)

    def show(q):
        data.qpos[:6] = q
        mujoco.mj_forward(model, data)
        viewer.sync()
    return viewer, show


def cmd_zero():
    from so101.objects import SCENE
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    arm = RealArm(_arg("port", "COM5"), calib=None)
    arm.relax()
    viewer, show = _viewer(model)
    cal = zero(arm.raw, show, input, model)
    CALIB.parent.mkdir(parents=True, exist_ok=True)
    CALIB.write_text(json.dumps(cal, indent=1))
    print(f"saved {CALIB}; now run: python -m so101.robot check --port=...")
    viewer.close()
    arm.close()


def cmd_check():
    """Motors off; the viewer copy follows the real arm. Readings printed in simulator degrees."""
    from so101.objects import SCENE
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    arm = RealArm(_arg("port", "COM5"))
    arm.relax()
    viewer, show = _viewer(model)
    print("Move the arm by hand; the viewer must copy it. Close the viewer to quit.")
    while viewer.is_running():
        arm.tick()
        q = arm.read()
        show(q)
        print("  ".join(f"{j[:8]} {np.degrees(v):6.1f}" for j, v in zip(JOINTS, q)), end="\r")
    arm.close()


def cmd_selftest():
    """A pretend LeRobot arm with random offsets and directions; a pretend person poses it within 2 degrees."""
    from so101.objects import SCENE
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    rng = np.random.default_rng(1)
    true_sign, true_off = rng.choice([-1, 1], 5), rng.uniform(-90, 90, 5)  # LeRobot deg = sign * sim deg + off
    pct = (rng.uniform(0, 10), rng.uniform(85, 100))  # gripper % at the sim's closed and open limits
    lo, hi = model.jnt_range[:6].T
    pose = {"q": np.zeros(6)}

    def raw_read():
        return raw_read_at(pose["q"] + np.radians(rng.normal(0, 0.1, 6)), true_sign, true_off, pct, lo, hi)  # reading noise

    def show(q):
        pose["q"] = np.array(q) + np.r_[np.radians(rng.normal(0, 2, 5)), 0]  # the person copies it, 2 deg off

    cal = zero(raw_read, show, lambda text: None, model, window=0.0)
    test = np.r_[rng.uniform(-1, 1, 5), 0.6]
    raw = raw_read_at(test, true_sign, true_off, pct, lo, hi)
    print("directions found:", cal["sign"] == true_sign.tolist())
    print("angle error after zero, degrees:", np.round(np.degrees(to_sim(raw, cal) - test), 2),
          "(the person's 2 deg posing error shows here)")
    print("round trip error:", float(np.abs(to_lerobot(to_sim(raw, cal), cal) - raw).max()))


def raw_read_at(q, sign, off, pct, lo, hi):
    raw = np.empty(6)
    raw[:5] = sign * np.degrees(q[:5]) + off
    raw[5] = pct[0] + (q[5] - lo[5]) / (hi[5] - lo[5]) * (pct[1] - pct[0])
    return raw


if __name__ == "__main__":
    {"zero": cmd_zero, "check": cmd_check, "selftest": cmd_selftest}[sys.argv[1]]()
