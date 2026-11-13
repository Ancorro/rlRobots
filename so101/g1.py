"""Unitree G1 humanoid with Dex3 hands at the benchmark table, as a second robot for the same pick-and-place stack
(perception, grasp planning from points, skills, safety guard). Humanoid step 1: the pelvis is fixed in place (no
walking or balance yet); the right arm and hand do the work and the waist turns and leans to help it reach.

Model: MuJoCo Menagerie unitree_g1/g1_with_hands.xml (BSD-3), fetched by so101/assets/fetch_g1.py into a gitignored
folder; scene_g1.xml puts it at the same table as the SO-101 (table top z = 0, same trays, shelf and cameras).
prepare() welds the pelvis to the world at PELVIS, sets the table edge and the ground from it, and adds the pinch site.

The Dex3 hand (thumb 3 joints, index and middle 2 each) as a two-jaw gripper, palm down, picking from above:
  - the index finger stays curled at INDEX, pointing straight down from the palm: the fixed jaw. The middle finger
    is tucked up out of the way: index and middle are 5.7 cm apart, and small things (a 2-4 cm ball or cylinder)
    were pushed through the gap between them (2026-10-05 scripted test).
  - the thumb is the moving jaw, turned (thumb_0) to face the index finger. One opening value (0 = closed, 1 = open)
    sets thumb_1 and thumb_2 by interpolating THUMB's keyframes, chosen so the thumb tip stays level with the index
    tip as it opens (a straight line between open and closed swings it ~2.5 cm lower half-way).
  - the jaws close along the hand's length (the thumb sits behind the index), so the wrist is ~12 cm back from the
    grasp along the closing line; the arm takes a grasp heading from either side, whichever it reaches.
A grasp is detected as on the SO-101: told to close, the thumb stops part-way (position error).

Commands (cluster; python -m so101.g1 ...):
  hand    measure the opening (gap between the thumb and the index pad vs the opening value): the table in Hand
  reach   check the right arm reaches the workspace, both trays and the shelf with the hand pointing down
  claw    scripted picks of small boxes, balls and cylinders without cameras (points from the true object)
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

from so101.robot import Robot
from so101.skills import Arm

SCENE = Path(__file__).parent / "assets" / "scene_g1.xml"
PELVIS = np.array([-0.08, 0.0, 0.02])  # pelvis position: hips just behind the table edge, table top at hip height
TABLE_EDGE = 0.11  # the table's near edge, in front of the pelvis (the hips reach 10 cm forward)
FEET = 0.793  # pelvis height above the feet, legs straight (the model's standing height)

ARM = ("waist_yaw_joint", "waist_pitch_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
       "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
       "right_wrist_yaw_joint")
HAND = tuple(f"right_hand_{n}_joint" for n in ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"))
HAND_BODY = "right_wrist_yaw_link"  # the hand's frame (palm, finger bases)
LEFT_ARM = {"left_shoulder_pitch_joint": 0.2, "left_shoulder_roll_joint": 0.2, "left_elbow_joint": 1.28}  # at its side
# (the model's standing pose; at zero the forearm points forward over the table)
INDEX, MIDDLE = (0.9, 0.6), (1.57, 1.74)  # base and tip joints: index straight down from the palm, middle tucked
THUMB_0 = -0.33  # turns the thumb to face the index pad (its face stays within ~2 mm of the line through the pad)
# opening value -> thumb_1, thumb_2 (radians). Solved 2026-10-05 for the thumb tip level with the index tip
# (within ~1 mm) at each thumb_1.
THUMB = np.array([[0.00, -0.96, 0.00], [0.12, -0.84, 0.00], [0.25, -0.40, -0.72], [0.51, 0.00, -1.17],
                  [0.69, 0.24, -1.37], [1.00, 0.60, -1.52]])
# in the hand frame: the axis that points down in a grasp (the curled index's direction), the closing direction (from
# the index pad toward the thumb, 21 degrees across the hand) and the pinch site: at the contact level 8 mm up from the
# index tip, FIXED_OFFSET (2 cm) from the index pad's centre toward the thumb.
DOWN = np.array([0.0707, 0.9975, 0.0])
CLOSE = np.array([-0.9297, 0.0659, -0.3636])
PINCH = np.array([0.1269, 0.0753, 0.0213])
AXES = np.column_stack([DOWN, CLOSE, np.cross(DOWN, CLOSE)])  # the hand frame's grasp axes (ik's `frame` targets)
TIPS = {"right_hand_thumb_2_link": (0.0, 0.052, 0.0), "right_hand_index_1_link": (0.052, 0.0, 0.0)}  # tip ends (link
# frames): the floor rule's points
READY = np.array([0.0, 0.0, -0.3, -0.2, 0.0, 1.0, 0.0, 0.0, 0.0])  # arm start pose and IK posture (ARM order)
# IK postures tried, in order, for a grip that keeps the arm off the torso and the table (G1Arm.choose): READY; waist
# turned left 0.4 with the upper arm turned out 0.6; the wrist rolled 1.0; waist turned left 0.4 with the elbow out.
# The best 3 of 14 tried (2026-10-06, runs/scratch/posture_scan.py): clean reach of (point, heading) pairs, READY alone
# -> these 3: objects 109 -> 150/180, red tray 17 -> 39/54, blue tray 45 -> 47/54, shelf 27 -> 48/54.
POSTURES = (READY, READY + [0.4, 0, 0, 0, 0.6, 0, 0, 0, 0], READY + [0, 0, 0, 0, 0, 0, 1.0, 0, 0],
            READY + [0.4, 0, 0, -0.4, 0, 0, 0, 0, 0])
THUMB_FORCE = 1.4  # thumb_1 and thumb_2 torque limit, N m (the model's; the index finger that holds against it has 1.4)


class Hand:
    """Grasp geometry for so101/grasp.py (same meaning as its SO-101 constants), measured with `python -m so101.g1 hand`.
    The thumb cannot reach closer than ~0.8 cm to the index pad at the contact level: thinner things are not held."""
    GAP_Q = np.linspace(0, 1, 11)  # opening values
    GAP_M = np.array([0.0080, 0.0137, 0.0199, 0.0271, 0.0336, 0.0400, 0.0466, 0.0530, 0.0599, 0.0665, 0.0729])  # gap
    FIXED_OFFSET = 0.02  # index pad to the pinch site
    FIXED_CLEAR, MOVING_CLEAR = 0.006, 0.012  # pad clearance from the object before closing
    MAX_GAP = 0.07
    FINGER_HALF = 0.012  # half a pad's width (thumb and index are both ~2.6 cm wide)
    MIN_Z = 0.012  # lowest pinch height above the support: the index tip is 8 mm below the site


G1 = Robot("g1", ARM + HAND, len(ARM), "pelvis", "pinch", tuple(f"tip_{b}" for b in TIPS),
           reach_from=(PELVIS[0], PELVIS[1]), reach=0.75, ceiling=0.45, steps=15)


def prepare(spec):
    """Fix the G1 in place behind the table and add what the planner needs (called on the scene spec before compiling)."""
    pelvis = spec.body("pelvis")
    spec.delete(pelvis.first_joint())  # the free joint: no walking yet, the pelvis is welded to the world
    pelvis.pos = PELVIS
    for key in list(spec.keys):  # its keyframe has the free joint's 7 numbers
        spec.delete(key)
    spec.geom("floor").pos = [PELVIS[0] + TABLE_EDGE + 0.4, 0, -0.5]  # table (0.8 m deep): near edge at TABLE_EDGE
    spec.geom("ground").pos = [0, 0, PELVIS[2] - FEET]
    for b in spec.bodies:
        if b.name and b.name != "world":
            b.gravcomp = 1  # stands in for the gravity feed-forward a humanoid's joint controller adds (no sag)
    for j in ("right_hand_thumb_1_joint", "right_hand_thumb_2_joint"):
        spec.joint(j).actfrcrange = [-THUMB_FORCE, THUMB_FORCE]
    hand = spec.body(HAND_BODY)
    hand.add_site(name="pinch", pos=PINCH, size=[0.004, 0, 0], group=5)
    for b, p in TIPS.items():  # small markers, no collisions, not drawn
        spec.body(b).add_geom(name=f"tip_{b}", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.002, 0, 0], pos=p,
                              contype=0, conaffinity=0, group=5, rgba=[0, 0, 0, 0], mass=0)


SYNERGY = json.loads((Path(__file__).parent / "assets" / "dex3_synergy.json").read_text())  # the real hand's closing
SQUEEZE = 2.0  # the synergy is commanded this far past the recorded closed pose, so the fingers press once they touch


def hand_q(opening, synergy=None):
    """The 7 hand joint targets for an opening value (0 closed, 1 open): the thumb-index pinch (THUMB, INDEX), or with
    `synergy` the real hand's habit, fingers wrapping toward a nearly still thumb (open -> SQUEEZE x the recorded
    closing move; so101/assets/dex3_synergy.json, from Unitree's G1 + Dex3 recordings)."""
    if synergy is not None:
        o, c = np.array(synergy["open"]), np.array(synergy["closed"])
        return o + (1 - opening) * SQUEEZE * (c - o)
    t1 = np.interp(opening, THUMB[:, 0], THUMB[:, 1])
    t2 = np.interp(opening, THUMB[:, 0], THUMB[:, 2])
    return np.array([THUMB_0, t1, t2, INDEX[0], INDEX[1], MIDDLE[0], MIDDLE[1]])


class G1Arm(Arm):
    """skills.Arm for the G1: the IK, the hand's opening and the grip heading. Joints carried by read()/send() are
    G1.joints: ARM (waist yaw and pitch, right arm) then HAND. `roll` holds the grip heading here (radians, world,
    the direction from the fingers to the thumb), not a joint angle; `posture` the IK posture chosen with it."""
    tip_site, wrist_body, root = "pinch", HAND_BODY, "pelvis"
    n_arm = len(ARM)
    grip_open, grip_closed = 1.0, 0.0
    grip_joint, grip_joint_closed = "right_hand_thumb_1_joint", THUMB[0, 1]
    hand = Hand
    ramp = 12  # ticks to move the opening command along the thumb's path (the joints then follow the path)

    def __init__(self, model, data, scratch, robot=None):
        names = G1.joints
        self.qadr = np.array([model.jnt_qposadr[model.joint(j).id] for j in names])
        self.dofadr = np.array([model.jnt_dofadr[model.joint(j).id] for j in ARM])
        super().__init__(model, data, scratch, robot)
        if robot is not None:  # joints the plug-in does not carry stay where start() put them (the contact check sees them)
            for j, v in LEFT_ARM.items():
                self.state.qpos[model.jnt_qposadr[model.joint(j).id]] = v
            mujoco.mj_kinematics(model, self.state)
        self.lo, self.hi = model.actuator_ctrlrange[[model.actuator(j).id for j in names]].T.copy()
        self.handb = model.body(HAND_BODY).id
        pelvis, shoulder = model.body("pelvis").id, model.body("right_shoulder_pitch_link").id

        def under(b):
            while b and b != shoulder:
                b = model.body_parentid[b]
            return b == shoulder
        # contacts that count (clashes): the right arm or hand with the rest of the robot, and the arm or palm with what
        # is fixed to the world (table, shelf). The finger tips may brush the table at the lowest grasps (MIN_Z): the
        # guard's floor rule watches them.
        own = np.array([model.body_rootid[b] == pelvis for b in range(model.nbody)])
        fingers = np.array([model.body(b).name.startswith(("right_hand_thumb", "right_hand_index", "right_hand_middle"))
                            for b in range(model.nbody)])
        self.moving = np.array([under(b) for b in range(model.nbody)])  # the right arm and hand
        self.rest, self.fixed = own & ~self.moving, ~own & (model.body_weldid == 0)
        self.arm_only = self.moving & ~fingers
        self.posture = READY
        self.frame = None  # a full hand orientation to hold instead of the heading (so101/g1_grasps.py)
        self.synergy = None  # SYNERGY: close the hand the way the real one does instead of the pinch (g1_grasps)

    def sync(self):
        self.prev, self.q = self.q, self.robot.read()
        self.state.qpos[self.qadr] = self.q
        mujoco.mj_kinematics(self.model, self.state)

    def solve(self, goal):
        """IK for `goal` with the held hand orientation (`frame`, g1_grasps) or heading, and its posture (READY when
        neither is held)."""
        if self.frame is not None:
            return ik(self, goal, frame=self.frame, posture=self.posture)
        return ik(self, goal, self.roll, posture=READY if self.roll is None else self.posture)

    def target(self, goal):
        return np.concatenate([self.solve(goal), hand_q(self.grip, self.synergy)])

    def with_grip(self, q):
        return np.concatenate([q[:self.n_arm], hand_q(self.grip, self.synergy)])

    def pose(self, goal):
        """Pose arm.scratch at the IK solution for `goal` (the held heading)."""
        q = self.solve(goal)
        self.scratch.qpos[:] = self.state.qpos
        self.scratch.qpos[self.qadr[:self.n_arm]] = q
        mujoco.mj_kinematics(self.model, self.scratch)

    def jaw_direction(self):
        n = (self.scratch.xmat[self.handb].reshape(3, 3) @ CLOSE)[:2]
        return n / np.linalg.norm(n)

    def jaw_axis(self, goal):
        self.pose(goal)
        return self.scratch.xmat[self.handb].reshape(3, 3) @ DOWN

    def carry_turn(self, goal, heading):
        """Keep the grip heading while carrying to `goal` (the release point) if the arm reaches it there cleanly (within
        3 mm, no contact) in some posture; else turn the held object in 30-degree steps, least first, until it does.
        Returns the turn (radians); with none clean, the held heading in its best posture."""
        if self.roll is None:
            return 0.0
        held = self.roll
        for t in sorted(np.radians(np.arange(-150, 181, 30)), key=abs):
            score, h, posture = self.choose(goal, [held + t])
            if not score[0] and score[1] == 0:
                self.roll, self.posture = h, posture
                return float(t)
        _, self.roll, self.posture = self.choose(goal, [held])
        return 0.0

    def clashes(self, q):
        """Contacts (body name pairs) of the right arm or hand with the rest of the robot, the table or the shelf with
        the arm at q (ARM order), posed on scratch. The IK ignores collisions, and a pose with the elbow against the
        torso left the hand 3-5 cm short of a grasp (2026-10-06). Objects do not count: the planner knows them only from
        the cameras."""
        m, s = self.model, self.scratch
        s.qpos[:] = self.state.qpos
        s.qpos[self.qadr[:self.n_arm]] = q
        mujoco.mj_kinematics(m, s)
        mujoco.mj_collision(m, s)
        b1, b2 = m.geom_bodyid[s.contact.geom1], m.geom_bodyid[s.contact.geom2]
        mv, rest, arm, fixed = self.moving, self.rest, self.arm_only, self.fixed
        hit = (mv[b1] & rest[b2]) | (mv[b2] & rest[b1]) | (arm[b1] & fixed[b2]) | (arm[b2] & fixed[b1])
        return [(m.body(i).name, m.body(j).name) for i, j in zip(b1[hit], b2[hit])]

    def can_grasp(self, g):
        """The arm reaches grasp g (thumb on either side, any of POSTURES) within 3 mm without touching itself, the
        table or the shelf."""
        goal = np.array([*g.xy, g.z])
        for h in (g.yaw, g.yaw + np.pi):
            for posture in POSTURES:
                q, err, _ = ik(self, goal, h, report=True, posture=posture)
                if err < 0.003 and not self.clashes(q):
                    return True
        return False

    def choose(self, goal, headings):
        """(score, heading, posture): of `headings` (radians) x POSTURES, what the arm reaches best at `goal`: no contact
        with itself or the table first, then least position error, least hand tilt, earliest posture, least motion."""
        best = None
        for h in headings:
            h = (h + np.pi) % (2 * np.pi) - np.pi
            for i, posture in enumerate(POSTURES):
                q, err, tilt = ik(self, goal, h, report=True, posture=posture)
                move = float(np.sum((q - self.state.qpos[self.qadr[:self.n_arm]]) ** 2))
                score = (bool(self.clashes(q)), round(err / 0.003), round(tilt / 0.1), i, move)
                if best is None or score < best[0]:
                    best = (score, h, posture)
        return best

    def turn_jaw(self, goal, yaws):
        """Hold the grip heading closest to one of `yaws` (either way along each: the thumb can be on either side), and
        the posture, that the arm reaches best at `goal` (choose)."""
        _, self.roll, self.posture = self.choose(goal, [h for y in np.atleast_1d(yaws) for h in (y, y + np.pi)])


def ik(arm, goal, yaw=None, w=None, iters=100, report=False, seed=None, posture=None, frame=None):
    """Arm joint angles (ARM order) putting the pinch site at `goal` with the hand pointing down and, unless yaw is
    None, closing along heading `yaw`. Damped least squares on MuJoCo's Jacobians, from the present pose (arm.state;
    from READY too if that ends more than 3 mm off: warm starts sometimes slide into a bent pose that cannot get there),
    solved on arm.scratch; READY's posture is pulled toward only in what the task leaves free (the arm has 9 joints for
    6 numbers). w trades the hand's tilt against position as on the SO-101 (arm.ik_jaw_down): default the strictest
    weight that still reaches within 3 mm (else the closest), so the hand tilts only at far reach.
    report=True: also return the position error (m) and the hand's tilt from straight down (radians).
    posture: the pose pulled toward instead of READY (and solved from, as READY is).
    frame: a full hand orientation instead (world 3x3, columns: where DOWN, CLOSE and their cross product should point;
    so101/g1_grasps.py tilted and side grasps), held strictly (w = 1); report then gives the orientation error angle."""
    rest = READY if posture is None else np.asarray(posture, float)
    if w is None:
        best = None
        for start in (seed, rest) if seed is None else (seed,):
            for w in (1.0,) if frame is not None else (1.0, 0.3, 0.1, 0.03):
                out = ik(arm, goal, yaw, w, iters, report=True, seed=start, posture=rest, frame=frame)
                if best is None or out[1] < best[1] - 1e-4:
                    best = out
                if out[1] < 0.003:
                    break
            if best[1] < 0.003:
                break
        return best if report else best[0]
    m, s = arm.model, arm.scratch
    qa, cols = arm.qadr[:arm.n_arm], arm.dofadr
    lo, hi = arm.lo[:arm.n_arm], arm.hi[:arm.n_arm]
    s.qpos[:] = arm.state.qpos
    q = s.qpos[qa].copy() if seed is None else np.array(seed, float)
    down = np.array([0.0, 0.0, -1.0])
    close = None if yaw is None else np.array([np.cos(yaw), np.sin(yaw), 0.0])
    wy = w if frame is not None else 0.0 if yaw is None else max(w, 0.3)  # the heading: kept stricter than the tilt
    cost = np.diag([10.0, 10.0] + [1.0] * (arm.n_arm - 2))  # the waist moves only where the arm alone falls short
    eye = np.eye(len(q))
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))

    def resid():
        s.qpos[qa] = q
        mujoco.mj_kinematics(m, s)
        R = s.xmat[arm.handb].reshape(3, 3)
        a = R @ DOWN
        e = np.cross(a, down)  # rotation that tilts the claw to straight down
        if frame is not None:  # rotation that brings all three hand axes onto the target frame
            e = 0.5 * np.cross((R @ AXES).T, frame.T).sum(0)
        elif close is not None:  # plus the turn about the vertical that brings the closing line to the heading
            c = R @ CLOSE
            e = 0.5 * (e + np.cross(c, close) + np.cross(np.cross(a, c), np.cross(down, close)))
        return np.concatenate([goal - s.site_xpos[arm.tip], [w * e[0], w * e[1], wy * e[2]]]), a

    for _ in range(iters):
        r, a = resid()
        if np.linalg.norm(r) < 1e-4:
            break
        mujoco.mj_comPos(m, s)
        mujoco.mj_jacSite(m, s, jp, jr, arm.tip)
        J = np.vstack([jp[:, cols], w * jr[:2, cols], wy * jr[2:, cols]])
        A = J.T @ J + 1e-4 * cost
        dq = np.linalg.solve(A, J.T @ r)
        dq += (eye - np.linalg.solve(A, J.T @ J)) @ (0.1 * (rest - q))  # posture, in the task's null space
        q = np.clip(q + np.clip(dq, -0.3, 0.3), lo, hi)
    if not report:
        return q
    r, a = resid()
    if frame is not None:
        Rc = s.xmat[arm.handb].reshape(3, 3) @ AXES
        return q, float(np.linalg.norm(r[:3])), float(np.arccos(np.clip((np.trace(Rc.T @ frame) - 1) / 2, -1, 1)))
    return q, float(np.linalg.norm(r[:3])), float(np.arccos(np.clip(-a[2], -1, 1)))


# --- commands ---------------------------------------------------------------------------------------------------------
def build(scene=SCENE):
    spec = mujoco.MjSpec.from_file(str(scene))
    prepare(spec)
    return spec.compile()


def start(model, data):
    """Arm at READY, hand open, left arm at its side, everything else (legs, waist roll) at zero; targets the same."""
    for j, v in [*zip(G1.joints, np.r_[READY, hand_q(1.0)]), *LEFT_ARM.items()]:
        data.qpos[model.jnt_qposadr[model.joint(j).id]] = v
        data.ctrl[model.actuator(j).id] = v
    mujoco.mj_forward(model, data)


def cmd_hand():
    """Gap between the thumb's face and the index pad along CLOSE at the contact level vs the opening value, the hand
    posed in free space, on the collision meshes; how far the thumb's face is off the closing line through the pad's
    centre, how far its tip is above or below the index tip, how far each face leans from upright over the contact level
    (+ = toward the other jaw lower down: the jaws then close in a V and squeeze round things upward, 2026-10-06); the
    pinch site's distance to the pad (FIXED_OFFSET)."""
    model = build()
    data = mujoco.MjData(model)
    start(model, data)
    hb = model.body(HAND_BODY).id
    geoms = lambda names: [g for g in range(model.ngeom) if model.geom_bodyid[g] in [model.body(n).id for n in names]
                           and model.geom_contype[g]]
    thumb = geoms(["right_hand_thumb_1_link", "right_hand_thumb_2_link"])
    index = geoms(["right_hand_index_0_link", "right_hand_index_1_link"])

    def points(gs):
        R, out = data.xmat[hb].reshape(3, 3), []
        for g in gs:
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                i = model.geom_dataid[g]
                v = model.mesh_vert[model.mesh_vertadr[i]:model.mesh_vertadr[i] + model.mesh_vertnum[i]]
            else:
                v = np.array(np.meshgrid(*[(-x, x) for x in model.geom_size[g]])).reshape(3, -1).T
            out.append((v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g] - data.xpos[hb]) @ R)
        return np.concatenate(out)

    level, side = PINCH @ DOWN, np.cross(DOWN, CLOSE)
    band = lambda P: P[np.abs(P @ DOWN - level) < 0.004]
    I = band(points(index))
    pad = I[(I @ CLOSE) > (I @ CLOSE).max() - 0.002].mean(0)  # the index pad's centre, on its face

    def triangles(gs):
        """The geoms' surface triangles (mesh faces; a box's 12), hand frame."""
        R, out = data.xmat[hb].reshape(3, 3), []
        for g in gs:
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                i = model.geom_dataid[g]
                v = model.mesh_vert[model.mesh_vertadr[i]:model.mesh_vertadr[i] + model.mesh_vertnum[i]]
                f = model.mesh_face[model.mesh_faceadr[i]:model.mesh_faceadr[i] + model.mesh_facenum[i]]
            else:
                v = np.array(np.meshgrid(*[(-x, x) for x in model.geom_size[g]], indexing="ij")).reshape(3, -1).T
                f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                              [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
            out.append(((v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g] - data.xpos[hb]) @ R)[f])
        return np.concatenate(out)

    def lean(T, toward):
        """Degrees a face leans toward the other jaw lower down: area-weighted outward normal of the triangles facing
        `toward` within 6 mm of the contact level (a normal tilted up means the face comes closer lower down)."""
        n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        n *= np.sign(((T.mean(1) - T.reshape(-1, 3).mean(0)) * n).sum(1))[:, None]  # outward
        u = n / np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]
        sel = (u @ toward > 0.6) & (np.abs(T.mean(1) @ DOWN - level) < 0.006)
        m = n[sel].sum(0)
        return float(-np.degrees(np.arcsin(m @ DOWN / np.linalg.norm(m)))) if sel.any() else float("nan")

    print(f"index face lean {lean(triangles(index[-1:]), CLOSE):.0f} deg")  # the tip link's pad
    print("opening  gap_cm  thumb_off_line_mm  thumb_tip_vs_index_tip_mm (+ = lower)  thumb_face_lean_deg")
    for o in np.linspace(0, 1, 11):
        data.qpos[[model.jnt_qposadr[model.joint(j).id] for j in HAND]] = hand_q(o)
        mujoco.mj_kinematics(model, data)
        T = points(thumb)
        Tb = band(T)
        face = Tb[(Tb @ CLOSE) < (Tb @ CLOSE).min() + 0.003] if len(Tb) else None
        gap = (face @ CLOSE).mean() - pad @ CLOSE if face is not None else float("nan")
        off = (face.mean(0) - pad) @ side if face is not None else float("nan")
        print(f"{o:6.2f}  {gap * 100:6.2f}  {off * 1000:6.1f}  {((T @ DOWN).max() - (points(index) @ DOWN).max()) * 1000:6.1f}"
              f"  {lean(triangles(thumb), -CLOSE):6.0f}")
    print(f"index pad to pinch site: {(PINCH - pad) @ CLOSE * 100:.2f} cm along CLOSE, {(PINCH - pad) @ side * 1000:.1f} mm"
          f" across; index tip {((points(index) @ DOWN).max() - level) * 1000:.1f} mm below the site")


def cmd_reach():
    """IK at points over the workspace, trays and shelf, every 30 degrees of heading: the share reached within 3 mm with
    the claw within 10 degrees of straight down and the arm touching neither itself nor the table, the worst tilt
    needed, and how many headings were lost to contact alone."""
    model = build()
    data, scratch = mujoco.MjData(model), mujoco.MjData(model)
    start(model, data)
    arm = G1Arm(model, data, scratch)
    regions = {"objects": [(x, y, z) for x in (0.12, 0.17, 0.22) for y in (-0.13, -0.06, 0.0, 0.06, 0.13) for z in (0.015, 0.05)],
               "red tray": [(x, y, 0.03) for x in (0.11, 0.14, 0.17) for y in (0.16, 0.19, 0.22)],
               "blue tray": [(x, y, 0.03) for x in (0.11, 0.14, 0.17) for y in (-0.16, -0.19, -0.22)],
               "shelf": [(x, y, 0.09) for x in (0.29, 0.32, 0.35) for y in (-0.05, 0.0, 0.05)],
               "hover": [(x, y, 0.15) for x in (0.12, 0.22, 0.32) for y in (-0.19, 0.0, 0.19)],
               "park": [(0.05, -0.30, 0.15)]}  # bench_general.py PARK
    for name, pts in regions.items():
        ok, n, touch, tilts, misses = 0, 0, 0, [], []
        for p in pts:
            for yaw in np.radians(np.arange(0, 180, 30)):
                arm.roll = None
                arm.turn_jaw(np.array(p), [yaw])
                q, err, tilt = ik(arm, np.array(p), arm.roll, report=True, posture=arm.posture)
                c = arm.clashes(q)
                n += 1
                good = err < 0.003 and tilt < np.radians(10)
                ok += good and not c
                touch += good and bool(c)
                tilts.append(np.degrees(tilt))
                if not good or c:
                    misses.append((p, round(float(np.degrees(yaw))), round(err * 1000, 1), round(float(np.degrees(tilt))),
                                   sorted(set(c))))
        print(f"{name:10s} {ok}/{n} reached, {touch} more only with contact; tilt median {np.median(tilts):.0f} max"
              f" {np.max(tilts):.0f} deg; misses (point, heading, mm, tilt, contacts) e.g. {misses[:4]}")


def cmd_claw():
    """Scripted pick and place without cameras: the object's points from rays cast straight down on it (a perfect
    overhead depth camera), then grasp.plan -> pick_at -> lift -> place_on the red tray, blue tray or shelf (in turn),
    for small boxes, balls and cylinders at three spots. Prints each (outcome, plan, guard counts, the arm's contacts
    with itself or the table after the pick), then lifted and placed per shape."""
    from so101.grasp import pick_at, place_on, plan
    from so101.objects import body_name, build as build_with
    from so101.robot import Guard, SimArm
    from so101.skills import run

    def thing(kind, size, z, axis=None):
        return dict(shape="prim", type=kind, size=size, z=z, quat_axis=axis, name="red thing", rgba=[0.8, 0.2, 0.2, 1],
                    mass=0.05, friction=1.0)
    cases = {"ball r1.2": thing("sphere", [0.012, 0, 0], 0.012), "ball r2": thing("sphere", [0.02, 0, 0], 0.02),
             "cylinder r1 h3": thing("cylinder", [0.01, 0.015, 0], 0.015),
             "cylinder r2 h3": thing("cylinder", [0.02, 0.015, 0], 0.015),
             "lying cylinder r1 l6": thing("cylinder", [0.01, 0.03, 0], 0.01, axis="y"),
             "box 2x6x3": thing("box", [0.01, 0.03, 0.015], 0.015), "box 3x3x3": thing("box", [0.015, 0.015, 0.015], 0.015),
             "box 1.5x4x2": thing("box", [0.0075, 0.02, 0.01], 0.01)}
    spots = [(0.17, 0.0, 0.3), (0.15, -0.10, 1.2), (0.20, 0.10, -0.6)]  # x, y, heading
    lifted, placed, places = {}, {}, ("red_tray", "blue_tray", "shelf")
    for name, spec in cases.items():
        for i, (x, y, a) in enumerate(spots):
            model = build_with([spec], [(x, y, a)], SCENE, prepare)
            data, scratch = mujoco.MjData(model), mujoco.MjData(model)
            start(model, data)
            guard = Guard(SimArm(model, data, robot=G1), model, mode="warn", robot=G1)
            arm = G1Arm(model, data, scratch, robot=guard)
            arm.blocked_margin = 0.06  # as bench_general.py
            arm.goal = np.array([0.05, -0.30, 0.15])  # bench_general.py PARK
            q = arm.target(arm.goal)
            for _ in range(67):
                arm.step(q)
            b = model.body(body_name(spec)).id
            pts, gid = [], np.array([-1], np.int32)
            for dx in np.arange(-0.06, 0.06, 0.002):
                for dy in np.arange(-0.06, 0.06, 0.002):
                    p0 = np.array([x + dx, y + dy, 0.3])
                    d = mujoco.mj_ray(model, data, p0, np.array([0, 0, -1.0]), None, 1, -1, gid)
                    if d > 0 and model.geom_bodyid[gid[0]] == b:
                        pts.append(p0 - [0, 0, d])
            g = plan(np.array(pts).reshape(-1, 3), np.zeros((0, 3)), 0.0, hand=Hand, check=arm.can_grasp)
            if g is None:
                print(f"{name:21s} ({x:.2f}, {y:+.2f}) no grasp fits", flush=True)
                lifted.setdefault(name, []).append(False)
                continue
            hang, hover = g.z - g.support, max(0.12, 0.06 + g.z - g.support + 0.025, g.top + 0.02)  # as bench_general.py
            ok, why = run(arm, pick_at(arm, g, hover))
            up = bool(ok and data.xpos[b][2] > 0.05)
            lifted.setdefault(name, []).append(up)
            contacts = sorted(set(arm.clashes(arm.q[:arm.n_arm])))
            put = False
            if ok:
                place = places[(i + len(lifted)) % 3]
                c = data.xpos[model.body(place).id].copy()
                surface = 0.06 if place == "shelf" else 0.0
                done, why2 = run(arm, place_on(arm, c[:2], surface, hang, max(hover, surface + hang + 0.03)))
                for _ in range(80):
                    arm.step(arm.target(arm.goal))
                p = data.xpos[b]
                put = bool(np.all(np.abs(p[:2] - c[:2]) < 0.045) and surface - 0.005 < p[2] < surface + 0.1)
                why = f"{why}, {why2} {place}"
            placed.setdefault(name, []).append(put)
            print(f"{name:21s} ({x:.2f}, {y:+.2f}) {why:21s} lifted={up} placed={put}  width {g.width * 100:.1f} cm, opening"
                  f" {g.gap * 100:.1f} cm, planned heading {np.degrees(g.yaw):.0f} (can grasp: {arm.can_grasp(g)}),"
                  f" contacts {contacts}, guard {({k: v for k, v in guard.counts.items() if v})},"
                  f" lowest tip {guard.lowest * 1000:.0f} mm", flush=True)
    for label, d in (("lifted", lifted), ("placed", placed)):
        print(f"{label}:", {k: f"{sum(v)}/{len(v)}" for k, v in d.items()},
              f"total {sum(map(sum, d.values()))}/{sum(map(len, d.values()))}")


if __name__ == "__main__":
    for a in sys.argv[2:]:
        if a.startswith("--pelvis="):
            PELVIS[[0, 2]] = [float(v) for v in a.split("=")[1].split(",")]
        if a.startswith("--thumb-force="):
            THUMB_FORCE = float(a.split("=")[1])
        if a.startswith("--index="):  # hand tuning: index base and tip joints
            INDEX = tuple(float(v) for v in a.split("=")[1].split(","))
    {"hand": cmd_hand, "reach": cmd_reach, "claw": cmd_claw}[sys.argv[1]]()
