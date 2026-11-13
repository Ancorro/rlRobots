"""Shared SO-101 arm helpers for the decision-model tasks: jaw-down IK (inverse kinematics) and the
gripper constants. Used by the pick-place env and the scripted skills."""
import mujoco
import numpy as np

from so101.envs import ARM_JOINTS

GRIP_OPEN, GRIP_CLOSED = 1.2, -0.17  # gripper joint targets, radians


def ik_jaw_down(model, data, scratch, tip, wrist, goal, lo, hi, w=None, iters=40, roll=None):
    """Joint angles putting site `tip` at `goal` with the jaw pointing down. Five joints cannot
    always do both, so `w` trades jaw angle against position. Default: the strictest jaw-down weight
    that still reaches the goal within 3 mm, so the jaw tilts only where it must (far reach).
    (2026-10-03: the old split, strict below 5 cm and loose above, swung the wrist upright next to the
    cube and pushed it at far reach; strict everywhere could not get down to far cubes. 91 -> 99/100 picks.)
    `roll` (radians): hold wrist_roll there (to turn the jaw), solving with the other four joints.
    Solves on `scratch` (a spare MjData) so the physics state is never disturbed."""
    if w is None:
        for w in (0.03, 0.01, 0.003, 0.001):
            q = ik_jaw_down(model, data, scratch, tip, wrist, goal, lo, hi, w, iters, roll)
            scratch.qpos[:ARM_JOINTS] = q
            mujoco.mj_kinematics(model, scratch)
            if np.linalg.norm(scratch.site_xpos[tip] - goal) < 0.003:
                break
        return q
    down = np.array([0.0, 0.0, -1.0])
    scratch.qpos[:] = data.qpos

    def resid(q):
        scratch.qpos[:ARM_JOINTS] = q
        mujoco.mj_kinematics(model, scratch)
        a = scratch.site_xpos[tip] - scratch.xpos[wrist]
        return np.concatenate([goal - scratch.site_xpos[tip], w * (down - a / np.linalg.norm(a))])

    q = data.qpos[:ARM_JOINTS].copy()
    free = list(range(ARM_JOINTS)) if roll is None else list(range(ARM_JOINTS - 1))
    if roll is not None:
        q[ARM_JOINTS - 1] = roll
    eye = np.eye(ARM_JOINTS)
    for _ in range(iters):
        r = resid(q)
        if np.linalg.norm(r) < 1e-3:
            break
        J = np.stack([(resid(q + 1e-5 * eye[j]) - r) / 1e-5 for j in free], 1)
        q[free] = q[free] - np.linalg.solve(J.T @ J + 1e-4 * np.eye(len(free)), J.T @ r)
        q = np.clip(q, lo[:ARM_JOINTS], hi[:ARM_JOINTS])
    return q


def closing_yaw(model, scratch):
    """Heading (radians, world xy) of the line the jaws close along, for the pose in `scratch`."""
    f = np.mean([scratch.geom_xpos[model.geom(f"fixed_jaw_sph_tip{i}").id] for i in (1, 2, 3)], 0)
    m = np.mean([scratch.geom_xpos[model.geom(f"moving_jaw_sph_tip{i}").id] for i in (1, 2, 3)], 0)
    return np.arctan2(f[1] - m[1], f[0] - m[0])


def roll_for(model, scratch, q, yaw, lo, hi):
    """wrist_roll that makes the jaws close along heading `yaw` (either direction), other joints as in q."""
    rolls = np.linspace(lo[ARM_JOINTS - 1], hi[ARM_JOINTS - 1], 181)
    err = []
    for r in rolls:
        scratch.qpos[:ARM_JOINTS] = [*q[:ARM_JOINTS - 1], r]
        mujoco.mj_kinematics(model, scratch)
        e = (closing_yaw(model, scratch) - yaw) % np.pi  # closing either way works
        err.append(min(e, np.pi - e) + 0.01 * abs(r))  # prefer small rolls when equal
    return rolls[int(np.argmin(err))]
