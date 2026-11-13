"""Robot-frame features for BC on the walking-between-tables (locomanipulation SDG) data. Shared by nav_to_bc.py
(dataset conversion) and nav_bc_eval.py (rollouts) so both build the same observations and undo the same actions.

Poses are [x, y, z, qw, qx, qy, qz] in the env frame; "base" is env.get_base().get_pose() (the recorded base_pose).
Observations, all low-dim: goal and approach poses, wrists and object in the base frame; the planned path resampled
to PATH_K points, xy in the base frame; the 14 hand joints. Actions: wrist targets in the base frame (the env takes
them in the env frame), the 14 finger targets and the 4 lower-body numbers [vx, vy, wz, hip_height] unchanged.
"""
import numpy as np

PATH_K = 8
OBS_KEYS = ["goal_b", "approach_b", "path_b", "left_eef_b", "right_eef_b", "object_b", "hand_joint_state"]


def qmul(a, b):
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def qconj(q):
    return q * np.array([1, -1, -1, -1], dtype=q.dtype)


def qrot(q, v):
    """Rotate vectors v (..., 3) by unit quaternions q (..., 4)."""
    qv = np.concatenate([np.zeros(v.shape[:-1] + (1,), v.dtype), v], -1)
    return qmul(qmul(q, qv), qconj(q))[..., 1:]


def to_base(base, pose):
    """Pose(s) in the env frame -> base frame. base (..., 7), pose (..., 7)."""
    qi = qconj(base[..., 3:])
    return np.concatenate([qrot(qi, pose[..., :3] - base[..., :3]), qmul(qi, pose[..., 3:])], -1)


def from_base(base, pose_b):
    q = base[..., 3:]
    return np.concatenate([qrot(q, pose_b[..., :3]) + base[..., :3], qmul(q, pose_b[..., 3:])], -1)


def resample_path(pts, k=PATH_K):
    """Polyline (n, 2) -> k points evenly spaced by arc length (repeats the end if it has one point)."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) == 1:
        return np.repeat(pts, k, 0)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    t = np.linspace(0, s[-1], k)
    return np.stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])], -1)


def path_to_base(base, path_k):
    """(T, k, 2) or (k, 2) env-frame xy -> base frame xy, flattened to (..., 2k)."""
    p = np.concatenate([path_k, np.zeros(path_k.shape[:-1] + (1,))], -1) - base[..., None, :3] * [1, 1, 0]
    q = np.broadcast_to(qconj(base[..., None, 3:]), p.shape[:-1] + (4,))
    return qrot(q, p)[..., :2].reshape(p.shape[:-2] + (-1,))


def obs(base, goal, approach, path_k, left, right, obj, hands):
    """All arrays with a leading time axis (or none). Returns the BC observation dict, float32."""
    o = dict(goal_b=to_base(base, goal), approach_b=to_base(base, approach), path_b=path_to_base(base, path_k),
             left_eef_b=to_base(base, left), right_eef_b=to_base(base, right), object_b=to_base(base, obj),
             hand_joint_state=hands)
    return {k: np.asarray(v, dtype=np.float32) for k, v in o.items()}


def actions_to_base(base, a):
    a = np.array(a, dtype=np.float64)
    a[..., 0:7], a[..., 7:14] = to_base(base, a[..., 0:7]), to_base(base, a[..., 7:14])
    return a


def actions_from_base(base, a_b):
    a = np.array(a_b, dtype=np.float64)
    for s in (slice(0, 7), slice(7, 14)):
        p = a[..., s]
        p[..., 3:] /= np.linalg.norm(p[..., 3:], axis=-1, keepdims=True)  # the net's quaternions are not unit
        a[..., s] = from_base(base, p)
    return a


def on_table(obj, table, half=(1.45, 0.45), min_z=0.5):
    """Object pose (7) on the table's footprint (table pose 7, yaw from its quaternion) and off the floor."""
    local = qrot(qconj(table[3:]), obj[:3] - table[:3])
    return bool(abs(local[0]) <= half[0] and abs(local[1]) <= half[1] and obj[2] > min_z)


class TableMap:
    """Rigid map taking poses near the end table to the same place relative to the start table, and back."""

    def __init__(self, start, end):
        self.start, self.end = start, end
        self.q = qmul(start[3:], qconj(end[3:]))

    def pose(self, x):  # end-table world pose -> start-table world pose
        return from_base(self.start, to_base(self.end, x))

    def back(self, x):
        return from_base(self.end, to_base(self.start, x))

    def vec(self, v):
        return qrot(self.q, v)


def placed(m, obj, obj_vel, right_wrist):
    """The stock task's success check (task_done_pick_place: object in its target area, below 1.1 m, under 0.2 m/s on
    each axis, right wrist pulled back past x 0.26) with the end table's poses moved to the start table (m: TableMap).
    obj pose (7), obj linear velocity (3), right wrist position (3), all env frame."""
    x = m.pose(obj)
    v = np.abs(m.vec(obj_vel))
    w = m.pose(np.concatenate([right_wrist, [1.0, 0, 0, 0]]))
    return bool(0.40 < x[0] < 0.85 and 0.35 < x[1] < 0.60 and x[2] < 1.10 and w[0] < 0.26 and (v < 0.20).all())
