"""The tensor logic of isaac/g1_dexsuite.py's walking-base additions (walk_dr, the action filter, task mode), as plain
torch functions so they can be tested on the CPU with N environments (tests/test_task_logic.py). Quaternions are
(w, x, y, z) as in Isaac Lab; the helpers come from isaac/grab_policy.py (same convention)."""
import math

import torch

from grab_policy import matrix_from_quat


def quat_apply(q, v):
    """Rotate vectors v (..., 3) by quaternions q (..., 4)."""
    return (matrix_from_quat(q) @ v[..., None])[..., 0]


def quat_inv(q):
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device, dtype=q.dtype)


def yaw_of(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def pitch_deg(q):
    w, x, y, z = q.unbind(-1)
    return torch.rad2deg(torch.asin(torch.clamp(2 * (w * y - z * x), -1, 1)))


# --- walk_dr: random carry motions for the legs during training -----------------------------------------------------------
def walk_dr_plan(u, still_frac=0.25):
    """Per env (u: (n, 4, 4) uniforms, the last row's first entry decides "stand still"), 4 segments [vx, vy, wz, s]:
    back off (vx -0.1..-0.3), turn (wz +-0.6..1.2), walk (vx 0.1..0.4, vy +-0.15), then stand to the end."""
    n = u.shape[0]
    seg = torch.zeros(n, 4, 4, device=u.device)
    seg[:, 0, 0] = -0.1 - 0.2 * u[:, 0, 0]
    seg[:, 1, 2] = torch.where(u[:, 1, 0] < 0.5, -1.0, 1.0) * (0.6 + 0.6 * u[:, 1, 1])
    seg[:, 2, 0] = 0.1 + 0.3 * u[:, 2, 0]
    seg[:, 2, 1] = 0.3 * (u[:, 2, 1] - 0.5)
    seg[:, :, 3] = 0.8 + 1.2 * u[:, :, 3]
    seg[:, 3, 3] = 10.0
    still = (u[:, 3, 0] < still_frac)[:, None, None]
    return torch.where(still, torch.zeros_like(seg), seg)


def walk_dr_command(plan, t):
    """The [vx, vy, wz] of each env's plan at time t (n,) since the schedule started (negative: stand)."""
    ends = torch.cumsum(plan[:, :, 3], dim=1)
    k = (t[:, None] >= ends).sum(1).clamp(max=3)
    cmd = plan[torch.arange(plan.shape[0], device=plan.device), k, :3]
    return torch.where((t >= 0)[:, None], cmd, torch.zeros_like(cmd))


# --- the action filter (EMA, rate clip, frozen joints) -----------------------------------------------------------------------
def filter_actions(filt, actions, alpha, rate_clip, keep):
    """filt (n, d) the previous filtered action, actions (n, d), keep (d,) 1/0 per joint -> (new filt, processed)."""
    filt = alpha * actions + (1.0 - alpha) * filt
    a = filt.clamp(-rate_clip, rate_clip) if rate_clip > 0 else filt
    return filt, a * keep


# --- task geometry ----------------------------------------------------------------------------------------------------------
def b_pose(root_xy, back, b_ahead, turn):
    """Table B's slab centre (x, y) and yaw: back off `back` from the stand spot (facing +x), turn by `turn`, the near
    edge b_ahead ahead; the slab is 0.6 deep so its centre is 0.3 past the edge."""
    turn_pt = (root_xy[0] - back, root_xy[1])
    d = (math.cos(turn), math.sin(turn))
    return turn_pt[0] + (b_ahead + 0.3) * d[0], turn_pt[1] + (b_ahead + 0.3) * d[1], turn


def b_frame(bx, by, byaw, device):
    """(edge centre (2,), forward unit (2,), side unit (2,)) of table B in the env frame."""
    d = torch.tensor([math.cos(byaw), math.sin(byaw)], dtype=torch.float32, device=device)
    side = torch.tensor([-math.sin(byaw), math.cos(byaw)], dtype=torch.float32, device=device)
    edge = torch.tensor([bx, by], dtype=torch.float32, device=device) - 0.3 * d
    return edge, d, side


def gap_to_edge(p_xy, edge, d):
    """Distance from env-frame positions (n, 2) to the edge line, along d (positive: short of the edge)."""
    return ((edge - p_xy) * d).sum(1)


def set_down_goal(rp, rq, org, edge, d, rest_z, place_d):
    """The put-down target in the root frame: place_d past B's edge along d at the robot's own lateral spot, at the
    object's resting root height (env frame) + 5 mm. rp, rq: world root pose (n, 3), (n, 4); org (n, 3); rest_z (n,)."""
    p_xy = rp[:, :2] - org[:, :2]
    gap = gap_to_edge(p_xy, edge, d)
    goal_env = torch.cat([p_xy + (gap + place_d)[:, None] * d, (rest_z + 0.005)[:, None]], 1)
    return quat_apply(quat_inv(rq), goal_env + org - rp)


def on_table_b(o_env, edge, d, side, lin_vel, half_width=0.8, depth=0.6):
    """Resting on B's slab: within its footprint, above its top and below 12 cm, slower than 5 cm/s."""
    along = ((o_env[:, :2] - edge) * d).sum(1)
    across = ((o_env[:, :2] - edge) * side).sum(1)
    return (along > 0) & (along < depth) & (across.abs() < half_width) & (o_env[:, 2] > 0.0) & (o_env[:, 2] < 0.12) \
        & (lin_vel.norm(dim=1) < 0.05)


def tilt_deg(q, start_q, long_axis, lying):
    """Standing shapes: the up axis's angle from its start; lying shapes: the long axis from horizontal. (n,) degrees."""
    n = q.shape[0]
    up = torch.tensor([0.0, 0.0, 1.0], device=q.device).expand(n, 3)
    zq, zs = quat_apply(q, up), quat_apply(start_q, up)
    axis = torch.rad2deg(torch.arccos(torch.clamp((zq * zs).sum(1), -1, 1)))
    flat = torch.rad2deg(torch.arcsin(torch.clamp((quat_apply(q, long_axis) * up).sum(1).abs(), 0, 1)))
    return torch.where(lying, flat, axis)


def heading_error(target_yaw, q):
    return torch.remainder(target_yaw - yaw_of(q) + math.pi, 2 * math.pi) - math.pi


# --- training monitor: stop when learning has stopped ------------------------------------------------------------------------
def episode_stat(ep_infos, key):
    """Mean of `key` over rsl_rl's ep_infos (Isaac Lab's per-iteration episode logs; each value a scalar tensor or
    float); NaN when the key is absent."""
    vals = []
    for ep in ep_infos or []:
        if key in ep:
            v = ep[key]
            vals.append(float(v.mean()) if hasattr(v, "mean") else float(v))
    return float(sum(vals) / len(vals)) if vals else float("nan")


class LearnMonitor:
    """Plateau / divergence detection on a per-iteration learning signal (the success reward's mean, else the episode
    reward). Windows of `window` iterations; best = the best window mean so far. After `patience` iterations without a
    window beating best by min_gain (relative, at least 1e-3 absolute): "plateau". A window below half of the first
    full window after patience: "diverged". patience 0: never stops."""

    def __init__(self, patience, min_gain=0.05, window=50):
        self.patience, self.min_gain, self.window = patience, min_gain, window
        self.vals, self.best, self.best_it, self.first = [], float("-inf"), 0, None

    def update(self, it, value):
        if value != value:  # NaN: nothing to judge
            return ""
        self.vals.append(value)
        if len(self.vals) < self.window:
            return ""
        w = sum(self.vals[-self.window:]) / self.window
        if self.first is None:
            self.first = w
        if w > self.best + max(abs(self.best) * self.min_gain, 1e-3):
            self.best, self.best_it = w, it
        if self.patience <= 0 or it - self.best_it < self.patience:
            return ""
        if self.first > 0 and w < 0.5 * self.first:
            return "diverged"
        return "plateau"
