"""A closed-loop grab for the walking G1: small per-step corrections on top of the open-loop pinch (path 3, 2026-11-11).

Why (docs/results.md 6.1, LOG 2026-11-11): the grab chosen once per object (scripted, or v8's 8 adjustments) leaves most
of the losses at one stage: the hand is on target, the fingers close, and a thin rolling object is pushed away
(validation, batch IK, 384 layouts: v8 lifts 211, 143 grabs never rise). Nothing in an open-loop pinch can react to
that. This module is the part a per-step policy adds, shared by the trainer (isaac/grab_rl.py) and the benchmark
(isaac/walk_grasp.py --residual), so the walk runs exactly what was trained (as grab_policy.py for v8).

The base motion is the pinch as walk_grasp.py runs it (`schedule`): raise, approach, descend, close, lift, hold, toward
a goal read once after the robot has stopped. The residual policy acts every 20 ms and gives 5 numbers in [-1, 1]:
  0-2  where the pinch-point offset should go, as a share of MAX_OFF, along the robot's forward, left and up,
  3    the hand's turn about the vertical through the pinch point, as a share of MAX_YAW,
  4    a change of the finger closing, as a share of MAX_GRIP.
Each is a target the residual moves toward by ALPHA of the remaining way per step (a low-pass filter, time constant
about 0.13 s). All zeros is exactly the base pinch. The first version added the action to the residual every step:
the policy's exploration noise (std 0.3) then walked the hand about 2 cm off over an episode and the lift rate fell
from 0.49 to 0.02 before any learning (pilot p1, 2026-11-11). With the filter, the same noise moves the offset
0.285 x std x MAX_OFF (1.7 mm at std 0.15), and it does not grow with time. Torch only (no Isaac imports): importable and testable on the CPU.
"""
import math

import torch

PHASES = [("raise", 1.0), ("approach", 1.5), ("descend", 1.0), ("close", 1.2), ("lift", 1.0), ("hold", 1.0)]  # walk_grasp.py's, after its settle
EDGES = [0.0]
for _n, _d in PHASES:
    EDGES.append(EDGES[-1] + _d)
T_TOTAL = EDGES[-1]  # 6.7 s
HIGH, LIFT = 0.20, 0.10  # walk_grasp.py --high and LIFT
N_ACT = 5
ALPHA = 0.15  # share of the way to the action's target per 20 ms step
MAX_OFF = 0.04  # m
MAX_YAW = 0.6  # rad
MAX_GRIP = 0.6  # closing fraction
SHAPES = ["box", "cylinder", "lying cylinder", "ball", "capsule", "egg"]


def schedule(t, goal_c, hand0, rate, high=HIGH, lift=LIFT):
    """The base pinch at time t (n,) seconds after the settle, as walk_grasp.py's grasp loop: the pinch point (n, 3; the
    frame goal_c and hand0 are in), the closing fraction before the per-robot limit (n,), and the phase index (n,).
    goal_c: where the pinch closes; hand0: the pinch point when the grasp starts; rate: closing speed (v8's 2 ** adj)."""
    n = t.shape[0]
    up = torch.tensor([0.0, 0.0, 1.0], device=t.device).expand(n, 3)
    edges = torch.tensor(EDGES, device=t.device)
    j = torch.clamp(torch.searchsorted(edges, t.contiguous(), right=True) - 1, 0, len(PHASES) - 1)
    dur = torch.tensor([d for _, d in PHASES], device=t.device)[j]
    frac = torch.clamp((t - edges[j]) / dur, 0.0, 1.0)
    s = torch.clamp(frac * 1.5, max=1.0)[:, None]
    top = goal_c + high * up
    pts = torch.stack([hand0 + s * (top - hand0), top + s * (0.08 - high) * up, goal_c + (1 - s) * 0.08 * up,
                       goal_c, goal_c + s * lift * up, goal_c + lift * up], 0)  # (6, n, 3)
    pinch = pts[j, torch.arange(n, device=t.device)]
    f = torch.where(j < 3, torch.zeros_like(frac), torch.where(j == 3, frac * rate, torch.ones_like(frac)))
    return pinch, f, j


def apply(state, action, mask=None):
    """One policy step into the residual state (n, 5): offset in the robot's heading frame (3), hand yaw, grip. The
    state moves ALPHA of the way to action x limit. mask (n,) bool: only these rows act (the others keep their state)."""
    lim = torch.tensor([MAX_OFF, MAX_OFF, MAX_OFF, MAX_YAW, MAX_GRIP], device=state.device)
    new = state + ALPHA * (torch.clamp(action, -1.0, 1.0) * lim - state)
    return new if mask is None else torch.where(mask[:, None], new, state)


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def quat_apply(q, v):
    w, u = q[..., :1], q[..., 1:]
    t = 2 * torch.linalg.cross(u, v, dim=-1)
    return v + w * t + torch.linalg.cross(u, t, dim=-1)


def corrected(pinch, f, goal_q, close_to, state, yaw, z_min):
    """The base pinch with the residual on it: pinch point (n, 3), hand orientation (n, 4; w, x, y, z) and closing
    fraction (n,). yaw: each robot's heading (the offset's frame turns with it). z_min: the lowest pinch height (the
    fingertips stay off the table)."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    off = torch.stack([c * state[:, 0] - s * state[:, 1], s * state[:, 0] + c * state[:, 1], state[:, 2]], 1)
    p = pinch + off
    p = torch.cat([p[:, :2], torch.maximum(p[:, 2:3], z_min[:, None] if torch.is_tensor(z_min) else torch.full_like(p[:, 2:3], z_min))], 1)
    half = state[:, 3] / 2
    turn = torch.stack([torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)], 1)
    grip = torch.clamp(torch.minimum(f, close_to) + state[:, 4], 0.0, 1.0)
    return p, quat_mul(turn, goal_q), grip


def observation(t, phase, state, obj_p, obj_q, obj_v, obj_w, pinch_p, pinch_target, hand_q, goal_c, finger_q, finger_target,
                root_p, root_q, root_w, shape_id, size):
    """The policy's inputs (n, OBS_DIM), everything in the robot's heading frame (x forward, y left, z up; the same for
    a robot standing anywhere): time and phase; the residual so far; the object relative to the pinch point, to the
    goal read at the start, and its velocity; where the pinch point is relative to where it was sent; the hand's and
    the object's orientation; the right hand's finger angles and how far they are from their targets (the fingers
    stalling on an object is how the hand feels it; this robot's hand has no contact sensor in this scene); the base's
    tilt and turn rate; the object's class and half sizes."""
    n = t.shape[0]
    yaw = yaw_of(root_q)
    inv = torch.stack([torch.cos(-yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(-yaw / 2)], 1)
    rot = lambda v: quat_apply(inv, v)  # noqa: E731
    one_hot = torch.nn.functional.one_hot(shape_id.clamp(0, len(SHAPES) - 1), len(SHAPES)).float() * (shape_id >= 0)[:, None]
    ph = torch.nn.functional.one_hot(phase, len(PHASES)).float()
    up = torch.tensor([0.0, 0.0, 1.0], device=t.device).expand(n, 3)
    parts = [(t / T_TOTAL)[:, None], ph, state / torch.tensor([MAX_OFF, MAX_OFF, MAX_OFF, MAX_YAW, MAX_GRIP], device=t.device),
             rot(obj_p - pinch_p) / 0.05, rot(obj_p - goal_c) / 0.05, rot(obj_v) / 0.2, rot(obj_w) / 5.0,
             rot(pinch_p - pinch_target) / 0.02, quat_mul(inv, hand_q), quat_mul(inv, obj_q),
             finger_q, (finger_target - finger_q) / 0.5,
             quat_apply(quat_conj(root_q), -up), rot(root_w) / 2.0, ((root_p[:, 2:3]) - 0.72) / 0.05,
             one_hot, size / 0.03]
    return torch.cat(parts, 1)


OBS_DIM = 1 + 6 + 5 + 3 + 3 + 3 + 3 + 3 + 4 + 4 + 7 + 7 + 3 + 3 + 1 + 6 + 3  # 65


def quat_conj(q):
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device)


def yaw_of(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


# --- the trained policy, read without rsl_rl (the benchmark runs it; the trainer writes rsl_rl checkpoints) -----------------
class Actor(torch.nn.Module):
    """rsl_rl 3.1.2's ActorCritic actor: observation normalisation ((x - mean) / (std + 0.01)), then an ELU MLP."""

    def __init__(self, sd):
        super().__init__()
        ws = sorted((k for k in sd if k.startswith("actor.") and k.endswith(".weight")), key=lambda k: int(k.split(".")[1]))
        layers = []
        for i, k in enumerate(ws):
            lin = torch.nn.Linear(sd[k].shape[1], sd[k].shape[0])
            lin.weight.data, lin.bias.data = sd[k].clone(), sd[k.replace("weight", "bias")].clone()
            layers += [lin] + ([torch.nn.ELU()] if i < len(ws) - 1 else [])
        self.net = torch.nn.Sequential(*layers)
        mean, std = sd.get("actor_obs_normalizer._mean"), sd.get("actor_obs_normalizer._std")
        self.register_buffer("mean", mean.clone() if mean is not None else torch.zeros(1, ws and sd[ws[0]].shape[1]))
        self.register_buffer("std", std.clone() if std is not None else torch.ones(1, ws and sd[ws[0]].shape[1]))

    def forward(self, obs):
        return self.net((obs - self.mean) / (self.std + 1e-2))


def load(path, device="cpu"):
    """The actor of an rsl_rl checkpoint (model_*.pt), in eval mode; its input and output sizes are checked."""
    sd = torch.load(path, map_location=device, weights_only=False)["model_state_dict"]
    actor = Actor(sd).to(device).eval()
    d_in, d_out = actor.net[0].in_features, actor.net[-1].out_features
    assert (d_in, d_out) == (OBS_DIM, N_ACT), f"{path}: {d_in} inputs / {d_out} outputs, this code {OBS_DIM} / {N_ACT}"
    return actor


def shape_ids(shapes):
    """Class index per object (SHAPES order); -1 for objects that are not one of the six primitives."""
    return torch.tensor([SHAPES.index(s) if s in SHAPES else -1 for s in shapes])


def describe():
    return dict(obs_dim=OBS_DIM, actions=N_ACT, alpha=ALPHA, max_off_m=MAX_OFF, max_yaw=MAX_YAW, max_grip=MAX_GRIP,
                seconds=T_TOTAL, high=HIGH, lift=LIFT)


assert abs(T_TOTAL - 6.7) < 1e-9 and math.isclose(EDGES[3], 3.5)
