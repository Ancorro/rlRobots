"""The learned grab of isaac/g1_grasp_rl.py (v8: adjustments to the scripted pinch), as one module that the fixed-base
trainer and the walking scene (isaac/walk_grasp.py --grab <name> --policy <file>) share, so the walk applies exactly the
grasp the policy was trained to pick. Torch and numpy only (no Isaac imports): importable and testable on the CPU.

The policy reads the object as the fixed-base scene sees it (`features`: shape class, half sizes, narrow-side heading,
centre in the robot's frame: x forward from the pelvis, y left, z above the table top; g1_grasp_rl.py pose_tensors) and
returns up to 8 adjustments (NAMES), scaled by SCALE and clipped at +-2 units (`adjustments`). `grasp_goals` turns them
into the pinch point and hand orientation (g1_grasp_rl.py goals(), pinch "center"); `close_params` into how far and how
fast the fingers close.

Frames. The fixed-base scene (so101/assets/scene_g1.xml) has the pelvis at g1.PELVIS = (-0.08, 0, 0.02), the table top
at z = 0 and objects at x 0.12-0.21, y within +-0.14 (so101/objects.py layout). A walking robot that stopped with its
pelvis `gap` m from the table edge sees the same object at forward = gap + d (d: past the edge): pass
x = forward - 0.08 (PELVIS_X), y = left offset, z = centre height above the table top (`robot_frame`).
"""
import math

import numpy as np
import torch

SHAPES = ["box", "cylinder", "lying cylinder", "ball", "capsule", "egg"]  # one-hot order (g1_grasp_rl.py, bench_spec)
NAMES = ["along_close", "across", "height", "heading", "close", "tilt_close", "tilt_side", "close_rate"]
SCALE = torch.tensor([0.015, 0.015, 0.01, 0.35, 0.15, 0.2, 0.2, 0.5])  # m, m, m, rad, fraction, rad, rad, log2
PELVIS_X = -0.08  # the fixed-base pelvis, scene frame (g1.PELVIS[0])
CLOSE_S = 1.2  # the scripted close, seconds (PHASES "close" in both scripts)


class Policy(torch.nn.Module):
    """Gaussian policy over the adjustments: mean from a small tanh MLP (last layer starts at zero: the scripted pinch),
    one learned log std per output."""

    def __init__(self, d_in, d_out=5):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d_in, 64), torch.nn.Tanh(), torch.nn.Linear(64, 64), torch.nn.Tanh(),
                                       torch.nn.Linear(64, d_out))
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)
        self.log_std = torch.nn.Parameter(torch.full((d_out,), np.log(0.6)))

    def forward(self, x):
        return self.net(x), self.log_std.exp()


def load(path, device="cpu"):
    """A saved policy (torch.save of its state_dict); the sizes come from the weights."""
    sd = torch.load(path, map_location=device)
    pol = Policy(sd["net.0.weight"].shape[1], sd["net.4.weight"].shape[0]).to(device)
    pol.load_state_dict(sd)
    pol.eval()
    return pol


def narrow_heading(spec, heading):
    """The closing heading across the object's narrow side (g1_grasp_rl.py narrow_heading): boxes across their shorter
    footprint side, lying shapes (quat_axis y, the egg along x) across their length, the rest the object's heading."""
    if spec["type"] == "box":
        return heading if spec["size"][0] <= spec["size"][1] else heading + math.pi / 2
    if spec.get("quat_axis") == "y" or spec["type"] == "ellipsoid":
        return heading + math.pi / 2
    return heading


def features(shape, size, heading, x, y, z, size_scale=1.0):
    """The policy's 14 inputs for one object: class one-hot, half sizes (x size_scale, a perception error) / 3 cm,
    sin and cos of twice the narrow-side heading (robot frame), centre (x - 0.15) / 5 cm, y / 10 cm, z / 3 cm."""
    s = list(np.asarray(size[:3], float) * size_scale / 0.03)
    return [float(shape == k) for k in SHAPES] + s + [math.sin(2 * heading), math.cos(2 * heading), (x - 0.15) / 0.05, y / 0.1, z / 0.03]


def robot_frame(obj_xy, obj_z, pelvis_xy, yaw, table_top):
    """Object centre in the fixed-base scene's frame from a standing robot's pelvis position and yaw (numpy, one
    object): x forward from where the fixed-base pelvis sits, y left, z above the table top."""
    d = np.asarray(obj_xy, float) - np.asarray(pelvis_xy, float)
    forward = d @ [math.cos(yaw), math.sin(yaw)]
    left = d @ [-math.sin(yaw), math.cos(yaw)]
    return forward + PELVIS_X, left, obj_z - table_top


def adjustments(policy, feat):
    """The policy's mean adjustments (N x 8, physical units) for features (N x d_in); no sampling."""
    with torch.no_grad():
        a, _ = policy(torch.as_tensor(np.asarray(feat, np.float32), device=next(policy.parameters()).device))
    return SCALE.to(a.device) * torch.nn.functional.pad(a, (0, 8 - a.shape[1])).clamp(-2, 2)


# --- quaternion helpers (w, x, y, z), as isaaclab.utils.math --------------------------------------------------------------
def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def quat_from_angle_axis(angle, axis):
    half = angle[:, None] / 2
    return torch.cat([torch.cos(half), torch.sin(half) * axis / axis.norm(dim=-1, keepdim=True)], -1)


def quat_from_matrix(R):
    """Rotation matrices (N, 3, 3) -> unit quaternions (N, 4), w >= 0 (Shepperd's method)."""
    m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]
    cands = torch.stack([
        torch.stack([1 + m00 + m11 + m22, m21 - m12, m02 - m20, m10 - m01], 1),
        torch.stack([m21 - m12, 1 + m00 - m11 - m22, m01 + m10, m02 + m20], 1),
        torch.stack([m02 - m20, m01 + m10, 1 - m00 + m11 - m22, m12 + m21], 1),
        torch.stack([m10 - m01, m02 + m20, m12 + m21, 1 - m00 - m11 + m22], 1)], 1)  # (N, 4, 4): row k scaled by 4 q_k
    k = torch.argmax(torch.stack([cands[:, i, i] for i in range(4)], 1), 1)
    q = cands[torch.arange(len(R)), k]
    q = q / torch.sqrt(torch.clamp(cands[torch.arange(len(R)), k, k], min=1e-12))[:, None] / 2
    return torch.where((q[:, :1] < 0), -q, q)


def matrix_from_quat(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


# --- the adjusted grasp -------------------------------------------------------------------------------------------------
def grasp_goals(adj, c0, h0, R0, H, z_min=None, up=None):
    """Per env (g1_grasp_rl.py goals(), pinch "center"): pinch centre and hand orientation (w, x, y, z) for adjustments
    adj (N x 8) to the scripted pinch at c0 (N x 3) with closing heading h0 (N): heading + adj[3]; of the closing
    direction's two signs the one nearer the hand's start orientation R0 (N x 3 x 3); centre moved along the closing
    direction, across it and up (adj[0:3]); then tilted about the pinch point (adj[5], adj[6]) in a frame fixed by the
    heading. H: the hand frame's grasp axes (columns DOWN, CLOSE, their cross; so101/g1.py AXES). z_min clamps the
    centre height (the fixed-base scene: 8 mm above the table)."""
    N, dev = len(c0), c0.device
    up = torch.tensor([0.0, 0.0, 1.0], device=dev) if up is None else up
    h = h0 + adj[:, 3]
    dw = -up.expand(N, 3)
    best_q = best_s = best_cw = None
    for sgn in (1.0, -1.0):
        cw = sgn * torch.stack([torch.cos(h), torch.sin(h), torch.zeros_like(h)], 1)
        W = torch.stack([dw, cw, torch.linalg.cross(dw, cw)], 2)
        R = W @ H.T
        score = torch.einsum("nij,nij->n", R0, R)
        q = quat_from_matrix(R)
        if best_s is None:
            best_q, best_s, best_cw = q, score, cw
        else:
            take = score > best_s
            best_q = torch.where(take[:, None], q, best_q)
            best_cw = torch.where(take[:, None], cw, best_cw)
            best_s = torch.maximum(score, best_s)
    across = torch.linalg.cross(up.expand(N, 3), best_cw)
    c = c0 + adj[:, :1] * best_cw + adj[:, 1:2] * across + adj[:, 2:3] * up
    if z_min is not None:
        c[:, 2] = torch.clamp(c[:, 2], min=z_min)
    eh = torch.stack([torch.cos(h), torch.sin(h), torch.zeros_like(h)], 1)
    ep = torch.linalg.cross(up.expand(N, 3), eh)
    tilt = quat_mul(quat_from_angle_axis(adj[:, 5], ep), quat_from_angle_axis(adj[:, 6], eh))
    return c, quat_mul(tilt, best_q)


def close_params(adj):
    """(close_to, rate): how far the fingers close (1 = the scripted full close, at least 0.6) and the closing speed
    relative to the scripted CLOSE_S close (2 ** adj[7])."""
    return torch.clamp(1.0 - adj[:, 4].abs(), 0.6, 1.0), torch.pow(2.0, adj[:, 7])
