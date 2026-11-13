"""isaac/grab_policy.py: the learned grab shared by g1_grasp_rl.py and walk_grasp.py, CPU only."""
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "isaac"))
import grab_policy as G  # noqa: E402

from so101 import g1  # noqa: E402

_U, _, _Vt = torch.linalg.svd(torch.tensor(g1.AXES, dtype=torch.float64))
H = (_U @ _Vt).float()  # g1.AXES is orthonormal only to 4 decimals: the nearest rotation, so the checks can be tight
DOWN, CLOSE = H[:, 0], H[:, 1]


def random_quats(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    axis = torch.randn(n, 3, generator=g)
    return G.quat_from_angle_axis(torch.rand(n, generator=g) * 2 * math.pi - math.pi, axis)


def test_quat_helpers_round_trip():
    q = random_quats(200)
    q = torch.where(q[:, :1] < 0, -q, q)
    assert torch.allclose(G.quat_from_matrix(G.matrix_from_quat(q)), q, atol=1e-5)
    # multiplication matches matrix products
    a, b = random_quats(50, 1), random_quats(50, 2)
    assert torch.allclose(G.matrix_from_quat(G.quat_mul(a, b)), G.matrix_from_quat(a) @ G.matrix_from_quat(b), atol=1e-5)
    # identity and a 90-degree turn about z
    assert torch.allclose(G.quat_from_matrix(torch.eye(3)[None]), torch.tensor([[1.0, 0, 0, 0]]))
    q90 = G.quat_from_angle_axis(torch.tensor([math.pi / 2]), torch.tensor([[0.0, 0, 1]]))
    assert torch.allclose(G.matrix_from_quat(q90)[0] @ torch.tensor([1.0, 0, 0]), torch.tensor([0.0, 1, 0]), atol=1e-6)


def test_zero_adjustment_is_the_scripted_pinch():
    N = 4
    c0 = torch.tensor([[0.2, 0.05, 0.02]] * N)
    h0 = torch.tensor([0.3, 1.2, -2.0, 3.0])
    R0 = G.matrix_from_quat(random_quats(N, 3))
    c, q = G.grasp_goals(torch.zeros(N, 8), c0, h0, R0, H)
    assert torch.allclose(c, c0)
    R = G.matrix_from_quat(q)
    down = R @ DOWN  # the hand's DOWN axis points straight down
    assert torch.allclose(down, torch.tensor([[0.0, 0, -1.0]] * N), atol=1e-5)
    close = R @ CLOSE  # the closing axis along the heading, either way
    for i in range(N):
        e = torch.tensor([math.cos(h0[i]), math.sin(h0[i]), 0.0])
        assert abs(float(close[i] @ e)) == pytest.approx(1.0, abs=1e-5)
    # the sign nearer the start orientation wins: start the hand at one of the two and it must be chosen
    for sgn in (1.0, -1.0):
        cw = sgn * torch.stack([torch.cos(h0), torch.sin(h0), torch.zeros(N)], 1)
        W = torch.stack([torch.tensor([[0.0, 0, -1.0]] * N), cw, torch.linalg.cross(torch.tensor([[0.0, 0, -1.0]] * N), cw)], 2)
        Rs = W @ H.T
        _, qs = G.grasp_goals(torch.zeros(N, 8), c0, h0, Rs, H)
        assert torch.allclose(G.matrix_from_quat(qs), Rs, atol=1e-4)


def test_adjustments_move_the_pinch_as_named():
    c0, h0 = torch.tensor([[0.2, 0.0, 0.02]]), torch.tensor([0.0])  # closing along +x
    dw, cw = torch.tensor([[0.0, 0, -1.0]]), torch.tensor([[1.0, 0, 0]])
    R0 = torch.stack([dw, cw, torch.linalg.cross(dw, cw)], 2) @ H.T  # the hand already closing along +x
    adj = torch.zeros(1, 8)
    adj[0, 0], adj[0, 1], adj[0, 2] = 0.01, 0.02, 0.03  # along the closing direction, across (z x cw = +y), up
    c, _ = G.grasp_goals(adj, c0, h0, R0, H)
    assert torch.allclose(c[0], torch.tensor([0.21, 0.02, 0.05]), atol=1e-6)
    c, _ = G.grasp_goals(torch.zeros(1, 8), torch.tensor([[0.2, 0.0, -0.01]]), h0, R0, H, z_min=0.008)
    assert float(c[0, 2]) == pytest.approx(0.008)
    adj = torch.zeros(1, 8)
    adj[0, 3] = math.pi / 2  # heading turned: the closing axis now along y
    _, q = G.grasp_goals(adj, c0, h0, R0, H)
    close = G.matrix_from_quat(q)[0] @ CLOSE
    assert abs(float(close[1])) == pytest.approx(1.0, abs=1e-4)
    adj = torch.zeros(1, 8)
    adj[0, 5] = 0.2  # a tilt keeps the pinch point, turns the hand 0.2 rad off straight down
    _, q = G.grasp_goals(adj, c0, h0, R0, H)
    down = G.matrix_from_quat(q)[0] @ DOWN
    assert float(torch.arccos(-down[2])) == pytest.approx(0.2, abs=1e-4)


def test_policy_io_and_scaling(tmp_path):
    pol = G.Policy(14, 8)
    feat = [G.features("box", [0.01, 0.02, 0.015], 0.4, 0.18, -0.05, 0.015)]
    assert len(feat[0]) == 14 and feat[0][:6] == [1, 0, 0, 0, 0, 0]
    assert torch.equal(G.adjustments(pol, feat), torch.zeros(1, 8))  # a fresh policy is the scripted pinch
    torch.save(pol.state_dict(), tmp_path / "p.pt")
    pol2 = G.load(tmp_path / "p.pt")
    assert pol2.net[-1].out_features == 8
    with torch.no_grad():  # outputs are clipped at +-2 units of SCALE
        pol2.net[-1].bias.fill_(10.0)
    a = G.adjustments(pol2, feat)
    assert torch.allclose(a, 2 * G.SCALE)
    pol5 = G.Policy(14, 5)
    assert G.adjustments(pol5, feat).shape == (1, 8)  # the 5-parameter policy leaves the last three at zero
    close_to, rate = G.close_params(a)
    assert float(close_to[0]) == pytest.approx(0.7) and float(rate[0]) == pytest.approx(2.0)


def test_robot_frame_matches_the_fixed_base_scene():
    # a robot facing +y (the walk's start), pelvis at (0.1, 2.0), an object 0.3 ahead and 0.05 to its left (-x)
    x, y, z = G.robot_frame([0.05, 2.3], 0.75, [0.1, 2.0], math.pi / 2, 0.72)
    assert (x, y, z) == pytest.approx((0.3 + G.PELVIS_X, 0.05, 0.03), abs=1e-9)
    spec = dict(type="box", size=[0.02, 0.01, 0.01])  # wide box: close across the short side, heading + 90 deg
    assert G.narrow_heading(spec, 0.1) == pytest.approx(0.1 + math.pi / 2)
    assert G.narrow_heading(dict(type="cylinder", size=[0.01, 0.02, 0], quat_axis="y"), 0.1) == pytest.approx(0.1 + math.pi / 2)
    assert G.narrow_heading(dict(type="sphere", size=[0.02, 0, 0], quat_axis=None), 0.1) == pytest.approx(0.1)
