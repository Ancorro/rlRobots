"""isaac/task_logic.py with N environments: shapes, dtypes and the geometry of the task mode and walk_dr."""
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "isaac"))
import grab_policy as G  # noqa: E402
import task_logic as TL  # noqa: E402

N = 7


def yaw_quat(yaw):
    return torch.stack([torch.cos(yaw / 2), 0 * yaw, 0 * yaw, torch.sin(yaw / 2)], -1)


def test_walk_dr_plan_and_command_shapes():
    torch.manual_seed(0)
    plan = TL.walk_dr_plan(torch.rand(N, 4, 4))
    still = (plan == 0).all(-1).all(-1)
    assert plan.shape == (N, 4, 4) and (plan[~still, :, 3] > 0).all()  # still envs: an all-zero plan (stand throughout)
    # before the schedule: stand; in segment 0: backing off or standing still; after every segment: stand
    assert torch.equal(TL.walk_dr_command(plan, torch.full((N,), -1.0)), torch.zeros(N, 3))
    c0 = TL.walk_dr_command(plan, torch.zeros(N))
    assert c0.shape == (N, 3) and ((c0[:, 0] < 0) | (c0 == 0).all(1)).all()
    assert torch.equal(TL.walk_dr_command(plan, torch.full((N,), 100.0)), torch.zeros(N, 3))
    assert 0 < still.sum() < N or N < 8  # some envs stand the whole episode (a quarter on average)
    # per-env time indexing: env i in segment i % 4
    u = torch.rand(N, 4, 4); u[:, 3, 0] = 1.0  # nobody stands
    plan = TL.walk_dr_plan(u)
    ends = torch.cumsum(plan[:, :, 3], 1)
    t = ends[:, 1] + 0.01  # just into segment 2 (walk)
    c = TL.walk_dr_command(plan, t)
    assert (c[:, 0] > 0).all() and torch.allclose(c, plan[:, 2, :3])


def test_filter_actions():
    filt = torch.zeros(N, 9)
    a = torch.full((N, 9), 2.0)
    keep = torch.tensor([0.0, 0.0] + [1.0] * 7)
    filt, out = TL.filter_actions(filt, a, 0.4, 0.5, keep)
    assert filt.shape == out.shape == (N, 9) and torch.allclose(filt, torch.full((N, 9), 0.8))
    assert torch.allclose(out[:, 2:], torch.full((N, 7), 0.5)) and (out[:, :2] == 0).all()  # clipped, waist frozen
    _, out = TL.filter_actions(filt, a, 1.0, 0.0, torch.ones(9))
    assert torch.allclose(out, a)  # no filter, no clip


def test_b_pose_and_frame():
    bx, by, byaw = TL.b_pose((-0.08, 0.0), 0.45, 0.90, -math.pi / 2)  # back off, turn right, B ahead (facing -y)
    assert bx == pytest.approx(-0.53) and by == pytest.approx(-1.2) and byaw == pytest.approx(-math.pi / 2)
    edge, d, side = TL.b_frame(bx, by, byaw, "cpu")
    assert torch.allclose(edge, torch.tensor([-0.53, -0.9])) and torch.allclose(d, torch.tensor([0.0, -1.0]), atol=1e-6)
    p = torch.tensor([[-0.53, -0.5], [-0.53, -0.72]])  # at the turn spot + 0.4 and 0.18 short of the edge
    assert torch.allclose(TL.gap_to_edge(p, edge, d), torch.tensor([0.4, 0.18]), atol=1e-6)


def test_set_down_goal_in_root_frame():
    bx, by, byaw = TL.b_pose((-0.08, 0.0), 0.45, 0.90, -math.pi / 2)
    edge, d, side = TL.b_frame(bx, by, byaw, "cpu")
    org = torch.tensor([[3.0, 0.0, 0.0]] * N)
    rq = yaw_quat(torch.full((N,), -math.pi / 2))  # the robot faces B
    rp = org + torch.tensor([-0.53, -0.72, 0.02])  # 0.18 short of the edge
    rest = torch.full((N,), 0.03)
    g = TL.set_down_goal(rp, rq, org, edge, d, rest, 0.14)
    assert g.shape == (N, 3)
    assert torch.allclose(g[:, 0], torch.full((N,), 0.32), atol=1e-5)  # 0.18 + 0.14 ahead
    assert torch.allclose(g[:, 1], torch.zeros(N), atol=1e-5) and torch.allclose(g[:, 2], torch.full((N,), 0.015), atol=1e-5)


def test_on_table_b_and_tilt():
    bx, by, byaw = TL.b_pose((-0.08, 0.0), 0.45, 0.90, -math.pi / 2)
    edge, d, side = TL.b_frame(bx, by, byaw, "cpu")
    o = torch.tensor([[-0.53, -1.05, 0.03], [-0.53, -0.5, 0.03], [-0.53, -1.05, 0.3], [0.4, -1.05, 0.03]])  # on B; short; high; off the side
    r = TL.on_table_b(o, edge, d, side, torch.zeros(4, 3))
    assert r.tolist() == [True, False, False, False]
    assert not TL.on_table_b(o[:1], edge, d, side, torch.tensor([[0.2, 0.0, 0.0]]))[0]  # moving
    q0 = torch.tensor([[1.0, 0, 0, 0]] * 3)
    q = G.quat_from_angle_axis(torch.tensor([0.0, 0.3, 0.3]), torch.tensor([[1.0, 0, 0]] * 3))
    long_axis = torch.tensor([[0, 1.0, 0]] * 3)
    lying = torch.tensor([False, False, True])
    t = TL.tilt_deg(q, q0, long_axis, lying)
    assert t.shape == (3,) and t[0] == pytest.approx(0.0, abs=1e-4) and t[1] == pytest.approx(math.degrees(0.3), abs=1e-3)
    assert t[2] == pytest.approx(math.degrees(0.3), abs=1e-3)  # a lying shape rolled about x: its y axis leaves the horizontal


def test_yaw_pitch_heading():
    q = yaw_quat(torch.tensor([0.0, 1.0, -2.0]))
    assert torch.allclose(TL.yaw_of(q), torch.tensor([0.0, 1.0, -2.0]), atol=1e-6)
    assert torch.allclose(TL.pitch_deg(q), torch.zeros(3), atol=1e-5)
    qp = G.quat_from_angle_axis(torch.tensor([0.2]), torch.tensor([[0.0, 1.0, 0]]))
    assert TL.pitch_deg(qp)[0] == pytest.approx(math.degrees(0.2), abs=1e-3)
    e = TL.heading_error(torch.full((3,), -math.pi / 2), q)
    assert e.shape == (3,) and abs(float(e[0])) == pytest.approx(math.pi / 2, abs=1e-6)


def test_learn_monitor_plateau_and_divergence():
    m = TL.LearnMonitor(patience=100, min_gain=0.05, window=10)
    # improving for 60 iterations: never stops
    for it in range(60):
        assert m.update(it, 0.1 + 0.01 * it) == ""
    best_it = m.best_it
    # flat for 100 more: plateau at patience
    out = [m.update(60 + k, 0.7) for k in range(110)]
    assert "plateau" in out and out.index("plateau") == (best_it + 100 - 60)
    m = TL.LearnMonitor(patience=50, window=10)
    for it in range(20):
        m.update(it, 1.0)
    out = [m.update(20 + k, 0.2) for k in range(80)]
    assert "diverged" in out and "plateau" not in out[:out.index("diverged")]
    assert TL.LearnMonitor(patience=0, window=5).update(1000, 0.0) == ""  # patience 0: never
    for it in range(100):  # NaN values are ignored
        assert TL.LearnMonitor(patience=5, window=2).update(it, float("nan")) == ""


def test_episode_stat():
    eps = [{"Episode_Reward/success": torch.tensor(0.5)}, {"Episode_Reward/success": 0.1}, {"other": 1.0}]
    assert TL.episode_stat(eps, "Episode_Reward/success") == pytest.approx(0.3)
    assert TL.episode_stat([], "x") != TL.episode_stat([], "x")  # NaN
