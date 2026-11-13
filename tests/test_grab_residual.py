"""isaac/grab_residual.py on the CPU: the base pinch, the residual filter, the observation size, the actor loader, and
isaac/walk_grasp.py --residual end to end on the fake Isaac (tests/fake_isaac.py)."""
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "isaac"))
sys.path.insert(0, str(REPO / "tests"))
import grab_residual as GR  # noqa: E402


def test_schedule_matches_the_walk_phases():
    t = torch.tensor([0.0, 0.5, 1.2, 2.9, 3.6, 5.0, 6.69])
    n = len(t)
    goal, hand0 = torch.tensor([[0.0, 1.2, 0.74]]).repeat(n, 1), torch.tensor([[0.1, 0.9, 0.9]]).repeat(n, 1)
    p, f, j = GR.schedule(t, goal, hand0, torch.ones(n))
    assert j.tolist() == [0, 0, 1, 2, 3, 4, 5]
    assert torch.allclose(p[0], hand0[0]) and abs(float(p[4, 2]) - 0.74) < 1e-6 and abs(float(p[6, 2]) - 0.84) < 1e-6
    assert float(f[3]) == 0.0 and 0 < float(f[4]) < 0.2 and float(f[5]) == 1.0


def test_zero_action_is_the_base_pinch_and_the_filter_is_bounded():
    n = 5
    s = GR.apply(torch.zeros(n, GR.N_ACT), torch.zeros(n, GR.N_ACT))
    assert float(s.abs().max()) == 0.0
    q = torch.tensor([[0.7071, 0.0, 0.0, 0.7071]]).repeat(n, 1)
    p, f = torch.rand(n, 3) + 1.0, torch.rand(n)
    pc, qc, g = GR.corrected(p, f, q, torch.ones(n), s, torch.zeros(n), 0.0)
    assert torch.allclose(pc, p) and torch.allclose(qc, q) and torch.allclose(g, f)
    for _ in range(200):  # a constant full action converges to the limit and never passes it
        s = GR.apply(s, torch.full((n, GR.N_ACT), 5.0))
    lim = torch.tensor([GR.MAX_OFF] * 3 + [GR.MAX_YAW, GR.MAX_GRIP])
    assert torch.allclose(s, lim.expand(n, 5), atol=1e-4)
    kept = GR.apply(s, torch.zeros(n, GR.N_ACT), mask=torch.zeros(n, dtype=torch.bool))
    assert torch.equal(kept, s)


def test_observation_size_and_actor_loader(tmp_path):
    n = 4
    q = torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1)
    z3 = torch.zeros(n, 3)
    o = GR.observation(torch.zeros(n), torch.zeros(n, dtype=torch.long), torch.zeros(n, GR.N_ACT), z3, q, z3, z3, z3, z3, q, z3,
                       torch.zeros(n, 7), torch.zeros(n, 7), torch.tensor([[0.0, 0.0, 0.72]]).repeat(n, 1), q, z3,
                       GR.shape_ids(["box", "egg", "pool", "ball"]), torch.full((n, 3), 0.02))
    assert o.shape == (n, GR.OBS_DIM) and torch.isfinite(o).all()
    sd = {"actor.0.weight": torch.randn(8, GR.OBS_DIM), "actor.0.bias": torch.zeros(8), "actor.2.weight": torch.zeros(GR.N_ACT, 8),
          "actor.2.bias": torch.zeros(GR.N_ACT), "actor_obs_normalizer._mean": torch.zeros(1, GR.OBS_DIM),
          "actor_obs_normalizer._std": torch.ones(1, GR.OBS_DIM)}
    path = tmp_path / "model_0.pt"
    torch.save({"model_state_dict": sd}, path)
    with torch.no_grad():
        a = GR.load(str(path))(o)
    assert a.shape == (n, GR.N_ACT) and float(a.abs().max()) == 0.0  # a zeroed last layer: the base pinch


def test_walk_with_a_zero_residual(tmp_path, monkeypatch, capsys):
    from test_walk_grasp_fake import run_walk
    sd = {"actor.0.weight": torch.randn(8, GR.OBS_DIM), "actor.0.bias": torch.zeros(8), "actor.2.weight": torch.zeros(GR.N_ACT, 8),
          "actor.2.bias": torch.zeros(GR.N_ACT)}
    path = tmp_path / "model_0.pt"
    torch.save({"model_state_dict": sd}, path)
    out, rec = run_walk(tmp_path, monkeypatch, capsys, "--carry", "--rand_layout", "--shape=box,ball", "--n=2", "--split=val",
                        "--residual", str(path), "--residual_name", "zero")
    assert rec[0]["policy"] == "scripted+zero" and rec[0]["checkpoint"] and len(rec[0]["rows"]) == 4
    assert all(max(r["residual"]) == 0.0 for r in rec[0]["rows"])
