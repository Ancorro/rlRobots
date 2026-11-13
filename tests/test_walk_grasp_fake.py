"""isaac/walk_grasp.py end to end on the fake Isaac (tests/fake_isaac.py): every code path runs with real tensor shapes."""
import json
import runpy
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tests"))
import fake_isaac  # noqa: E402


def run_walk(tmp_path, monkeypatch, capsys, *argv):
    fake_isaac.install()
    monkeypatch.setattr(sys, "argv", ["walk_grasp.py", "--out", str(tmp_path), *argv])
    monkeypatch.setattr("os._exit", lambda code=0: (_ for _ in ()).throw(SystemExit(code)))
    with pytest.raises(SystemExit) as e:
        runpy.run_path(str(REPO / "isaac" / "walk_grasp.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert e.value.code == 0, out[-3000:]
    rec = [json.loads(l[len("BENCH_RESULT "):]) for l in out.splitlines() if l.startswith("BENCH_RESULT ")]
    return out, rec


def test_scripted_carry_random_layouts(tmp_path, monkeypatch, capsys):
    out, rec = run_walk(tmp_path, monkeypatch, capsys, "--carry", "--rand_layout", "--shape=box,cylinder,lying-cylinder,ball,capsule,egg", "--n=2", "--split=val")
    assert len(rec) == 1 and rec[0]["bench"] == "walk_carry" and rec[0]["policy"] == "scripted" and len(rec[0]["rows"]) == 12
    r = rec[0]["rows"][0]
    for k in ("seed", "shape", "success", "fell", "upright", "released", "lifted", "placed", "level_deg", "moved_walk_cm", "moved_cm", "tilt_hold"):
        assert k in r, k
    assert all(700000 <= x["seed"] < 800000 for x in rec[0]["rows"])
    assert "walk_carry success" in out and rec[0]["settings"]["host"]


def test_learned_grab_zero_policy_and_fixed_layout(tmp_path, monkeypatch, capsys):
    sys.path.insert(0, str(REPO / "isaac"))
    import grab_policy
    pol = tmp_path / "zero.pt"
    torch.save(grab_policy.Policy(14, 8).state_dict(), pol)
    out, rec = run_walk(tmp_path, monkeypatch, capsys, "--carry", "--shape=box,ball", "--n=3", "--split=test",
                        "--grab", "zero", "--policy", str(pol), "--noise", "0.4")
    assert rec[0]["policy"] == "zero" and rec[0]["layout"] == "fixed" and rec[0]["checkpoint"] and len(rec[0]["rows"]) == 6
    assert all(abs(v) < 1e-9 for v in rec[0]["rows"][0]["adj"].values())
    assert "learned grab zero" in out


def test_grasp_only_no_carry(tmp_path, monkeypatch, capsys):
    out, rec = run_walk(tmp_path, monkeypatch, capsys, "--shape=cylinder", "--n=2")
    assert rec == [] and "lifted" in out  # no record without --carry (the benchmark is the carry)


def test_household_objects(tmp_path, monkeypatch, capsys):
    out, rec = run_walk(tmp_path, monkeypatch, capsys, "--carry", "--rand_layout", "--objects=gso", "--n=2", "--split=test")
    rows = rec[0]["rows"]
    assert rec[0]["objects"] == "gso" and len(rows) == 22 and all(r["name"] and r["shape"] == "household" for r in rows)
    assert all(900000 <= r["seed"] < 1000000 for r in rows) and len({(r["name"], r["seed"]) for r in rows}) == 22


def test_rejects_grab_without_policy(tmp_path, monkeypatch, capsys):
    fake_isaac.install()
    monkeypatch.setattr(sys, "argv", ["walk_grasp.py", "--grab", "v8"])
    with pytest.raises(SystemExit) as e:
        runpy.run_path(str(REPO / "isaac" / "walk_grasp.py"), run_name="__main__")
    assert e.value.code != 0
