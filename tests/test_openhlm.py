"""openhlm/: the OpenHLM-data contract, the data reader, the policies and the whole pipeline on generated data (CPU,
no network). The end-to-end test takes about a minute."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

pytest.importorskip("pyarrow")  # the CPU venv from setup.sh has it; the openhlm tools need it
from openhlm import data, fake_data, policy, spec  # noqa: E402

PY = sys.executable


# ---------------------------------------------------------------------------------------------------------- contract
def test_split_is_fixed_and_sized():
    eps = {s: list(range(30)) for s in spec.SETS}
    split = spec.split_episodes(eps)
    assert sum(sum(1 for r in roles.values() if r == "test") for roles in split.values()) == spec.SAMPLE
    assert all(sum(1 for r in roles.values() if r == "test") == 4 for roles in split.values())
    assert split == spec.split_episodes(eps)  # deterministic
    assert spec.split_episodes(eps) == spec.split_episodes({s: list(reversed(range(30))) for s in spec.SETS})  # order-free
    assert all(sum(1 for r in roles.values() if r == "val") >= spec.MIN_VAL for roles in split.values())
    # fewer sets: the sample stays 20, the first sets take the remainder
    assert spec.test_per_set(3) == [7, 7, 6] and spec.test_per_set(1) == [20]
    three = spec.split_episodes({s: list(range(30)) for s in spec.SETS[:3]})
    assert [sum(1 for r in roles.values() if r == "test") for roles in three.values()] == [7, 7, 6]
    # canonical shares on one set alone agree with the five-set split
    one = spec.split_episodes({spec.SETS[1]: list(range(30))}, canonical=True)
    assert one[spec.SETS[1]] == split[spec.SETS[1]]
    with pytest.raises(ValueError):
        spec.split_episodes({spec.SETS[0]: list(range(5))})


def test_success_rule_and_stats():
    assert spec.skill(0.25, 0.5) == 0.5 and spec.success(0.5) and not spec.success(0.4999)
    assert spec.skill(1.0, 0.0) == 0.0 and spec.skill(0.0, 1.0) == 1.0
    assert spec.horizon(30) == 30 and spec.history(30) == 10 and spec.horizon(50) == 50 and spec.horizon(1000) == spec.MAX_HORIZON
    assert spec.mcnemar(0, 0) == 1.0 and spec.mcnemar(16, 0) == pytest.approx(2 / 2 ** 16)
    lo, hi = spec.wilson(10, 20)
    assert 0.29 < lo < 0.31 and 0.69 < hi < 0.71
    r = spec.record(spec.SETS[0], 3, "test", "mlp", 0.2, 0.5, 4, 120)
    assert r["success"] and r["skill"] == 0.6 and r["set"] == spec.SETS[0] and r["episode"] == 3
    with pytest.raises(ValueError):
        spec.record("not-a-set", 0, "test", "mlp", 0.1, 0.2, 1, 10)
    s = spec.summary([r, spec.record(spec.SETS[1], 0, "test", "mlp", 0.4, 0.5, 2, 60)])
    assert s["episodes"] == 2 and s["successes"] == 1 and s["at_level"]["0.25"] == 1 and set(s["by_set"]) == {spec.SETS[0], spec.SETS[1]}


# -------------------------------------------------------------------------------------------------------------- data
@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    root = tmp_path_factory.mktemp("openhlm")
    videos = True
    try:
        import av  # noqa: F401
    except ImportError:
        videos = False
    fake_data.make(root, sets=2, episodes=30, fps=10, dims=28, videos=True, seed=1, size=32, min_len=40, max_len=80)  # the real schema (PNG rows)
    fake_data.make_set(root, spec.SETS[2], 30, 10, 28, np.random.default_rng(2), videos=videos, size=32, min_len=40, max_len=80, schema="lerobot")
    return root


def test_reader_and_hold(fake):
    sd = data.SetData(spec.SETS[2], fake / spec.SETS[2])  # the LeRobot schema: named joints, observation.state / action
    assert sd.state_key == "observation.state" and sd.action_key == "action"
    assert sd.fps == 10 and sd.history == 3 and sd.horizon == 10 and sd.state_dim == sd.action_dim == 28
    assert sd.state_names and sd.state_names[0] == "left_shoulder_pitch" and (sd.hold_map() == np.arange(28)).all() and sd.hold_source == "names"
    assert len(sd.episodes) == 30 and all(len(v["state"]) >= 40 for v in sd.episodes.values())
    starts = sd.chunk_starts(0)
    assert starts[0] == sd.history - 1 and all(b - a == sd.horizon for a, b in zip(starts, starts[1:]))
    assert starts[-1] + sd.horizon <= len(sd.episodes[0]["state"])
    chunk = data.hold_chunk(sd, 0, starts[1], sd.hold_map())
    assert chunk.shape == (10, 28) and (chunk == sd.episodes[0]["state"][starts[1]]).all()
    std = np.ones(28, np.float32)
    assert data.nmae(np.zeros((10, 28)), np.ones((10, 28)), std) == pytest.approx(1.0)
    std[0] = 0.001  # a dimension that does not move (under 1 % of the largest std) is left out
    assert spec.scored_dims(std) == [False] + [True] * 27
    assert data.nmae(np.zeros((10, 28)), np.full((10, 28), 2.0), std) == pytest.approx(2.0)
    # a subset of episodes, and the split from meta without reading tables
    part = data.SetData(spec.SETS[2], fake / spec.SETS[2], episodes=[1, 2])
    assert set(part.episodes) == {1, 2}
    split = data.split_sets(str(fake), [spec.SETS[0], spec.SETS[1]])
    assert sum(sum(1 for r in roles.values() if r == "test") for roles in split.values()) == 20


def test_openhlm_schema_keys_and_inferred_hold_map(fake):
    """The real dataset's columns: `state` (31 here) and `actions` (30), no joint names, cameras by their own keys."""
    sd = data.SetData(spec.SETS[0], fake / spec.SETS[0])
    assert sd.state_key == "state" and sd.action_key == "actions" and sd.state_names is None
    assert sd.state_dim == 31 and sd.action_dim == 30
    hmap = sd.hold_map(sorted(sd.episodes)[:20])
    assert sd.hold_source == "inferred" and hmap.tolist() == list(range(28)) + [-1, -1]
    assert data.cameras_of(sd.info) == ["head_image_left", "left_wrist_image"] and data.camera_kind(sd.info, "head_image_left") == "image"
    assert not (fake / spec.SETS[0] / "videos").exists()  # the frames are PNG bytes in the parquet rows, as in the real data
    info = dict(sd.info)
    with pytest.raises(KeyError):
        data.resolve_keys(info, state_key="nope")
    assert data.resolve_keys({"features": {"robot_state": {"dtype": "float32"}, "actions": {"dtype": "float32"}}}) == ("robot_state", "actions")
    assert data.resolve_keys({"features": {"my_state_vec": {"dtype": "float32"}, "my_action_vec": {"dtype": "float32"}, "img": {"dtype": "video"}}}) == ("my_state_vec", "my_action_vec")
    # the inference on its own: a permuted, lagging copy plus an unrelated dimension
    rng = np.random.default_rng(0)
    A = np.cumsum(rng.normal(0, 0.05, (2000, 4)), 0).astype(np.float32)
    S = np.concatenate([A[:, [2, 0, 3, 1]], rng.normal(0, 1, (2000, 1))], 1).astype(np.float32)
    S[1:, :4] = S[:-1, :4]  # one frame of lag
    assert data.infer_hold_map(S, np.concatenate([A, rng.normal(0, 1, (2000, 1))], 1)).tolist() == [1, 3, 0, 2, -1]


def test_names_flat_nested_and_missing():
    info = {"features": {"a": {"shape": [2], "names": [["x", "y"]]}, "b": {"shape": [2], "names": ["x", "y"]},
                         "c": {"shape": [2], "names": None}, "d": {"shape": [3], "names": ["x", "y"]},
                         "e": {"shape": [2], "names": {"motors": ["x", "y"]}}}}
    assert data.feature_names(info, "a") == ["x", "y"] == data.feature_names(info, "b") == data.feature_names(info, "e")
    assert data.feature_names(info, "c") is None and data.feature_names(info, "d") is None and data.feature_names(info, "zz") is None


def test_hold_map_by_name():
    class Fake:
        state_names = ["a", "b", "c"]
        action_names = ["c", "a", "v"]
        state_dim, action_dim, fps = 3, 3, 30
        episodes = {0: dict(state=np.zeros((5, 3), np.float32), action=np.zeros((5, 3), np.float32))}
    assert data.SetData.hold_map(Fake()).tolist() == [2, 0, -1]
    Fake.action_names = ["x", "y", "z"]  # names that match nothing: inferred from the data (all constant here: none)
    f = Fake()
    assert data.SetData.hold_map(f).tolist() == [-1, -1, -1] and f.hold_source == "inferred"


def test_unaligned_state_and_action(tmp_path):
    """State-only and action-only dimensions: the hold map, the reference and the policies still work."""
    rng = np.random.default_rng(3)
    fake_data.make_set(tmp_path, spec.SETS[2], 12, 10, 28, rng, min_len=40, max_len=60, extra_state=2, extra_action=1, schema="lerobot")
    sd = data.SetData(spec.SETS[2], tmp_path / spec.SETS[2])
    assert sd.state_dim == 30 and sd.action_dim == 29
    hmap = sd.hold_map()
    assert hmap.tolist() == list(range(28)) + [-1]
    eps = sorted(sd.episodes)
    stats = sd.stats(eps[:9])
    std = np.maximum(stats["action_std"], spec.STD_FLOOR)
    t = sd.chunk_starts(eps[9])[1]
    chunk = data.hold_chunk(sd, eps[9], t, hmap)
    assert (chunk[:, :28] == sd.episodes[eps[9]]["state"][t, :28]).all() and (chunk[:, 28] == sd.episodes[eps[9]]["action"][t - 1, 28]).all()
    w = policy.Windows(sd, eps[:9], stats, hmap, "cpu")
    hist, base, act, _, ctx = w.batch(w.starts[:5])
    assert hist.shape == (5, 3, 30) and base.shape == (5, 29) and act.shape == (5, 10, 29)
    assert torch.allclose(base[:, 28], w.A[w.starts[:5] - 1, 28])  # an unmatched dimension: the last command is the base
    ridge = policy.Ridge(stats, hmap, sd.history, sd.horizon).fit(w)
    assert len(policy.score(sd, ridge, eps[9:], std, hmap, "val", "ridge")) == 3
    model = policy.ChunkMLP(sd.history, 30, sd.horizon, 29, hidden=32, layers=1, ctx_dim=sd.n_tasks)
    assert model(hist, base, ctx=ctx).shape == (5, 10, 29) and ctx.shape == (5, sd.n_tasks)


# ---------------------------------------------------------------------------------------------------------- policies
def test_policies_on_fake(fake):
    sd = data.SetData(spec.SETS[0], fake / spec.SETS[0])
    eps = sorted(sd.episodes)
    train, val = eps[:24], eps[24:]
    stats, hmap = sd.stats(train), sd.hold_map(train)
    std = np.maximum(stats["action_std"], spec.STD_FLOOR)
    hold = policy.score(sd, policy.Hold(hmap), val, std, hmap, "val", "hold")
    assert len(hold) == len(val) and all(r["skill"] == 0.0 and not r["success"] for r in hold)
    w = policy.Windows(sd, train, stats, hmap, "cpu")
    ridge = policy.Ridge(stats, hmap, sd.history, sd.horizon).fit(w)
    rr = policy.score(sd, ridge, val, std, hmap, "val", "ridge")
    assert np.mean([r["skill"] for r in rr]) > 0.1  # the fake motions are predictable: ridge beats hold
    model = policy.ChunkMLP(sd.history, sd.state_dim, sd.horizon, sd.action_dim, hidden=32, layers=1, ctx_dim=sd.n_tasks)
    zero = policy.score(sd, policy.Learned(model, stats, hmap), val, std, hmap, "val", "mlp")
    assert all(abs(r["skill"]) < 1e-5 for r in zero)  # zero-initialised head: exactly the hold baseline
    # checkpoints round-trip
    p = policy.save(fake / "ridge.pt", "ridge", sd, stats, hmap, {"kind": "ridge", "alpha": 1.0}, W=ridge.W)
    back, payload = policy.load(p)
    assert payload["kind"] == "ridge" and np.allclose(back.predict(sd, val[0], sd.chunk_starts(val[0])), ridge.predict(sd, val[0], sd.chunk_starts(val[0])))
    p = policy.save(fake / "mlp.pt", "mlp", sd, stats, hmap, dict(policy.CANDIDATES["mlp"], hidden=32, layers=1), state_dict=model.state_dict())
    back, payload = policy.load(p)
    assert payload["kind"] == "mlp" and back.model.H == sd.history


def test_vision_windows_and_model(fake, tmp_path):
    r = subprocess.run([PY, "-m", "openhlm.frames", "--camera", "head_image_left", "--size", "32", "--data", str(fake), "--sets", spec.SETS[0],
                        "--out", str(tmp_path / "frames"), "--workers", "2"], capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode == 0 and "FRAMES-DONE" in r.stdout, r.stdout + r.stderr  # frames from the parquet rows
    try:
        import av  # noqa: F401
        r = subprocess.run([PY, "-m", "openhlm.frames", "--camera", "observation.images.cam_left_high", "--size", "32", "--data", str(fake),
                            "--sets", spec.SETS[2], "--out", str(tmp_path / "frames"), "--workers", "2"], capture_output=True, text=True, cwd=str(REPO))
        assert r.returncode == 0 and "errors=0" in r.stdout, r.stdout + r.stderr  # frames from mp4 files
        v = np.load(tmp_path / "frames" / spec.SETS[2] / "observation.images.cam_left_high" / "episode_000000.npy")
        assert v.shape[1:] == (32, 32, 3) and v.max() > 100
    except ImportError:
        pass
    sd = data.SetData(spec.SETS[0], fake / spec.SETS[0], cameras=["head_image_left"], frames_dir=tmp_path / "frames")
    e = sorted(sd.episodes)[0]
    L = len(sd.episodes[e]["state"])
    assert sd.frame_stride == 3 and sd.frames[e].shape == (-(-L // 3), 1, 32, 32, 3)
    stats, hmap = sd.stats(sorted(sd.episodes)[:20]), sd.hold_map(sorted(sd.episodes)[:20])
    w = policy.Windows(sd, sorted(sd.episodes)[:20], stats, hmap, "cpu", frames=True)
    hist, hold, act, fr, _ = w.batch(w.starts[:4])
    assert fr.shape == (4, 1, 3, 32, 32) and fr.max() <= 1.0
    t = int(w.starts[3]); e0 = sorted(sd.episodes)[0]  # the first episode's rows: frame t//3 of its cache
    if t < len(sd.episodes[e0]["state"]):
        assert np.array_equal((fr[3, 0].permute(1, 2, 0) * 255).round().numpy().astype(np.uint8), sd.frames[e0][t // 3, 0])
    model = policy.ChunkMLP(sd.history, sd.state_dim, sd.horizon, sd.action_dim, hidden=32, layers=1, cameras=1, ctx_dim=sd.n_tasks)
    assert model(hist, hold, fr).shape == (4, sd.horizon, sd.action_dim)  # no ctx given: task 0 assumed
    # two cameras from the parquet rows
    r = subprocess.run([PY, "-m", "openhlm.frames", "--camera", "left_wrist_image", "--size", "32", "--data", str(fake), "--sets", spec.SETS[0],
                        "--out", str(tmp_path / "frames"), "--workers", "2"], capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode == 0, r.stdout + r.stderr
    sd2 = data.SetData(spec.SETS[0], fake / spec.SETS[0], cameras=["head_image_left", "left_wrist_image"], frames_dir=tmp_path / "frames")
    w2 = policy.Windows(sd2, sorted(sd2.episodes)[:5], stats, hmap, "cpu", frames=True)
    fr2 = w2.batch(w2.starts[:3])[3]
    assert fr2.shape == (3, 2, 3, 32, 32)
    m2 = policy.ChunkMLP(sd.history, sd.state_dim, sd.horizon, sd.action_dim, hidden=32, layers=1, cameras=2, ctx_dim=sd.n_tasks)
    d = {}
    recs = policy.score(sd2, policy.Learned(m2, stats, hmap), sorted(sd2.episodes)[:3], np.maximum(stats["action_std"], spec.STD_FLOOR), hmap, "val", "m2", diagnostics=d)
    assert len(recs) == 3 and d["step"].shape == (sd.horizon,) and d["dim"].shape == (sd.action_dim,) and d["chunks"] == sum(r["chunks"] for r in recs)
    assert np.allclose(d["step"], d["step_hold"])  # zero-initialised head: identical to hold


# --------------------------------------------------------------------------------------------------------- pipeline
def test_pipeline_end_to_end(fake, tmp_path):
    out, logs = tmp_path / "runs", tmp_path / "logs"
    cmd = [PY, "-m", "openhlm.pipeline", "--tag", "t", "--data", str(fake), "--gpus", "cpu", "--budget-min", "0.75", "--reserve-min", "0.15",
           "--startup-s", "3", "--min-unit-s", "6", "--eval-every", "3", "--out", str(out), "--logs", str(logs), "--test", "--candidates", "mlp,mlp-wide,ridge",
           "--cache", str(out / "cache")]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode == 0 and "PIPELINE-DONE" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]
    split = json.load(open(out / "t" / "split.json"))
    assert set(split) == set(spec.SETS[:3]) and sum(sum(1 for v in roles.values() if v == "test") for roles in split.values()) == 20
    sel = json.load(open(out / "t" / "selection.json"))
    assert set(sel) == set(spec.SETS[:3]) and all(v["candidate"] in ("mlp", "mlp-wide", "ridge", "ens") and len(v["sha256"]) == 64 for v in sel.values())
    recs = spec.read_jsonl(logs / "test-t.jsonl")
    assert len(recs) == 20 and len({(r["set"], r["episode"]) for r in recs}) == 20
    assert all(r["split"] == "test" and split[r["set"]][str(r["episode"])] == "test" and isinstance(r["success"], bool) for r in recs)
    assert all(r["checkpoint"] == sel[r["set"]]["sha256"] for r in recs)
    assert (out / "t" / "report.md").exists() and (out / "t" / "test-refs.jsonl").exists()
    refs = spec.read_jsonl(out / "t" / "test-refs.jsonl")
    assert {r["policy"] for r in refs} >= {"hold"} and all(r["skill"] == 0.0 for r in refs if r["policy"] == "hold")
    assert "check_openhlm.py rc=0" in r.stdout or "check_openhlm.py rc=1" in r.stdout  # the check ran (the rate on fake data is not asserted)
    # the check's own structural rules pass on our records
    c = subprocess.run([PY, str(REPO / "scripts" / "check_openhlm.py"), "--offline", str(logs)], capture_output=True, text=True, cwd=str(REPO))
    assert "[pass] sample size: 20 distinct episodes" in c.stdout and "[pass] each episode scored once" in c.stdout, c.stdout
    hist = spec.read_jsonl(out / "test-history.jsonl")
    assert len(hist) == 1 and hist[0]["number"] == 1 and hist[0]["successes"] == sum(r["success"] for r in recs)
    # a second test run is refused with --refuse-retest; otherwise it is counted (#2), the old record moved aside
    r2 = subprocess.run(cmd + ["--refuse-retest"], capture_output=True, text=True, cwd=str(REPO))
    assert r2.returncode != 0 and "earlier test run" in (r2.stdout + r2.stderr)
    r3 = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    assert r3.returncode == 0 and "TEST RUN #2" in r3.stdout and len(list((out / "old").glob("*/test-t.jsonl"))) == 1 and (logs / "test-t.jsonl").exists(), r3.stdout[-2000:]
    hist = spec.read_jsonl(out / "test-history.jsonl")
    assert len(hist) == 2 and hist[1]["number"] == 2
    assert "run #2" in (out / "t" / "report.md").read_text() and "Earlier test runs" in (out / "t" / "report.md").read_text()
    # the ensemble was scored on validation and could be selected; the cache was built
    assert all((out / "t" / s / "ens.json").exists() for s in spec.SETS[:3]) and list((out / "cache").glob("*.npz"))
    assert all((out / "t" / s / "diagnostics.json").exists() for s in spec.SETS[:3]) and "Where the error is" in (out / "t" / "report.md").read_text()


def test_check_script_rules(tmp_path):
    """scripts/check_openhlm.py: the rules our records must meet."""
    recs = [spec.record(spec.SETS[i % 5], i, "test", "mlp", 0.1 if i < 12 else 0.4, 0.5, 3, 100) for i in range(20)]
    spec.write_jsonl(tmp_path / "test-x.jsonl", recs)
    c = subprocess.run([PY, str(REPO / "scripts" / "check_openhlm.py"), "--offline", str(tmp_path)], capture_output=True, text=True, cwd=str(REPO))
    assert c.returncode == 0 and "success 12/20 = 60%" in c.stdout, c.stdout
    spec.write_jsonl(tmp_path / "test-y.jsonl", recs[:1])  # a repeated episode fails the check
    c = subprocess.run([PY, str(REPO / "scripts" / "check_openhlm.py"), "--offline", str(tmp_path)], capture_output=True, text=True, cwd=str(REPO))
    assert c.returncode == 1 and "[FAIL] each episode scored once" in c.stdout


def test_act_candidate_from_lerobot_stack(fake, tmp_path):
    """ACT through openhlm/act_policy.py: only where lerobot is installed (.venv-vla); a few CPU steps on the fake frames."""
    pytest.importorskip("lerobot")
    pytest.importorskip("av")
    r = subprocess.run([PY, "-m", "openhlm.frames", "--camera", "head_image_left", "--size", "32", "--data", str(fake), "--sets", spec.SETS[0],
                        "--out", str(tmp_path / "frames"), "--workers", "2"], capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode == 0, r.stdout + r.stderr
    r = subprocess.run([PY, "-m", "openhlm.train", "--set", spec.SETS[0], "--candidate", "act", "--out", str(tmp_path / "run"), "--data", str(fake),
                        "--device", "cpu", "--budget-s", "25", "--eval-every", "5", "--camera", "head_image_left", "--frames", str(tmp_path / "frames")],
                       capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode == 0 and "TRAIN-DONE" in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]
    pol, payload = policy.load(tmp_path / "run" / spec.SETS[0] / "act.pt")
    assert payload["kind"] == "act" and payload["cameras"] == ["head_image_left"] and payload["state_key"] == "state"
    sd = data.SetData(spec.SETS[0], fake / spec.SETS[0], cameras=["head_image_left"], frames_dir=tmp_path / "frames")
    e = sorted(sd.episodes)[0]
    assert pol.predict(sd, e, sd.chunk_starts(e)).shape == (len(sd.chunk_starts(e)), sd.horizon, sd.action_dim)


def test_spec_agrees_with_check_script():
    import importlib.util
    s = importlib.util.spec_from_file_location("check_openhlm", REPO / "scripts" / "check_openhlm.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    assert list(spec.SETS) == m.SETS and m.REPO.startswith("https://huggingface.co/datasets/" + spec.REPO_ID)
    assert spec.SAMPLE == 20 and spec.PASS_RATE == 0.5
