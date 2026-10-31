"""isaac/obs_compare.py on synthetic probes: a planted frame offset must be named as the first cause."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "isaac"))
import obs_compare as C  # noqa: E402

TERMS = [["policy", "command", 3], ["proprio", "joint_pos", 16], ["perception", "object_point_cloud", 12]]


def probe(base, rng, n=20, T=30, obj_shift=0.0, pitch_deg=0.0, act_noise=0.0, track=0.0, seeds_from=700000, kick_deg=0.0):
    dim = sum(t[2] for t in TERMS)
    obs = rng.normal(0, 1, (T, n, dim)).astype(np.float32)
    obs[:, :, 19:31] = obs[:, :, 19:31] * 0.02 + 0.2 + obj_shift  # the point cloud: points spread ~2 cm across layouts; a frame offset moves every point
    obj_b = np.tile([0.2, 0.0, 0.03], (T, n, 1)).astype(np.float32) + obj_shift
    q = rng.normal(0, 0.3, (T, n, 16)).astype(np.float32)
    q_tgt = q + rng.normal(0, 0.01 + track, (T, n, 16)).astype(np.float32)
    pitch = np.radians(pitch_deg) + np.radians(kick_deg) * np.minimum(np.arange(T) / 5, 1)[:, None]  # a kick: pitch grows over 5 steps
    half = np.broadcast_to(pitch[:, :, None], (T, n, 1)) / 2 if np.ndim(pitch) else pitch / 2
    root = np.concatenate([np.tile([0, 0, 0.02], (T, n, 1)), np.cos(half), np.zeros((T, n, 1)), np.sin(half), np.zeros((T, n, 1))], -1).astype(np.float32)
    act = rng.normal(0, 1, (T, n, 16)).astype(np.float32) + act_noise * rng.normal(0, 1, (T, n, 16)).astype(np.float32)
    near = np.linspace(0.3, 0.02 + obj_shift * 3, T)[:, None].repeat(n, 1).astype(np.float32)
    return dict(base=np.array(base), checkpoint=np.array("v6b.pt"), dt=np.array(0.02), joints=np.array([f"j{i}_joint" for i in range(16)]),
                terms=TERMS, groups=["policy", "proprio", "perception"], root_nominal=np.array([-0.08, 0, 0.02], np.float32),
                seeds=np.arange(seeds_from, seeds_from + n), shapes=np.array(["box"] * n), success=near[-1] < 0.05, max_rise=np.zeros(n),
                why=np.array(["kept" if x else "not lifted" for x in near[-1] < 0.05]), obs=obs, act=act, q=q, q_tgt=q_tgt,
                hand=np.zeros((T, n, 3), np.float32), obj_b=obj_b, root=root, near=near, cmd=np.zeros((T, n, 3), np.float32))


def test_planted_frame_offset_is_named(capsys):
    a = probe("fixed", np.random.default_rng(1))
    b = probe("walking", np.random.default_rng(1), obj_shift=0.07, pitch_deg=4.0)  # same draws, the object 7 cm off
    lines = []
    v = C.compare(a, b, steps=30, out=lines.append)
    text = "\n".join(lines)
    assert v["obs_worst"][0] == "perception/object_point_cloud" and v["obs_worst"][1] > 3
    assert "OBSERVATION" in text and abs(v["frame_diff_m"] - 0.07) < 1e-3 and abs(v["pitch_deg"] - 4.0) < 0.01


def test_agreeing_probes_blame_later_phase():
    a = probe("fixed", np.random.default_rng(2))
    b = probe("walking", np.random.default_rng(2))
    lines = []
    v = C.compare(a, b, steps=30, out=lines.append)
    assert v["obs_worst"][1] < 1e-3 and "gap opens later" in "\n".join(lines)


def test_tracking_difference_is_named():
    a = probe("fixed", np.random.default_rng(3))
    b = probe("walking", np.random.default_rng(3), track=0.2)
    lines = []
    v = C.compare(a, b, steps=30, out=lines.append)
    assert v["track_ratio"] > 2 and "TRACKING" in "\n".join(lines)


def test_align_by_seed_and_shape():
    a = probe("fixed", np.random.default_rng(4), n=10)
    b = probe("walking", np.random.default_rng(4), n=10, seeds_from=700005)
    ia, ib = C.align(a, b)
    assert list(ia) == [5, 6, 7, 8, 9] and list(ib) == [0, 1, 2, 3, 4]


def test_base_kick_is_named():
    a = probe("fixed", np.random.default_rng(5))
    b = probe("walking", np.random.default_rng(5), kick_deg=20.0)
    lines = []
    v = C.compare(a, b, steps=30, out=lines.append)
    assert v["base_kick_deg"] > 15 and "BASE DYNAMICS" in "\n".join(lines)
