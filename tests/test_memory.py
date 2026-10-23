"""so101/memory.py: scene memory tracking on hand-made detections (CPU, no perception models)."""
import itertools

import numpy as np

import so101.memory as memory
from so101.memory import SceneMemory


def det(x, y, emb=None, where="table"):
    return dict(centre=[x, y, 0.02], dims=[3, 3, 3], where=where, emb=None if emb is None else np.asarray(emb, float))


def test_assignment_matches_brute_force():
    rng = np.random.default_rng(1)
    for _ in range(50):
        n, m = (int(v) for v in rng.integers(1, 5, 2))
        C = rng.uniform(0, 10, (n, m))
        r, c = memory.linear_sum_assignment(C)
        if n <= m:
            best = min(sum(C[i, j] for i, j in zip(range(n), p)) for p in itertools.permutations(range(m), n))
        else:
            best = min(sum(C[i, j] for i, j in zip(p, range(m))) for p in itertools.permutations(range(n), m))
        assert len(r) == min(n, m) and abs(C[r, c].sum() - best) < 1e-9


def test_ids_stay_put_when_objects_do_not_move():
    mem = SceneMemory(method="position")
    a, _ = mem.update([det(0.10, 0.00), det(0.20, 0.05)])
    b, events = mem.update([det(0.201, 0.05), det(0.101, 0.0)])  # same objects, other order, 1 mm off
    assert b == {0: a[1], 1: a[0]}
    assert events == []


def test_new_ids_are_never_reused():
    mem = SceneMemory(method="position")
    mem.update([det(0.1, 0.0)])
    mem.update([])
    mem.update([])  # p1 now missing
    a, events = mem.update([det(0.3, 0.1)])
    assert a == {0: "p2"} and ("new", "p2") in events


def test_missing_after_miss_looks_only_when_in_view():
    mem = SceneMemory(method="position")
    mem.update([det(0.1, 0.0)])
    for _ in range(memory.MISS_LOOKS + 1):
        mem.update([], in_view=lambda xy: False)  # behind the arm: never counts
    assert mem.tracks["p1"].status == "seen"
    for k in range(memory.MISS_LOOKS):
        _, events = mem.update([])
    assert mem.tracks["p1"].status == "missing" and ("missing", "p1") in events


def test_placed_object_predicted_at_its_place():
    mem = SceneMemory(method="position", pos_scale=0.05, gate=3.0)
    mem.update([det(0.10, 0.00)])
    mem.picked("p1")
    assert mem.tracks["p1"].status == "held"
    mem.placed("p1", (0.14, 0.19), "red_tray")
    a, events = mem.update([det(0.14, 0.19, where="red_tray")])  # 20 cm away from where it was seen
    assert a == {0: "p1"} and mem.tracks["p1"].where == "red_tray"
    assert not any(e == "drop?" for e, _ in events)


def test_drop_flagged_when_placed_object_seen_elsewhere():
    mem = SceneMemory(method="both", pos_scale=0.05, app_scale=0.1, gate=3.0)
    mem.update([det(0.10, 0.00, emb=[1, 0, 0])])
    mem.placed("p1", (0.14, 0.19), "red_tray")
    a, events = mem.update([det(0.14, 0.12, emb=[1, 0, 0])])  # 7 cm short of the tray
    assert a == {0: "p1"} and ("drop?", "p1") in events


def test_relink_by_appearance_after_unexplained_move():
    mem = SceneMemory(method="both", pos_scale=0.03, app_scale=0.1, gate=3.0, relink_gate=2.0)
    mem.update([det(0.10, 0.00, emb=[1, 0, 0]), det(0.20, 0.10, emb=[0, 1, 0])])
    a, events = mem.update([det(0.10, 0.00, emb=[1, 0, 0]), det(0.25, -0.15, emb=[0, 1, 0])])  # p2 knocked 25 cm
    assert a == {0: "p1", 1: "p2"} and ("relinked", "p2") in events


def test_position_method_does_not_relink():
    mem = SceneMemory(method="position", pos_scale=0.03, gate=3.0)
    mem.update([det(0.10, 0.00, emb=[1, 0, 0]), det(0.20, 0.10, emb=[0, 1, 0])])
    a, _ = mem.update([det(0.10, 0.00, emb=[1, 0, 0]), det(0.25, -0.15, emb=[0, 1, 0])])
    assert a[0] == "p1" and a[1] == "p3"


def test_v1_nearest_within_4cm():
    mem = SceneMemory(method="v1")
    mem.update([det(0.10, 0.00)])
    a, _ = mem.update([det(0.13, 0.00)])
    assert a == {0: "p1"}
    a, _ = mem.update([det(0.20, 0.00)])
    assert a == {0: "p2"}


def test_goal_left_and_text():
    mem = SceneMemory(method="position")
    mem.update([det(0.10, 0.00), det(0.20, 0.05)])
    mem.tracks["p1"].name = "red cube"
    mem.placed("p1", (0.32, 0.0), "shelf")
    assert mem.goal_left({"p1": "shelf", "p2": "red_tray"}) == ["p2"]
    text = mem.text()
    assert text.splitlines()[0].startswith("p1 red cube: shelf (32, 0 cm); placed shelf")
    assert "p2 unnamed: table" in text
