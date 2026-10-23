"""so101/planner.py: answer parsing, the answer checker and the oracle (CPU, no model)."""
import numpy as np

from so101 import planner
from so101.objects import layout, random_objects

OBJS = [dict(id="o1", name="red cube", where="table"), dict(id="o2", name="blue ball", where="table"),
        dict(id="o3", name="green bar", where="shelf")]


def test_parse_takes_last_json_after_thinking():
    text = '<think>example: {"moves": [], "cannot": true}</think> answer {"moves": [{"object": "o1", "place": "shelf"}]}'
    assert planner.parse(text) == {"moves": [{"object": "o1", "place": "shelf"}]}


def test_parse_skips_objects_without_moves_and_bad_json():
    assert planner.parse('{"note": 1}') is None
    assert planner.parse('{"moves": [') is None
    assert planner.parse("no json here") is None


def test_check_possible_request():
    goal = {"o1": "red_tray"}
    assert planner.check(OBJS, {"moves": [{"object": "o1", "place": "red_tray"}]}, goal)
    assert not planner.check(OBJS, {"moves": []}, goal)
    # moving another object too is wrong
    assert not planner.check(OBJS, {"moves": [{"object": "o1", "place": "red_tray"},
                                              {"object": "o2", "place": "shelf"}]}, goal)
    # unknown object or place
    assert not planner.check(OBJS, {"moves": [{"object": "o9", "place": "red_tray"}]}, goal)
    assert not planner.check(OBJS, {"moves": [{"object": "o1", "place": "floor"}]}, goal)
    assert not planner.check(OBJS, {"moves": [{"object": "o1", "place": "red_tray"}], "cannot": True}, goal)
    assert not planner.check(OBJS, None, goal)


def test_check_impossible_request():
    assert planner.check(OBJS, {"moves": [], "cannot": True}, None)
    assert not planner.check(OBJS, {"moves": [], "cannot": False}, None)
    assert not planner.check(OBJS, {"moves": [{"object": "o1", "place": "shelf"}], "cannot": True}, None)


def test_oracle_passes_its_own_check_on_generated_requests():
    rng = np.random.default_rng(0)
    seen = 0
    while seen < 3:
        specs = random_objects(rng, 4)
        try:
            poses = layout(rng, specs, tries=200, max_r=0.23)
        except RuntimeError:
            continue
        objs = planner.make_scene(specs, poses, start={0: "shelf"})
        for family, text, goal in planner.requests(objs, rng, scanned=False):
            assert planner.check(objs, planner.oracle(objs, goal), goal), (family, text)
        seen += 1
