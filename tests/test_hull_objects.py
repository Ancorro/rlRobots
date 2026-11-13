"""isaac/hull_objects.py: the scanned-object rules the benchmark (walk_grasp.py) and the trainer (grab_rl.py) share."""
import sys
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "isaac"))
import hull_objects as H  # noqa: E402


def test_household_split_is_by_name_hash_and_disjoint():
    names = {}
    for split, want in (("val", 1), ("test", 0), ("train", None)):
        es, hull_file = H.entries("gso", split)
        assert es == sorted(es, key=lambda e: e["name"]) and hull_file.name == "hulls.npz"
        for e in es:
            h = zlib.crc32(f"gso/{e['name']}".encode()) % 10
            assert e["kept"] and (h == want if want is not None else h >= 2)
        names[split] = {e["name"] for e in es}
    assert names["val"] and names["test"] and names["train"]
    assert not (names["train"] & names["val"]) and not (names["train"] & names["test"]) and not (names["val"] & names["test"])


def test_layouts_follow_the_benchmark_seed_rule():
    es, _ = H.entries("gso", "val")
    ls = H.layouts(es, 3, 760000, "household")
    assert [L["seed"] for L in ls] == [760000 + 100 * j + i for j in range(len(es)) for i in range(3)]
    assert [L["spec"]["name"] for L in ls[::3]] == [e["name"] for e in es]
    again = H.layouts(es, 3, 760000, "household")
    assert [(L["d"], L["y"], L["heading"], L["spec"]["friction"]) for L in ls] == [(L["d"], L["y"], L["heading"], L["spec"]["friction"]) for L in again]
    one = H.layout_of(es[1], 760000 + 100 + 2, "household")
    assert (one["d"], one["y"], one["heading"]) == (ls[5]["d"], ls[5]["y"], ls[5]["heading"])
    for L in ls:
        s, e = L["spec"], L["spec"]["entry"]
        assert s["type"] == "hull" and 0.7 <= s["friction"] <= 1.5 and L["y"] <= 0.04
        # the pinch closes at mid-height, and no deeper than FINGER_DEPTH below the top
        top = e["bottom_z_m"] + e["height_m"]
        assert s["pinch_dz"] <= top and top - s["pinch_dz"] <= H.FINGER_DEPTH + 1e-9


def test_root_placement_puts_the_footprint_centre_on_the_spot():
    es, _ = H.entries("gso", "val")
    spec = H.spec_of(es[0], np.random.default_rng(0), "household")
    cx, cy = es[0]["centre_xy_m"]
    for a in (0.0, 0.7, -2.1, np.pi):
        x, y = H.root_xy(spec, 0.3, 1.2, a)
        ca, sa = np.cos(a), np.sin(a)
        assert np.allclose([x + ca * cx - sa * cy, y + sa * cx + ca * cy], [0.3, 1.2])
    assert H.root_xy(dict(type="box"), 0.3, 1.2, 0.7) == (0.3, 1.2)
    assert H.grasp_local(dict(type="box")) == [0.0, 0.0, 0.0] and H.centre_local(dict(type="box")) == [0.0, 0.0, 0.0]
    assert H.grasp_local(spec) == [cx, cy, spec["pinch_dz"]]
    assert H.centre_local(spec) == [cx, cy, es[0]["bottom_z_m"] + es[0]["height_m"] / 2]


def test_stand_in_classes():
    e = dict(width_m=0.04, length_m=0.04, height_m=0.04, fill=0.95)
    assert H.stand_in(e) == ("box", [0.02, 0.02, 0.02])
    assert H.stand_in(dict(e, fill=0.6))[0] == "ball"
    assert H.stand_in(dict(e, fill=0.6, height_m=0.10))[0] == "cylinder"
    assert H.stand_in(dict(e, fill=0.6, length_m=0.12))[0] == "lying cylinder"
    assert H.stand_in(dict(e, fill=0.6, length_m=0.12, height_m=0.10))[0] == "box"
