"""isaac/bench_spec.py: the walking-G1 benchmark contract (seed splits, success tests, result records), CPU only."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("bench_spec", Path(__file__).resolve().parent.parent / "isaac" / "bench_spec.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)


def test_splits_disjoint_and_bases_inside():
    ranges = sorted(B.SPLITS.values())
    assert all(a < b for a, b in ranges) and all(ranges[i][1] <= ranges[i + 1][0] for i in range(len(ranges) - 1))
    for bench, bases in B.BASE.items():
        for split, base in bases.items():
            assert B.split_of_seed(base) == split and B.split_of_seed(base + 999) == split, (bench, split)
    assert B.split_of_seed(1000000) is None and B.split_of_seed(699999) is None


def test_seeds_per_shape():
    s = B.seeds("walk_carry", "val", 16, B.SHAPES)
    assert len(s) == 96 and len(set(s)) == 96 and all(B.split_of_seed(k) == "val" for _, k in s)
    assert B.seeds("box_carry", "test", 3) == [965000, 965001, 965002]
    with pytest.raises(ValueError):
        B.seeds("box_carry", "dev", 3)


def test_success_rules():
    assert B.lifted(0.10, 5.0) and not B.lifted(0.064, 90.0) and not B.lifted(0.10, 70.0) and not B.lifted(0.05, 0.0)
    assert B.upright(5.0, "standing") and not B.upright(45.0, "standing")
    assert B.upright(5.0, "lying") and not B.upright(60.0, "lying")
    assert B.upright(170.0, "ball")
    with pytest.raises(ValueError):
        B.upright(0.0, "cube")
    assert not B.released(0.0, True) and not B.released(0.10, True) and not B.released(0.005, False) and B.released(0.08, False)
    assert B.kind_of("ball") == "ball" and B.kind_of("egg") == "lying" and B.kind_of("box") == "standing"


def test_tilt():
    assert B.tilt_deg([0, 0, 1], [0, 0, 1]) == pytest.approx(0.0)
    assert B.tilt_deg([1, 0, 0], [0, 0, 1]) == pytest.approx(90.0)
    assert B.tilt_deg([1, 0, 0], horizontal=True) == pytest.approx(0.0)  # a lying axis, flat
    assert B.tilt_deg([0, 0, 1], horizontal=True) == pytest.approx(90.0)  # stood up


def test_record_checks_rows(capsys):
    rows = [dict(seed=760000 + i, shape="box", success=i == 0, fell=False, upright=True, released=True) for i in range(3)]
    rec = B.record("walk_carry", "val", "scripted", None, "random", "primitives", dict(n=3, vx=0.4, stand=0.18), rows)
    B.print_record(rec)
    line = capsys.readouterr().out.strip()
    assert line.startswith("BENCH_RESULT ") and json.loads(line[len("BENCH_RESULT "):])["rows"][0]["success"] is True
    bad = [dict(rows[0], upright=False)]
    with pytest.raises(AssertionError):
        B.record("walk_carry", "val", "scripted", None, "random", "primitives", {}, bad)
    with pytest.raises(AssertionError):  # a test seed in a val record
        B.record("walk_carry", "val", "scripted", None, "random", "primitives", {}, [dict(rows[0], seed=960000)])
    with pytest.raises(AssertionError):  # numpy bools are not bools: the scripts cast before recording
        B.record("walk_carry", "val", "scripted", None, "random", "primitives", {}, [dict(rows[0], success=1)])


def test_sha256(tmp_path):
    f = tmp_path / "w.pt"
    f.write_bytes(b"abc")
    assert B.sha256(str(f)) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert B.sha256("") is None
