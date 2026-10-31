"""The walking-G1 benchmark contract (goal day 1, 2026-10-27; scripts/goal_common.py): seed splits, the success tests
every script shares, and the BENCH_RESULT record a run prints at its end. Standard library only, so the CPU checks
(scripts/check_day*.py) can import it without Isaac.

Benchmarks:
  walk_carry  isaac/walk_grasp.py --carry --rand_layout: walk to a table, pinch one primitive object, carry it to a second
              table, set it down. Success: lifted (9 cm up, not tipped), set down on table B, upright, released, robot
              standing. The goal's benchmark.
  box_carry   isaac/box_walk.py: two-hand squeeze of a box, walk to the end table, set it down. Success: the box at rest
              on the end table at its start height, upright, both hands clear, robot standing. Recorded the same way;
              not part of the goal checks.

Seeds. One layout = (shape, seed); the seed drives every random draw of that layout (object size, mass, friction, pose,
table placement). The splits are disjoint ranges; each benchmark draws its layouts from a base inside the split
(BASE[bench][split] + i), so the historic walk runs at 960000+ stay test layouts and nothing tuned on them is "held out".
The Isaac training scripts already use 800000+ (train) and 900000+ (test); g1_dexsuite.py's pool validation 710000+
sits in val.

Audit holes closed here (2026-10-23 audit):
  - an object knocked onto its side reads "up" (a box ~6 cm, box_walk.py walk6/7): `lifted` needs the rise AND the
    object still upright; `upright` is the shared tilt rule (standing shapes: axis within UPRIGHT_DEG of its start;
    lying shapes: long axis within UPRIGHT_DEG of horizontal; balls always)
  - a hand still gripping is not "released": `released` needs no contact and a clear gap
  - evaluations default to the validation split; test is scored once, on day 3
"""
import hashlib
import json
import math

LIFT_M = 0.09  # the lift tests' rule: at least 9 cm above the start (walk_grasp.py LIFT 0.10 - 0.01)
LIFTED_TILT_DEG = 45.0  # more tilt than this while "up" is a tipped or dangling object, not a lift
UPRIGHT_DEG = 20.0  # set down within this of its start attitude (walk_grasp.py's rule since 2026-10-22)
RELEASE_GAP_M = 0.04  # hand clear of the object by this much (walk_grasp.py `free` > 0.04)

SPLITS = {"train": (800000, 900000), "val": (700000, 800000), "test": (900000, 1000000)}
BASE = {"walk_carry": {"train": 860000, "val": 760000, "test": 960000},
        "box_carry": {"train": 865000, "val": 765000, "test": 965000}}
SHAPES = ("box", "cylinder", "lying cylinder", "ball", "capsule", "egg")  # walk_carry's 6 classes (so101/objects.py)


def split_of_seed(seed):
    """'train' | 'val' | 'test' for a layout seed, None outside every split."""
    for name, (a, b) in SPLITS.items():
        if a <= seed < b:
            return name
    return None


def seeds(bench, split, n, shapes=None):
    """Layout seeds of a benchmark on a split: n per shape (walk_carry) or n trials (box_carry), BASE + i."""
    if split not in SPLITS:
        raise ValueError(f"split {split!r}: one of {list(SPLITS)}")
    base = BASE[bench][split]
    if shapes is None:
        return [base + i for i in range(n)]
    return [(s, base + i) for s in shapes for i in range(n)]


def lifted(rise_m, tilt_deg):
    """Up from the start and still upright. A box tipped on its side reads about 6 cm up and 90 deg: not lifted."""
    return bool(rise_m >= LIFT_M and tilt_deg < LIFTED_TILT_DEG)


def upright(tilt_deg, kind):
    """kind 'standing': axis tilt from the start attitude; 'lying' (lying cylinder, capsule, egg): long axis from
    horizontal; 'ball': always upright."""
    if kind == "ball":
        return True
    if kind in ("standing", "lying"):
        return bool(tilt_deg < UPRIGHT_DEG)
    raise ValueError(f"kind {kind!r}: standing, lying or ball")


def kind_of(shape):
    """walk_carry shape class -> upright kind."""
    return "ball" if shape == "ball" else "lying" if shape in ("lying cylinder", "capsule", "egg") else "standing"


def released(hand_gap_m, contact):
    """The hand is clear of the object: no contact and a gap of at least RELEASE_GAP_M."""
    return bool(not contact and hand_gap_m >= RELEASE_GAP_M)


def tilt_deg(axis_now, axis_start=None, horizontal=False):
    """Degrees between a unit axis now and at the start; with horizontal=True, of the axis from the horizontal plane
    (lying shapes). Accepts lists or numpy arrays of 3."""
    x, y, z = (float(v) for v in axis_now)
    n = math.sqrt(x * x + y * y + z * z) or 1.0
    if horizontal:
        return math.degrees(math.asin(min(1.0, abs(z) / n)))
    a, b, c = (float(v) for v in axis_start)
    m = math.sqrt(a * a + b * b + c * c) or 1.0
    return math.degrees(math.acos(max(-1.0, min(1.0, (x * a + y * b + z * c) / (n * m)))))


def sha256(path):
    """Checksum of a checkpoint file (the record's 'checkpoint'), None without one."""
    if not path:
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def machine():
    """Where a run happened (settings["host"], settings["gpu"]): GPU PhysX is not deterministic across machines (val
    scripted 2026-10-26: cluster RTX 8000 40/96 vs a local RTX 3090 41/96, 9 layouts differ), so paired comparisons
    should use records from one machine."""
    import platform
    gpu = ""
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        pass
    return dict(host=platform.node(), gpu=gpu)


def record(bench, split, policy, checkpoint, layout, objects, settings, rows):
    """The result record (scripts/goal_common.py): checks the rows before they are written; adds the machine."""
    assert bench in BASE, bench
    assert split in SPLITS, split
    need = {"seed": int, "shape": str, "success": bool, "fell": bool, "upright": bool, "released": bool}
    for i, r in enumerate(rows):
        for k, t in need.items():
            assert isinstance(r.get(k), t), f"row {i}: {k} missing or not {t.__name__}: {r.get(k)!r}"
        assert split_of_seed(r["seed"]) == split, f"row {i}: seed {r['seed']} is not in the {split} split"
        assert not (r["success"] and (r["fell"] or not r["upright"] or not r["released"])), \
            f"row {i}: a success that fell, tipped or still grips"
    return dict(bench=bench, split=split, policy=policy, checkpoint=checkpoint, layout=layout, objects=objects,
                settings={k: v for k, v in settings.items()} | machine(), rows=rows)


def print_record(rec):
    """One line, so it lands in the job log that the checks read: BENCH_RESULT {...}."""
    print("BENCH_RESULT " + json.dumps(rec, default=lambda x: x.item() if hasattr(x, "item") else str(x)), flush=True)
