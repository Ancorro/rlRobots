"""Day 1 check: the walk_carry benchmark is fixed, and the scripted baseline is re-run on it.

Covers the 2026-10-23 audit holes that touch the benchmark (scripts/goal_common.py):
  1. one success test: isaac/bench_spec.py (lifted, upright, released, seed splits), imported by walk_grasp.py
  2. an object tipped on its side is not "lifted" (a box reads ~6 cm up); a hand still gripping is not "released"
  3. evaluations default to validation seeds, never test (--split default "val")
  4. random layouts: sideways walking command, and table B the same width as in the fixed layout (was 1.2 vs 1.6 m)
  5. cluster batch scripts pass the run's exit code on (walk_cluster.sbatch ended 0 on a crash; bench.sbatch `|| true`)
  6. every run prints a BENCH_RESULT record with its settings and per-layout rows
  7. the scripted baseline on validation: random layouts, 6 primitive shapes, 16+ layouts each
CPU only, no Isaac: code is read, not run. Results come from bench/results/ (or the paths given).

  python scripts/check_day1.py [RESULTS ...]
"""
import re
import sys

from goal_common import BENCHES, REPO, Report, arg_default, bench_spec, covers_shapes, find, imports, load_results, validate

WALK = "isaac/walk_grasp.py"
SPEC_USERS = [WALK]
EVALS = [WALK, "isaac/g1_grasp_rl.py", "isaac/g1_dexsuite.py"]
SBATCH = ["isaac/walk_cluster.sbatch", "isaac/py_cluster.sbatch", "isaac/sdg_cluster.sbatch", "vla/bench.sbatch"]


def spec_behaviour(rep, m):
    s = m.SPLITS
    ranges = sorted(s.values())
    rep.check(set(s) == {"train", "val", "test"} and all(a < b for a, b in ranges)
              and all(ranges[i][1] <= ranges[i + 1][0] for i in range(2)), "SPLITS: train, val, test seed ranges, disjoint", str(s))
    rep.check(all(m.split_of_seed(a) == k and m.split_of_seed(b - 1) == k for k, (a, b) in s.items())
              and m.split_of_seed(ranges[-1][1] + 10) is None, "split_of_seed matches SPLITS, None outside")
    rep.check(not m.lifted(0.064, 90.0), "a box tipped on its side (6.4 cm 'up', 90 deg) is not lifted")
    rep.check(m.lifted(0.10, 5.0), "10 cm up and upright is lifted")
    rep.check(not m.lifted(0.10, 70.0), "10 cm up but tilted 70 deg is not lifted")
    rep.check(m.upright(5.0, "standing") and not m.upright(45.0, "standing"), "upright: standing shapes by axis tilt")
    rep.check(m.upright(5.0, "lying") and not m.upright(60.0, "lying"), "upright: lying shapes by long axis from horizontal")
    rep.check(m.upright(170.0, "ball"), "upright: a ball always")
    rep.check(not m.released(0.0, True) and not m.released(0.10, True), "released: never while in contact")
    rep.check(not m.released(0.005, False) and m.released(0.08, False), "released: needs a gap as well")


def main(argv):
    rep = Report("Day 1: benchmarks fixed, baselines re-run")
    m, err = bench_spec()
    if rep.check(m is not None, "isaac/bench_spec.py imports on CPU (stdlib and numpy)", err):
        try:
            spec_behaviour(rep, m)
        except Exception as e:  # noqa: BLE001
            rep.check(False, "bench_spec functions callable as documented", f"{type(e).__name__}: {e}")
    for f in SPEC_USERS:
        rep.check(imports(f, "bench_spec"), f"{f} uses bench_spec")
    for f in EVALS:
        d = arg_default(f, "--split")
        rep.check(d == "val", f"{f}: --split defaults to val", f"default {d!r}" if d is not None else "no --split")
    walk = (REPO / WALK).read_text(encoding="utf-8")
    rep.check(re.search(r"v\[:,\s*1\]\s*=", walk) is not None, f"{WALK}: sideways walking command (vy) is set")
    bw = re.search(r"^B_W\s*=\s*(.+)$", walk, re.M)
    rep.check(bw is not None and "rand_layout" not in bw.group(1), f"{WALK}: table B width the same for both layouts",
              bw.group(1).split("#")[0].strip() if bw else "no B_W")
    for f in SBATCH:
        t = (REPO / f).read_text(encoding="utf-8") if (REPO / f).exists() else ""
        passes = "|| true" not in t and (re.search(r"set -e", t) or re.search(r"exit\s+\\?\$\{?rc\b", t))
        rep.check(bool(t) and bool(passes), f"{f}: exit code passed on")
    rep.check("BENCH_RESULT" in (REPO / WALK).read_text(encoding="utf-8"), f"{WALK} prints a BENCH_RESULT record")

    recs = load_results(argv)
    rep.note(f"{len(recs)} result records read")
    for r in recs:
        validate(r, rep, m)
    w = find(recs, bench="walk_carry", split="val", policy="scripted", **BENCHES["walk_carry"])
    if rep.check(len(w) >= 1, "walk_carry scripted baseline on val, random layouts, primitives"):
        covers_shapes(rep, w[-1], "walk_carry val baseline")
        rep.note(f"walk_carry val scripted: {sum(x['success'] for x in w[-1]['rows'])}/{len(w[-1]['rows'])}")
    return rep.print()


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1:]) else 1)
