"""Day 2 check: a learned grab inside the walking scene, chosen on validation only, beats the scripted grab there.

The learned grab (g1_grasp_rl.py's v8 grasp-parameter policy, applied by walk_grasp.py --grab) must be:
  1. selectable from the command line: walk_grasp.py takes --grab (default "scripted") and --policy
  2. chosen on validation seeds: bench/selection.json names the chosen policy, its checkpoint sha256 and the
     validation records compared (scripts/select_grab.py writes it); every candidate record is on the val split
  3. better than the scripted grab on the same validation layouts (paired, McNemar p reported, not required)
Test seeds stay unused until day 3: no learned test record may exist yet unless day 3 is under way.

bench/selection.json: {"walk_carry": {"policy": "v8", "checkpoint": "<sha256>", "candidates": ["v8", "v8-b"]}}
CPU only, no Isaac.

  python scripts/check_day2.py [RESULTS ...]
"""
import sys

from goal_common import (BENCHES, Report, arg_default, bench_spec, find, load_results, paired, same_machine, selection, validate, wins)


def main(argv):
    rep = Report("Day 2: learned grab chosen on validation, beats scripted there")
    f = "isaac/walk_grasp.py"
    d = arg_default(f, "--grab")
    rep.check(d == "scripted", f"{f}: --grab, default scripted", f"default {d!r}" if d is not None else "no --grab")
    rep.check(arg_default(f, "--policy") is not None, f"{f}: --policy (checkpoint path)")
    m, err = bench_spec()
    rep.check(m is not None, "isaac/bench_spec.py imports (seed splits checked against it)", err)
    recs = load_results(argv)
    rep.note(f"{len(recs)} result records read")
    for r in recs:
        validate(r, rep, m)
    sel = selection()
    rep.check(isinstance(sel, dict), "bench/selection.json present")
    for bench, extra in BENCHES.items():
        s = (sel or {}).get(bench)
        if not rep.check(isinstance(s, dict) and all(k in s for k in ("policy", "checkpoint", "candidates")),
                         f"{bench}: selection names policy, checkpoint, candidates"):
            continue
        cands = [find(recs, bench=bench, split="val", policy=c, **extra) for c in s["candidates"]]
        rep.check(all(cands), f"{bench}: a val record for every candidate",
                  ", ".join(c for c, x in zip(s["candidates"], cands) if not x))
        scored = {c: wins(x[-1]) for c, x in zip(s["candidates"], cands) if x}
        if scored:
            rep.note(f"{bench} val candidates: {scored}")
            rep.check(s["policy"] == max(scored, key=scored.get), f"{bench}: the chosen policy is the val best")
        chosen = find(recs, bench=bench, split="val", policy=s["policy"], **extra)
        base = find(recs, bench=bench, split="val", policy="scripted", **extra)
        if rep.check(bool(chosen) and bool(base), f"{bench}: chosen and scripted records on val"):
            rep.check(chosen[-1]["checkpoint"] == s["checkpoint"], f"{bench}: val record's checkpoint is the selected one")
            paired(rep, chosen[-1], same_machine(chosen[-1], base), f"{bench} val")
    early = [r for r in find(recs, split="test", policy="learned")]
    rep.note(f"learned test records so far: {len(early)} (day 3 runs these once, with the selected checkpoints)")
    return rep.print()


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1:]) else 1)
