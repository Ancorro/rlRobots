"""Day 3 check: the headline run on held-out test seeds beats the GPU development best (19/96); 100 % is noted.

  1. walk_carry, test seeds, random layouts, 6 primitive shapes x 16+: the selected learned grab succeeds at a higher
     rate than the GPU best 19/96; the scripted grab on the same layouts is reported beside it
  2. test scored once: exactly one learned test record, with the checkpoint chosen on day 2
  3. without a selection (no learned grab won on day 2), the scripted grab is the entry and must beat the GPU best
CPU only, no Isaac.

  python scripts/check_day3.py [RESULTS ...]
"""
import sys

from goal_common import (BENCHES, Report, beats_sota, bench_spec, covers_shapes, find, load_results, paired, same_machine,
                         selection, validate)


def main(argv):
    rep = Report("Day 3: held-out test, above the GPU best")
    m, err = bench_spec()
    rep.check(m is not None, "isaac/bench_spec.py imports (seed splits checked against it)", err)
    recs = load_results(argv)
    rep.note(f"{len(recs)} result records read")
    for r in recs:
        validate(r, rep, m)
    sel = selection() or {}
    for bench, extra in BENCHES.items():
        s = sel.get(bench, {})
        learned = find(recs, bench=bench, split="test", policy="learned", **extra)
        base = find(recs, bench=bench, split="test", policy="scripted", **extra)
        if not s:  # no learned grab chosen on val: the (fixed) scripted grab is the day-3 entry
            rep.note(f"{bench}: no learned grab selected on val; the scripted grab is scored")
            if rep.check(len(learned) == 0, f"{bench}: no learned test record without a selection", f"{len(learned)} found") \
                    and rep.check(len(base) >= 1, f"{bench}: scripted test record"):
                covers_shapes(rep, base[-1], f"{bench} test")
                beats_sota(rep, base[-1], f"{bench} test (scripted)")
            continue
        if not rep.check(len(learned) == 1, f"{bench}: one learned test record (test scored once)", f"{len(learned)} found"):
            continue
        a = learned[0]
        rep.check(a["policy"] == s.get("policy") and a["checkpoint"] == s.get("checkpoint"),
                  f"{bench}: test used the policy and checkpoint chosen on val", f"{a['policy']} vs {s.get('policy')}")
        covers_shapes(rep, a, f"{bench} test")
        beats_sota(rep, a, f"{bench} test")
        if rep.check(bool(base), f"{bench}: scripted test record"):
            paired(rep, a, same_machine(a, base), f"{bench} test", must_win=False)  # reported beside the learned grab
    return rep.print()


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1:]) else 1)
