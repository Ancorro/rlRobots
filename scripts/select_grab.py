"""Day 2: pick the learned grab on validation and write bench/selection.json (scripts/check_day2.py's input).

Reads the result records (bench/results/ or the paths given), keeps the walk_carry validation records, and for every
learned policy name (anything but "scripted") takes its latest val record. The candidate with the most successes is
selected; ties go to the earlier file name. The scripted val record from the same machine is reported beside each
candidate (paired on the same layouts, McNemar p) but does not decide. "zero" (bench/run.sh zero: the learned-grab code
path with a zero-adjustment policy, a wiring check) is reported, never selected. Refuses a candidate whose val record
has no checkpoint. --write writes the selection (entries already in the file are kept).

  python scripts/select_grab.py [RESULTS ...] [--write]      (without --write: prints what it would choose)
"""
import json
import sys

from goal_common import BENCHES, REPO, SELECTION, find, load_results, mcnemar, same_machine, wins


def baseline_for(r, base):
    """The scripted val record to pair with: the latest from the same machine (settings host) when known, else the
    latest of all. GPU physics differs between machines by about as much as a small policy effect."""
    host = (r.get("settings") or {}).get("host")
    same = [b for b in base if host and (b.get("settings") or {}).get("host") == host]
    return (same or base)[-1]


def main(argv):
    write = "--write" in argv
    recs = load_results([a for a in argv if not a.startswith("--")])
    sel, ok = {}, True
    for bench, extra in BENCHES.items():
        val = find(recs, bench=bench, split="val", **extra)
        names = sorted({r["policy"] for r in val if r["policy"] != "scripted"})
        base = find(val, policy="scripted")
        if not names:
            print(f"{bench}: no learned val record yet ({len(base)} scripted)")
            ok = False
            continue
        cands = {n: find(val, policy=n)[-1] for n in names}
        if "zero" in cands and base:  # the wiring check: zero adjustment must not differ from the scripted grab
            z, b = cands["zero"], base[-1]
            rz = {(x["shape"], x["seed"]): x["success"] for x in z["rows"]}
            rb = {(x["shape"], x["seed"]): x["success"] for x in b["rows"]}
            keys = sorted(set(rz) & set(rb))
            d = sum(rz[k] != rb[k] for k in keys)
            print(f"{bench} wiring check: zero policy {sum(rz[k] for k in keys)} vs scripted {sum(rb[k] for k in keys)} on {len(keys)} common "
                  f"layouts, {d} differ (physics noise only; many = the learned path is not the scripted pinch)")
        names = [n for n in names if n != "zero"]
        if not names:
            print(f"{bench}: only the zero wiring check so far, nothing to select")
            ok = False
            continue
        for n, r in cands.items():
            line = f"{bench} {n}: {wins(r)}/{len(r['rows'])} on val [{r['_file']}] checkpoint {str(r['checkpoint'])[:12]}"
            if base:
                b = same_machine(r, base)
                ra = {(x["shape"], x["seed"]): x["success"] for x in r["rows"]}
                rb = {(x["shape"], x["seed"]): x["success"] for x in b["rows"]}
                keys = set(ra) & set(rb)
                oa, ob = sum(ra[k] and not rb[k] for k in keys), sum(rb[k] and not ra[k] for k in keys)
                line += f"; scripted {sum(rb[k] for k in keys)}/{len(keys)} on the same layouts, only learned {oa}, only scripted {ob}, p = {mcnemar(oa, ob):.3g}"
            print(line)
        best = max(names, key=lambda n: (wins(cands[n]), -names.index(n)))
        if not cands[best]["checkpoint"]:
            print(f"{bench}: {best} has no checkpoint in its record: not selectable")
            ok = False
            continue
        sel[bench] = dict(policy=best, checkpoint=cands[best]["checkpoint"], candidates=names)
        print(f"{bench}: select {best}")
    if write and sel:
        old = {}
        try:
            old = json.loads(SELECTION.read_text())
        except (OSError, json.JSONDecodeError):
            pass
        SELECTION.parent.mkdir(parents=True, exist_ok=True)
        SELECTION.write_text(json.dumps(old | sel, indent=1) + "\n")
        print(f"wrote {SELECTION.relative_to(REPO)}: {sorted(sel)}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1:]) else 1)
