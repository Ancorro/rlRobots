"""Compare g1_grasp_rl.py eval results: lifted / kept / why per file, and exact McNemar tests between pairs (rows
matched by shape and layout seed; only the two grasps' disagreements count).

  python isaac/compare_evals.py A.json B.json [C.json ...] [--metric=success|kept] [--vs=FIRST]
"""
import json
import sys
from math import comb
from pathlib import Path


def mcnemar(b, c):
    """Exact two-sided McNemar p: b, c = pairs where only the first / only the second succeeded."""
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def main():
    files = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
    metric = opts.get("metric", "success")
    res = {Path(f).stem: json.loads(Path(f).read_text()) for f in files}
    for k, r in res.items():
        kept = r.get("kept", "-")
        print(f"{k}: lifted {r['lifted']}/{r['of']}, kept {kept}/{r['of']}; why {r.get('why', {})}")
    names = list(res)
    pairs = [(names[0], n) for n in names[1:]] if "vs" in opts else [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    for a, b in pairs:
        ra = {(x["shape"], x["seed"], x.get("name")): x.get(metric, False) for x in res[a]["rows"]}
        rb = {(x["shape"], x["seed"], x.get("name")): x.get(metric, False) for x in res[b]["rows"]}
        if set(ra) != set(rb):  # household rows carry the fitted class: match on seed and name only
            ra = {k[1:]: v for k, v in ra.items()}
            rb = {k[1:]: v for k, v in rb.items()}
        keys = set(ra) & set(rb)
        only_a = sum(ra[k] and not rb[k] for k in keys)
        only_b = sum(rb[k] and not ra[k] for k in keys)
        print(f"{metric}: {a} {sum(ra[k] for k in keys)} vs {b} {sum(rb[k] for k in keys)} of {len(keys)}: "
              f"only first {only_a}, only second {only_b}, p = {mcnemar(only_a, only_b):.2g}")


if __name__ == "__main__":
    main()
