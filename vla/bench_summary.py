"""Print one line per benchmark result file: python vla/bench_summary.py [GLOB]  (default runs/general/*.json).
--why: also every attempt's outcome counted, placed by place and by shape, and the guard's rule counts summed."""
import glob
import json
import sys
from collections import Counter

args = [a for a in sys.argv[1:] if not a.startswith("--")]
for f in sorted(glob.glob(args[0] if args else "runs/general/*.json")):
    d = json.load(open(f))
    s, r = d["summary"], d["results"]
    why = Counter(x["why"][-1].split(";")[0] for x in r if not x["placed"]).most_common(3)
    print(f"{f.split('/')[-1]:38s} placed {s['placed']}/{s['tasks']}  picked {s['picked']}  first {s['picked_first_try']}"
          f"  drops {s.get('drops', '?')}  disturbed {s.get('disturbed_tasks', '?')}  wrong {s['wrong_object_looks']}  {s.get('wall_minutes', '?')} min  fails {dict(why)}")
    if "--why" in sys.argv:
        print("  every attempt:", dict(Counter(w for x in r for w in x["why"]).most_common()))
        for key in ("place", "shape"):
            n = Counter(x[key] for x in r)
            print(f"  placed by {key}:", {k: f"{sum(x['placed'] for x in r if x[key] == k)}/{v}" for k, v in sorted(n.items())})
        print("  guard:", dict(sum((Counter(x.get("guard", {})) for x in r), Counter())),
              f" mean s/task {s['mean_seconds']:.0f}")
