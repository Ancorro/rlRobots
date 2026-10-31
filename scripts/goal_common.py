"""Shared parts of the G1 goal checks (scripts/check_day1.py, check_day2.py, check_day3.py, check_goal.py).

The goal: beat the GPU (Isaac Lab) score for the walking G1 on one benchmark, fixed first:
  walk_carry  isaac/walk_grasp.py --carry --rand_layout, 6 primitive shapes x 16+ layouts: walk to a table, pinch one
              object, carry it to a second table, set it down. Development best 19/96 (2026-10-22); pass = above it.
That was a development run on a benchmark with the 2026-10-23 audit's holes. The checks run on CPU without Isaac: they
read the code (AST, no imports of Isaac scripts), import the pure isaac/bench_spec.py, and read result records.
(The two-hand box task and the scanned household objects are recorded by their scripts but are not part of this goal.)

Result records. A benchmark run prints one line to stdout at its end (so it lands in the job log):
  BENCH_RESULT {"bench": "walk_carry", "split": "val" | "test", "policy": "scripted" | <name>,
                "checkpoint": <sha256 of the weights> | null, "layout": "random" | "fixed",
                "objects": "primitives", "settings": {<every hyperparameter of the run, plus host and gpu>},
                "rows": [{"seed": int, "shape": str, "success": bool, "fell": bool, "upright": bool,
                          "released": bool, ...}, ...]}
The checks take files or folders: .json files holding one record, or logs with BENCH_RESULT lines (default: bench/results/).

isaac/bench_spec.py (standard library only; the Isaac scripts import it) must define:
  SPLITS = {"train": (start, stop), "val": (start, stop), "test": (start, stop)}   layout seed ranges, disjoint
  split_of_seed(seed) -> "train" | "val" | "test" | None
  lifted(rise_m, tilt_deg) -> bool          up from the start and still upright (a box tipped on its side reads ~6 cm up)
  upright(tilt_deg, kind) -> bool           kind "standing" (axis tilt), "lying" (long axis from horizontal), "ball"
  released(hand_gap_m, contact) -> bool     the hand is clear of the object: no contact and a gap
"""
import ast
import importlib.util
import json
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOTA = {"walk_carry": (19, 96)}  # README headline result, 2026-10-22
SHAPES = 6  # walk_carry: box, cylinder, ball, lying cylinder, capsule, egg
MIN_ROWS = {"walk_carry": 96}
BENCHES = {"walk_carry": dict(layout="random", objects="primitives")}  # the goal's benchmarks and how their records are found
SELECTION = REPO / "bench" / "selection.json"
ROW_KEYS = {"seed": int, "shape": str, "success": bool, "fell": bool, "upright": bool, "released": bool}


class Report:
    """Collects check lines; a day passes when no line failed."""

    def __init__(self, title):
        self.title, self.lines, self.failed = title, [], 0

    def check(self, ok, what, detail=""):
        self.lines.append(f"  {'ok  ' if ok else 'FAIL'} {what}{f'  ({detail})' if detail else ''}")
        self.failed += not ok
        return ok

    def note(self, text):
        self.lines.append(f"       {text}")

    def print(self):
        print(f"== {self.title}")
        print("\n".join(self.lines))
        print(f"   {self.title}: {'PASSED' if not self.failed else f'{self.failed} FAILED'}")
        return not self.failed


# --- code, read without importing Isaac ----------------------------------------------------------------------------
def tree(rel):
    p = REPO / rel
    return ast.parse(p.read_text(encoding="utf-8")) if p.exists() else None


def arg_default(rel, flag):
    """The default of add_argument(flag, ...) in a script: the value, "<none>" without one, None if no such argument."""
    t = tree(rel)
    for n in ast.walk(t) if t else []:
        if (isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument" and n.args
                and isinstance(n.args[0], ast.Constant) and n.args[0].value == flag):
            for k in n.keywords:
                if k.arg == "default":
                    return k.value.value if isinstance(k.value, ast.Constant) else "<expr>"
            return "<none>"
    return None


def imports(rel, name):
    t = tree(rel)
    for n in ast.walk(t) if t else []:
        if isinstance(n, ast.Import) and any(a.name.split(".")[-1] == name for a in n.names):
            return True
        if isinstance(n, ast.ImportFrom) and ((n.module or "").split(".")[-1] == name or any(a.name == name for a in n.names)):
            return True
    return False


def bench_spec():
    """isaac/bench_spec.py as a module, or the error text."""
    p = REPO / "isaac" / "bench_spec.py"
    if not p.exists():
        return None, "isaac/bench_spec.py missing"
    try:
        spec = importlib.util.spec_from_file_location("bench_spec", p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m, ""
    except Exception as e:  # noqa: BLE001  (any import error is a failed check, not a crash)
        return None, f"import failed: {type(e).__name__}: {e}"


# --- result records ------------------------------------------------------------------------------------------------
def load_results(paths):
    """Records from .json files and BENCH_RESULT lines in logs; folders are read file by file."""
    files = []
    for p in map(Path, paths or [REPO / "bench" / "results"]):
        files += sorted(f for f in p.iterdir() if f.is_file()) if p.is_dir() else [p] if p.exists() else []  # not old-*/
    recs = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        if f.suffix == ".json":
            try:
                d = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(d, dict) and "bench" in d:
                recs.append(dict(d, _file=f.name))
            continue
        for line in text.splitlines():
            i = line.find("BENCH_RESULT ")
            if i >= 0:
                try:
                    recs.append(dict(json.loads(line[i + len("BENCH_RESULT "):]), _file=f.name))
                except json.JSONDecodeError:
                    recs.append(dict(_file=f.name, _bad=line[:80]))
    return recs


def label(r):
    return f"{r.get('bench')}/{r.get('split')}/{r.get('policy')}/{r.get('layout')}/{r.get('objects')} [{r.get('_file')}]"


def validate(r, rep, spec):
    """One record: fields present, settings listed, every success really upright, released and standing, seeds in
    the record's split."""
    if "_bad" in r:
        return rep.check(False, f"readable BENCH_RESULT in {r['_file']}", r["_bad"])
    name = label(r)
    ok = rep.check(all(k in r for k in ("bench", "split", "policy", "checkpoint", "layout", "objects", "settings", "rows")),
                   f"{name}: record has bench, split, policy, checkpoint, layout, objects, settings, rows")
    if not ok:
        return False
    ok &= rep.check(isinstance(r["settings"], dict) and len(r["settings"]) >= 3, f"{name}: hyperparameters listed in settings")
    rows = r["rows"]
    bad = [i for i, x in enumerate(rows) if not all(isinstance(x.get(k), t) for k, t in ROW_KEYS.items())]
    ok &= rep.check(not bad, f"{name}: every row has {', '.join(ROW_KEYS)}", f"rows {bad[:5]}" if bad else "")
    if bad:
        return False
    loose = [x["seed"] for x in rows if x["success"] and (x["fell"] or not x["upright"] or not x["released"])]
    ok &= rep.check(not loose, f"{name}: no success that fell, tipped or still grips", f"seeds {loose[:5]}" if loose else "")
    if spec is not None:
        wrong = [x["seed"] for x in rows if spec.split_of_seed(x["seed"]) != r["split"]]
        ok &= rep.check(not wrong, f"{name}: every seed in the {r['split']} range", f"seeds {wrong[:5]}" if wrong else "")
    keys = [(x["shape"], x["seed"]) for x in rows]
    ok &= rep.check(len(set(keys)) == len(keys), f"{name}: no repeated layouts")
    return ok


def wins(r):
    return sum(x["success"] for x in r["rows"])


def find(recs, **want):
    return [r for r in recs if "_bad" not in r and all(
        (r.get(k) != "scripted" if v == "learned" else r.get(k) == v) if k == "policy" else r.get(k) == v
        for k, v in want.items())]


def mcnemar(b, c):
    """Exact two-sided McNemar p: b, c = layouts where only the first / only the second succeeded."""
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def same_machine(r, others):
    """Of `others`, the latest record from r's machine (settings host) if any, else the latest: GPU physics is not
    deterministic across machines (val scripted 2026-10-26: RTX 8000 40/96 vs RTX 3090 41/96, 9 layouts differ), so
    pairs should come from one."""
    host = (r.get("settings") or {}).get("host")
    same = [o for o in others if host and (o.get("settings") or {}).get("host") == host]
    return (same or others)[-1]


def paired(rep, a, b, what, must_win=True):
    """a (learned) vs b (scripted) on the same layouts: both must cover exactly the same (shape, seed) set."""
    ra = {(x["shape"], x["seed"]): x["success"] for x in a["rows"]}
    rb = {(x["shape"], x["seed"]): x["success"] for x in b["rows"]}
    same = rep.check(set(ra) == set(rb), f"{what}: learned and scripted on the same layouts",
                     f"{len(set(ra) ^ set(rb))} differ" if set(ra) != set(rb) else f"{len(ra)} layouts")
    if not same:
        return False
    only_a = sum(ra[k] and not rb[k] for k in ra)
    only_b = sum(rb[k] and not ra[k] for k in ra)
    rep.note(f"{what}: learned {wins(a)}/{len(ra)}, scripted {wins(b)}/{len(rb)}; only learned {only_a}, "
             f"only scripted {only_b}, McNemar p = {mcnemar(only_a, only_b):.3g}")
    return rep.check(wins(a) > wins(b), f"{what}: learned places more than scripted") if must_win else True


def beats_sota(rep, r, what):
    """Pass: the test success rate is above the GPU development best (19/96; owner 2026-10-27: "perfect was a chase
    goal, improvement is good"), with enough layouts. 100 % stays a note."""
    k, n = SOTA[r["bench"]]
    w, m = wins(r), len(r["rows"])
    ok = rep.check(m >= MIN_ROWS[r["bench"]] and w / m > k / n, f"{what}: above the GPU best {k}/{n} = {k / n:.0%}",
                   f"{w}/{m} = {w / m:.1%}")
    rep.note(f"{what}: {'every layout succeeded' if w == m else f'{m - w} of {m} layouts still fail'} (100 % was the original target)")
    return ok


def covers_shapes(rep, r, what):
    shapes = {x["shape"] for x in r["rows"]}
    per = min(sum(x["shape"] == s for x in r["rows"]) for s in shapes) if shapes else 0
    return rep.check(len(shapes) >= SHAPES and per >= 16, f"{what}: {SHAPES} shapes, 16+ layouts each",
                     f"{len(shapes)} shapes, fewest {per}")


def selection():
    """bench/selection.json: the learned grab picked on validation (written on day 2), or None."""
    try:
        return json.loads(SELECTION.read_text())
    except (OSError, json.JSONDecodeError):
        return None
