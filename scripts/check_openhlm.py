"""OpenHLM check: read evaluation logs for episodes of the OpenHLM/OpenHLM-data sets (Hugging Face) and report the
success rate. Passes when a sample of exactly 20 distinct episodes reaches at least 50 % success.
Makes no runs and needs no GPU; it only reads logs.

  python scripts/check_openhlm.py [LOGS ...]            files or folders, default logs/openhlm/
  python scripts/check_openhlm.py --offline [LOGS ...]  skip checking episode ids against the dataset's meta files
  options: --episodes=20 --min-rate=0.5

Logs are JSON or JSON Lines (.json, .jsonl, .log). Each per-episode record looks like
  {"set": "g1_HLM-12_humi", "episode": 12, "success": true}
(aliases: dataset/subset for set, episode_index for episode, succeeded/ok for success); other fields are ignored.
Exit 0 only when every check passes.
"""
import json
import pathlib
import sys
import urllib.request

REPO = "https://huggingface.co/datasets/OpenHLM/OpenHLM-data/resolve/main"
SETS = ["g1_HLM-12_full_teleop", "g1_HLM-12_humi", "g1_HLM-12_stationary_teleop",
        "long_g1_5_fruits_full_teleop", "long_g1_5_fruits_humi"]
DEFAULT_LOGS = pathlib.Path(__file__).resolve().parent.parent / "logs" / "openhlm"
ALIASES = {"set": ("set", "dataset", "subset"), "episode": ("episode", "episode_index"),
           "success": ("success", "succeeded", "ok")}


def pick(rec, key):
    return next((rec[k] for k in ALIASES[key] if k in rec), None)


def read_records(paths):
    files = []
    for p in map(pathlib.Path, paths):
        files += sorted(f for f in p.rglob("*") if f.suffix in (".json", ".jsonl", ".log")) if p.is_dir() else [p]
    recs, bad = [], []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        try:
            data = json.loads(text)
            items = data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            items = []
            for n, line in enumerate(text.splitlines(), 1):
                line = line.strip()
                if line.startswith("{"):
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError:
                        bad.append(f"{f.name}:{n}")
        recs += [r for r in items if isinstance(r, dict)]
    return files, recs, bad


def dataset_lengths(name):
    """Episode index -> length from the set's meta/episodes.jsonl, or None when it can't be fetched."""
    try:
        with urllib.request.urlopen(f"{REPO}/{name}/meta/episodes.jsonl", timeout=30) as r:
            return {e["episode_index"]: e["length"] for e in map(json.loads, r.read().decode().splitlines()) if e}
    except Exception:
        return None


class Report:
    def __init__(self):
        self.ok = True

    def check(self, cond, what, why=""):
        print(f"  [{'pass' if cond else 'FAIL'}] {what}" + ("" if cond or not why else f": {why}"))
        self.ok &= bool(cond)
        return cond


def main(argv):
    opt = dict(a[2:].split("=", 1) for a in argv if a.startswith("--") and "=" in a)
    offline = "--offline" in argv
    want, min_rate = int(opt.get("episodes", 20)), float(opt.get("min-rate", 0.5))
    paths = [a for a in argv if not a.startswith("--")] or [DEFAULT_LOGS]

    print("OpenHLM: success rate on a small sample, from logs")
    rep = Report()
    files, recs, bad = read_records(p for p in paths if pathlib.Path(p).exists())
    print(f"  {len(files)} log files, {len(recs)} records read")
    rep.check(not bad, "every JSON line parses", ", ".join(bad[:5]))

    eps = [r for r in recs if pick(r, "episode") is not None]
    keys = [(pick(r, "set"), pick(r, "episode")) for r in eps]
    rep.check(bool(eps), "per-episode records found", f"none in {', '.join(map(str, paths))}")
    unknown = sorted({str(s) for s, _ in keys if s not in SETS})
    rep.check(not unknown, "every record names an OpenHLM set", "unknown: " + ", ".join(unknown))
    dupes = sorted({k for k in keys if keys.count(k) > 1}, key=str)
    rep.check(not dupes, "each episode scored once", f"{len(dupes)} repeated, e.g. {dupes[:3]}")
    n = len(set(keys))
    rep.check(n == want, f"sample size: {n} distinct episodes (needs {want})")

    if not offline:
        for name in sorted({s for s, _ in keys if s in SETS}):
            lengths = dataset_lengths(name)
            if lengths is None:
                print(f"  [note] {name}: meta/episodes.jsonl not reachable; ids not checked (use --offline to skip)")
                continue
            missing = sorted({e for s, e in keys if s == name and e not in lengths})
            rep.check(not missing, f"{name}: episode ids exist in the dataset ({len(lengths)} episodes)",
                      f"not found: {missing[:5]}")

    no_result = [k for r, k in zip(eps, keys) if not isinstance(pick(r, "success"), (bool, int))]
    rep.check(not no_result, "every episode has a success value", f"{len(no_result)} missing, e.g. {no_result[:3]}")
    wins = sum(bool(pick(r, "success")) for r in eps if isinstance(pick(r, "success"), (bool, int)))
    rate = wins / len(eps) if eps else 0.0
    for name in sorted({str(s) for s, _ in keys}):
        sub = [r for r, (s, _) in zip(eps, keys) if str(s) == name]
        print(f"  {name}: {sum(bool(pick(r, 'success')) for r in sub)}/{len(sub)}")
    rep.check(eps and rate >= min_rate, f"success {wins}/{len(eps)} = {rate:.0%} (at least {min_rate:.0%})")

    print(f"\n== OpenHLM: {'passed' if rep.ok else 'not yet'}")
    return rep.ok


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1:]) else 1)
