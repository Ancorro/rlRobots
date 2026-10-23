"""Training rows for the G1 grasp chooser from the camera-only benchmark (chooser v4b, LOG.md 2026-10-06).

bench_general.py --robot=g1 --g1-grasp=chooser records, per pick attempt, what the chooser saw (object centre, half
sizes and narrow side from the camera points, so101/g1_campick.observe), the grasp it chose and the outcome. This turns
those records into rows in the grasp sampler's format (so101/g1_grasps.py), so g1_chooser.train reads them beside the
sampler's data. Label (training only): lifted = the touch check said holding after the lift (pick ok) AND the object
truly rose at least 5 cm (sim truth, rise_cm); a touch 'holding' without the rise counts as a miss, a loss during the
lift as dropped. Grasps the touch check rejected cannot be told apart from misses: in the benchmark they fail either way.
Layout key for the held-out split: 10,000,000 + scene seed (a scene's attempts stay on one side; sampler seeds < 1,000,000).

  python -m so101.g1_camdata runs/general/g1-primitives-s30-n300-cam30.json ... --out=runs/grasps/cam-v4b.jsonl
"""
import json
import sys
from pathlib import Path

RISE = 5.0  # cm


def rows(path):
    out = []
    for r in json.loads(Path(path).read_text())["results"]:
        for c in r.get("chooser", []):
            if not c or "held" not in c:  # scripted fallback, or a run without outcome records
                continue
            ok = c["held"] and c["rise_cm"] >= RISE
            result = "lifted" if ok else "dropped" if c["why"] == "dropped while lifting" else "missed"
            o = c["obs"]
            out.append(dict(obj=dict(shape=r["shape"], seed=10_000_000 + r["scene"], scene=r["scene"],
                                     object=r["object"], centre=o["centre"], half=o["half"], narrow=o["narrow"]),
                            grasp={k: c[k] for k in ("tilt", "azimuth", "heading", "height", "clear", "margin", "back", "hand")},
                            policy="chooser-camera", result=result, success=bool(ok), held=c["held"], why=c["why"],
                            rise_cm=c["rise_cm"], predicted=c["predicted"], features=c.get("features")))
    return out


def main():
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = Path(next(a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")))
    allr = [x for p in paths for x in rows(p)]
    out.write_text("".join(json.dumps(x) + "\n" for x in allr))
    n = len(allr)
    k = sum(x["success"] for x in allr)
    fp = sum(x["held"] and not x["success"] for x in allr)
    print(f"{out}: {n} attempts, {k} lifted (truth-confirmed), {fp} touch 'holding' without a 5 cm rise,"
          f" {sum(x['result'] == 'dropped' for x in allr)} dropped while lifting")
    for s in sorted({x["obj"]["shape"] for x in allr}):
        r = [x for x in allr if x["obj"]["shape"] == s]
        print(f"  {s:15s} {sum(x['success'] for x in r)}/{len(r)}")


if __name__ == "__main__":
    main()
