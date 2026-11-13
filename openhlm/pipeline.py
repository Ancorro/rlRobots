"""The OpenHLM-data benchmark in one command, inside a time budget: split, train every candidate on every set (one
worker per GPU, the ridge baseline on the CPU), select per set on validation, score the test sample once, write the
per-episode records scripts/check_openhlm.py reads, and run that check.

  python -m openhlm.pipeline --tag run1 [--data DIR] [--sets a,b,...] [--candidates mlp,mlp-wide,ridge]
         [--gpus auto|0,1|cpu] [--budget-min 26] [--camera KEY --frames DIR] [--test] [--force] [--seed 0]
         [--out runs/openhlm] [--logs logs/openhlm]

Stages (each skipped when its files exist, so a stopped run resumes):
  split    <out>/<tag>/split.json: the contract's split (openhlm/spec.py) of the sets found
  train    <out>/<tag>/<set>/<candidate>.{pt,json,-val.jsonl} by openhlm/train.py; units are spread over the workers,
           each unit gets an equal share of the training budget (the budget minus a reserve for the test and start-up)
  select   <out>/<tag>/selection.json: per set, the candidate with the most validation successes (ties: mean skill)
  test     (--test) the selected candidates on the test episodes, once: <logs>/test-<tag>.jsonl (the check's input; the
           pipeline refuses to write a second test record unless --force, which moves the old ones to <out>/old/),
           <out>/<tag>/test-refs.jsonl (hold and ridge on the same episodes, for the paired comparison),
           <out>/<tag>/report.md, then scripts/check_openhlm.py --offline <logs>
The budget is wall time for this process; training units are stopped by their own deadlines.
"""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from openhlm import policy, spec  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
PY = sys.executable


def log(*a):
    print(f"[openhlm {datetime.datetime.now():%H:%M:%S}]", *a, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def devices(spec_str):
    """--gpus auto|cpu|0,1 -> list of worker device strings ('cuda:<visible index>' or 'cpu')."""
    if spec_str == "cpu":
        return ["cpu"]
    if spec_str == "auto":
        try:
            import torch
            n = torch.cuda.device_count()
        except Exception:  # noqa: BLE001
            n = 0
        return [f"cuda:{i}" for i in range(n)] or ["cpu"]
    return [f"cuda:{int(g)}" for g in spec_str.split(",") if g != ""]


def worker_env(device):
    env = dict(os.environ)
    if device.startswith("cuda:"):
        vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        ids = vis.split(",") if vis else None
        i = int(device.split(":")[1])
        env["CUDA_VISIBLE_DEVICES"] = ids[i] if ids else str(i)
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def run_units(units, a, run_dir, split_file, budget_s):
    """units: [(device, set, candidate)] grouped by device; each device's list runs in order in its own thread."""
    by_dev = {}
    for dev, s, c in units:
        by_dev.setdefault(dev, []).append((s, c))
    (run_dir / "logs").mkdir(exist_ok=True)
    results = {}

    def work(dev, items):
        end = time.time() + budget_s
        warned = False
        for i, (s, c) in enumerate(items):
            done = run_dir / s / f"{c}.json"
            if done.exists():
                results[(s, c)] = "cached"
                continue
            left = [u for u in items[i:] if not (run_dir / u[0] / f"{u[1]}.json").exists()]
            weights = sum(policy.CANDIDATES[u[1]].get("weight", 1) for u in left) or 1  # remaining time shared by weight
            per_unit = (end - time.time()) * policy.CANDIDATES[c].get("weight", 1) / weights - a.startup_s
            if per_unit < a.min_unit_s:
                if not warned:
                    log(f"{dev}: {len(left)} units left in {end - time.time():.0f} s; using the floor of {a.min_unit_s:.0f} s each (the budget will overrun)")
                warned, per_unit = True, a.min_unit_s
            cmd = [PY, "-m", "openhlm.train", "--set", s, "--candidate", c, "--out", str(run_dir), "--split-file", str(split_file),
                   "--budget-s", f"{per_unit:.0f}", "--device", "cuda" if dev.startswith("cuda") else "cpu", "--seed", str(a.seed),
                   "--eval-every", str(a.eval_every)]
            if a.data:
                cmd += ["--data", a.data]
            if a.cache:
                cmd += ["--cache", a.cache]
            if a.state_key:
                cmd += ["--state-key", a.state_key]
            if a.action_key:
                cmd += ["--action-key", a.action_key]
            if policy.CANDIDATES[c].get("vision"):
                cmd += ["--camera", a.camera, "--frames", a.frames]
            log(f"{dev}: {s} / {c} for {per_unit:.0f} s")
            with open(run_dir / "logs" / f"{s}-{c}.log", "w") as f:
                r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=worker_env(dev), cwd=str(REPO))
            results[(s, c)] = r.returncode
            if r.returncode != 0:
                tail = (run_dir / "logs" / f"{s}-{c}.log").read_text(errors="replace").splitlines()[-8:]
                log(f"{dev}: {s} / {c} FAILED rc={r.returncode}:\n  " + "\n  ".join(tail))

    threads = [threading.Thread(target=work, args=(d, items)) for d, items in by_dev.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def ensemble_row(a, run_dir, s, split, members):
    """The ensemble of the set's learned candidates scored on validation (no training: a few seconds)."""
    import numpy as np
    import torch
    from openhlm import data
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pols, payload, cams = [], None, []
    for c in members:
        pol, payload = policy.load(str(run_dir / s / f"{c}.pt"), device=dev)
        pols.append(pol)
        cams += [x for x in (payload.get("cameras") or []) if x not in cams]
    val_eps = [int(e) for e, r in split[s].items() if r == "val"]
    sd = data.SetData(s, data.set_dir(a.data, s), episodes=val_eps, cameras=cams, frames_dir=a.frames or None,
                      state_key=payload.get("state_key"), action_key=payload.get("action_key"), cache_dir=a.cache or None)
    stats = {k: np.asarray(v, np.float32) for k, v in payload["stats"].items()}
    hmap = np.asarray(payload["hmap"], np.int64)
    std = np.maximum(stats["action_std"], spec.STD_FLOOR)
    recs = policy.score(sd, policy.Ensemble(pols), sd.usable(val_eps), std, hmap, "val", "ens")
    spec.write_jsonl(run_dir / s / "ens-val.jsonl", recs)
    summ = spec.summary(recs)
    r = dict(candidate="ens", members=members, val=summ, checkpoint=[str(run_dir / s / f"{c}.pt") for c in members], steps=0, seconds=0,
             refs=json.load(open(run_dir / s / f"{members[0]}.json"))["refs"])
    (run_dir / s / "ens.json").write_text(json.dumps(r, indent=1))
    return r


def select(run_dir, sets, candidates, a=None, split=None):
    """Per set: the candidate with the most validation successes, ties by mean skill, then by candidate order. With
    two or more learned candidates their ensemble (mean chunk) is scored on validation and competes too."""
    sel = {}
    for s in sets:
        rows = []
        for c in candidates:
            p = run_dir / s / f"{c}.json"
            if p.exists():
                r = json.load(open(p))
                rows.append((r["val"]["successes"], r["val"]["mean_skill"], -candidates.index(c), c, r))
        learned = [c for c in candidates if c != "ridge" and (run_dir / s / f"{c}.pt").exists()]
        if a is not None and len(learned) >= 2 and not a.no_ensemble:
            try:
                r = json.load(open(run_dir / s / "ens.json")) if (run_dir / s / "ens.json").exists() else ensemble_row(a, run_dir, s, split, learned)
                rows.append((r["val"]["successes"], r["val"]["mean_skill"], -len(candidates), "ens", r))
            except Exception as ex:  # noqa: BLE001
                log(f"{s}: ensemble not scored ({type(ex).__name__}: {ex})")
        if not rows:
            continue
        rows.sort(reverse=True)
        _, _, _, c, r = rows[0]
        ck = r["checkpoint"]
        sha = sha256(ck) if isinstance(ck, str) else hashlib.sha256("".join(sha256(x) for x in ck).encode()).hexdigest()
        sel[s] = dict(candidate=c, checkpoint=ck, sha256=sha, val=r["val"], members=r.get("members"),
                      candidates={row[3]: dict(successes=row[4]["val"]["successes"], mean_skill=round(row[4]["val"]["mean_skill"], 4),
                                               episodes=row[4]["val"]["episodes"], steps=row[4].get("steps", 0), seconds=row[4].get("seconds"),
                                               best_step=row[4].get("best_step", 0), cameras=row[4].get("cameras", []))
                                  for row in rows},
                      hold_val=r["refs"]["hold"])
    return sel


def load_selected(info, dev):
    """The selected policy of a set (a single checkpoint or the ensemble), its first payload and the cameras it needs."""
    if info["candidate"] == "ens":
        members = [policy.load(ck, device=dev) for ck in info["checkpoint"]]
        pol, payload = policy.Ensemble([m for m, _ in members]), members[0][1]
        cams = []
        for _, pl in members:
            cams += [x for x in (pl.get("cameras") or []) if x not in cams]
        return pol, payload, cams
    pol, payload = policy.load(info["checkpoint"], device=dev)
    return pol, payload, payload.get("cameras") or []


def diagnose(a, run_dir, split, sel):
    """Where the selected policy's error is, on validation: by chunk step and by action dimension, against hold.
    Written to <run>/<set>/diagnostics.json and shown in the report (no effect on selection or the test)."""
    import numpy as np
    import torch
    from openhlm import data
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    for s, info in sel.items():
        try:
            pol, payload, cams = load_selected(info, dev)
            val_eps = [int(e) for e, r in split[s].items() if r == "val"]
            sd = data.SetData(s, data.set_dir(a.data, s), episodes=val_eps, cameras=cams, frames_dir=a.frames or None,
                              state_key=payload.get("state_key"), action_key=payload.get("action_key"), cache_dir=a.cache or None)
            stats = {k: np.asarray(v, np.float32) for k, v in payload["stats"].items()}
            hmap = np.asarray(payload["hmap"], np.int64)
            std = np.maximum(stats["action_std"], spec.STD_FLOOR)
            d = {}
            policy.score(sd, pol, sd.usable(val_eps), std, hmap, "val", info["candidate"], diagnostics=d)
            n = max(d.get("chunks", 0), 1)
            out[s] = dict(step=(d["step"] / n).round(4).tolist(), step_hold=(d["step_hold"] / n).round(4).tolist(),
                          dim=(d["dim"] / n / sd.horizon).round(4).tolist(), dim_hold=(d["dim_hold"] / n / sd.horizon).round(4).tolist(),
                          hold_map=hmap.tolist(), chunks=int(n), candidate=info["candidate"])
            (run_dir / s / "diagnostics.json").write_text(json.dumps(out[s]))
        except Exception as ex:  # noqa: BLE001
            log(f"{s}: diagnostics failed ({type(ex).__name__}: {ex})")
    return out


def test(a, run_dir, split, sel, logs_dir):
    import numpy as np
    import torch
    from openhlm import data
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    records, refs = [], []
    for s, info in sel.items():
        test_eps = [int(e) for e, r in split[s].items() if r == "test"]
        train_eps = [int(e) for e, r in split[s].items() if r == "train"]
        pol, payload, cams = load_selected(info, dev)
        sd = data.SetData(s, data.set_dir(a.data, s), episodes=test_eps, cameras=cams, frames_dir=a.frames or None,
                          state_key=payload.get("state_key"), action_key=payload.get("action_key"), cache_dir=a.cache or None)
        if sd.history != payload["history"] or sd.horizon != payload["horizon"]:
            raise SystemExit(f"{s}: checkpoint history/horizon {payload['history']}/{payload['horizon']} vs data {sd.history}/{sd.horizon}")
        stats = {k: np.asarray(v, np.float32) for k, v in payload["stats"].items()}  # the training split's statistics
        hmap = np.asarray(payload["hmap"], np.int64)
        std = np.maximum(stats["action_std"], spec.STD_FLOOR)
        short = [e for e in test_eps if e not in sd.usable(test_eps)]
        if short:
            raise SystemExit(f"{s}: test episodes too short for one chunk: {short}")
        recs = policy.score(sd, pol, test_eps, std, hmap, "test", info["candidate"])
        for r in recs:
            r["checkpoint"] = info["sha256"]
            r["hold_map"] = payload["hmap"]
            r["hold_source"] = payload.get("hold_source", "names")
        records += recs
        refs += policy.score(sd, policy.Hold(hmap), test_eps, std, hmap, "test", "hold")
        rp = run_dir / s / "ridge.pt"
        if rp.exists() and info["candidate"] != "ridge":
            ridge, _ = policy.load(rp)
            refs += policy.score(sd, ridge, test_eps, std, hmap, "test", "ridge")
        log(f"test {s}: {sum(r['success'] for r in recs)}/{len(recs)} with {info['candidate']} (train episodes {len(train_eps)})")
    logs_dir.mkdir(parents=True, exist_ok=True)
    spec.write_jsonl(logs_dir / f"test-{a.tag}.jsonl", records)
    spec.write_jsonl(run_dir / "test-refs.jsonl", refs)
    return records, refs


def gpu_names():
    try:
        import torch
        return ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())) or "no GPU"
    except Exception:  # noqa: BLE001
        return "unknown"


def report(a, run_dir, split, sel, records, refs, started, budget_min, diag=None):
    summ = spec.summary(records) if records else None
    lines = [f"# OpenHLM-data run `{a.tag}`", "",
             f"{datetime.datetime.now():%Y-%m-%d %H:%M}, host {platform.node()} ({gpu_names()}), {time.time() - started:.0f} s of a {budget_min:.0f} min budget, "
             f"devices {', '.join(a.device_list)}, seed {a.seed}, cameras {a.camera or 'none (state only)'}, candidates {', '.join(sel[next(iter(sel))]['candidates']) if sel else ''}.", "",
             "Contract: openhlm/spec.py (history %.2f s, horizon %.1f s, success = skill >= %.2f against the hold baseline)." % (spec.HISTORY_S, spec.HORIZON_S, spec.SKILL_PASS), "",
             "## Split", "", "| set | train | val | test |", "|---|---|---|---|"]
    for s, roles in split.items():
        c = {r: sum(1 for v in roles.values() if v == r) for r in ("train", "val", "test")}
        lines.append(f"| {s} | {c['train']} | {c['val']} | {c['test']} |")
    lines += ["", "## Validation (selection)", "", "| set | candidate | val successes | mean skill | steps (best) | s | cameras |", "|---|---|---|---|---|---|---|"]
    for s, info in sel.items():
        for c, v in info["candidates"].items():
            mark = " **(selected)**" if c == info["candidate"] else ""
            rising = " still improving" if v["steps"] and v.get("best_step", 0) >= 0.9 * v["steps"] else ""
            lines.append(f"| {s} | {c}{mark} | {v['successes']}/{v['episodes']} | {v['mean_skill']:.3f} | {v['steps']} ({v.get('best_step', 0)}){rising} | "
                         f"{v['seconds']} | {','.join(v.get('cameras') or []) or '-'} |")
    if diag:
        lines += ["", "## Where the error is (selected policy on validation, normalised absolute error, hold in brackets)", "",
                  "| set | step 1 | 5 | 10 | 15 | 20 | 25 | last | dims with the largest error (dim: policy [hold], * = held from state) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for s, d in diag.items():
            st, sh = d["step"], d["step_hold"]
            cells = [f"{st[i]:.3f} [{sh[i]:.3f}]" for i in (0, 4, 9, 14, 19, 24, len(st) - 1) if i < len(st)]
            order = sorted(range(len(d["dim"])), key=lambda i: -d["dim"][i])[:8]
            dims = ", ".join(f"{i}{'*' if d['hold_map'][i] >= 0 else ''}: {d['dim'][i]:.2f} [{d['dim_hold'][i]:.2f}]" for i in order)
            lines.append(f"| {s} | " + " | ".join(cells) + f" | {dims} |")
    if summ:
        n = getattr(a, "test_number", 1)
        lines += ["", f"## Test (run #{n}{'' if n == 1 else ' on the same test episodes: see the history below'})", ""]
        if getattr(a, "history", None):
            lines += ["Earlier test runs (runs/openhlm/test-history.jsonl): " + "; ".join(
                f"#{h['number']} {h['date']} {h['successes']}/{h['episodes']} (mean skill {h['mean_skill']:.3f}, {','.join(h['candidates'])}"
                f"{', camera ' + h['camera'] if h.get('camera') else ''})" for h in a.history), ""]
        lines += [
                  f"**{summ['successes']}/{summ['episodes']} episodes succeed ({summ['rate']:.0%}, Wilson 95 % {summ['wilson95'][0]:.2f}-{summ['wilson95'][1]:.2f})**; "
                  f"mean skill {summ['mean_skill']:.3f}, median {summ['median_skill']:.3f}; at skill >= " +
                  ", ".join(f"{l}: {n}" for l, n in summ["at_level"].items()) + ".", ""]
        lines += ["| set | candidate | test successes | mean skill | hold map (action dim -> state dim, -1 = last command) |", "|---|---|---|---|---|"]
        for s, v in summ["by_set"].items():
            hm = next((r for r in records if r["set"] == s), {})
            lines.append(f"| {s} | {sel[s]['candidate']} | {v['successes']}/{v['episodes']} | {v['mean_skill']:.3f} | "
                         f"{sum(1 for x in hm.get('hold_map', []) if x >= 0)}/{len(hm.get('hold_map', []))} {hm.get('hold_source', '')}: {hm.get('hold_map', '')} |")
        for ref in ("ridge", "hold"):
            rr = [r for r in refs if r["policy"] == ref]
            if rr:
                rs = spec.summary(rr)
                key = {(r["set"], r["episode"]): r for r in rr}
                pairs = [(r, key[(r["set"], r["episode"])]) for r in records if (r["set"], r["episode"]) in key]
                b = sum(x["success"] and not y["success"] for x, y in pairs)
                c = sum(y["success"] and not x["success"] for x, y in pairs)
                better = sum(x["skill"] > y["skill"] for x, y in pairs)
                lines += ["", f"Reference `{ref}` on the same episodes: {rs['successes']}/{rs['episodes']} succeed, mean skill {rs['mean_skill']:.3f}; "
                          f"selected policy better on {better}/{len(pairs)} episodes by skill; only-selected {b}, only-{ref} {c}, McNemar p = {spec.mcnemar(b, c):.3g}."]
        lines += ["", "| set | episode | policy | success | skill | error | error hold | chunks | length |", "|---|---|---|---|---|---|---|---|---|"]
        for r in records:
            lines.append(f"| {r['set']} | {r['episode']} | {r['policy']} | {'yes' if r['success'] else 'no'} | {r['skill']:.3f} | {r['error']:.4f} | {r['error_hold']:.4f} | {r['chunks']} | {r['length']} |")
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summ


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--data", default="")
    p.add_argument("--sets", default="", help="comma-separated; default: every set found")
    p.add_argument("--candidates", default="mlp,mlp-wide,ridge")
    p.add_argument("--gpus", default="auto")
    p.add_argument("--budget-min", type=float, default=26.0, help="wall time for the whole run (training, selection, test)")
    p.add_argument("--reserve-min", type=float, default=2.5, help="kept back for the ensemble, the diagnostics, the test and start-up")
    p.add_argument("--startup-s", type=float, default=20.0, help="per training unit, for loading the data")
    p.add_argument("--min-unit-s", type=float, default=10.0, help="floor of a training unit's budget")
    p.add_argument("--eval-every", type=float, default=20.0)
    p.add_argument("--camera", default="", help="vision: camera key(s) from the frame cache (openhlm/frames.py)")
    p.add_argument("--frames", default=str(REPO / "runs" / "openhlm" / "frames"))
    p.add_argument("--out", default=str(REPO / "runs" / "openhlm"))
    p.add_argument("--logs", default=str(REPO / "logs" / "openhlm"), help="where the test record goes (scripts/check_openhlm.py reads it)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--state-key", default="", help="table column of the state (default: auto, spec.STATE_KEYS)")
    p.add_argument("--action-key", default="", help="table column of the action (default: auto, spec.ACTION_KEYS)")
    p.add_argument("--test", action="store_true", help="score the test sample once after selection")
    p.add_argument("--force", action="store_true", help="(kept for compatibility: a second test is now always counted and allowed unless --refuse-retest)")
    p.add_argument("--refuse-retest", action="store_true", help="with --test: stop when a test record already exists")
    p.add_argument("--no-ensemble", action="store_true", help="do not score the ensemble of the learned candidates on validation")
    p.add_argument("--cache", default=str(REPO / "runs" / "openhlm" / "cache"), help="per-set .npz caches of the tables ('' to read parquet every time)")
    p.add_argument("--no-check", action="store_true")
    a = p.parse_args(argv)
    started = time.time()
    from openhlm import data

    run_dir = pathlib.Path(a.out) / a.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = pathlib.Path(a.logs)
    a.device_list = devices(a.gpus)
    candidates = [c for c in a.candidates.split(",") if c]
    unknown = [c for c in candidates if c not in policy.CANDIDATES]
    if unknown:
        raise SystemExit(f"unknown candidates {unknown}; known: {sorted(policy.CANDIDATES)}")
    if not a.camera:
        skipped = [c for c in candidates if policy.CANDIDATES[c].get("vision")]
        if skipped:
            log(f"no --camera: vision candidates {skipped} skipped")
            candidates = [c for c in candidates if not policy.CANDIDATES[c].get("vision")]

    # --- test guard, before any work: a second test is counted, never hidden (<out>/test-history.jsonl, the report)
    existing = sorted(logs_dir.glob("test-*.jsonl")) if logs_dir.exists() else []
    history_file = pathlib.Path(a.out) / "test-history.jsonl"
    history = spec.read_jsonl(history_file) if history_file.exists() else []
    if a.test and existing and a.refuse_retest:
        raise SystemExit(f"test records exist: {[str(x) for x in existing]} and --refuse-retest is set; "
                         f"{len(history)} earlier test run(s) in {history_file}")
    if a.test and existing:
        old = pathlib.Path(a.out) / "old" / f"{datetime.datetime.now():%Y%m%d-%H%M%S}"
        old.mkdir(parents=True, exist_ok=True)
        for x in existing:
            shutil.move(str(x), old / x.name)
        log(f"TEST RUN #{len(history) + 1}: {len(existing)} earlier test record(s) moved to {old}; every test run is listed in {history_file}")
    a.test_number = len(history) + 1
    a.history = history

    # --- split
    sets = [s for s in a.sets.split(",") if s] or data.available_sets(a.data)
    if not sets:
        raise SystemExit("no sets found; run openhlm/fetch.py or give --data")
    split_file = run_dir / "split.json"
    if split_file.exists():
        split = {s: {int(k): v for k, v in r.items()} for s, r in json.load(open(split_file)).items()}
        if list(split) != sets:
            raise SystemExit(f"{split_file} covers {list(split)}, this run asks for {sets}: use another --tag")
    else:
        split = data.split_sets(a.data, sets)
        split_file.write_text(json.dumps({s: {str(k): v for k, v in r.items()} for s, r in split.items()}, indent=1))
    n_test = sum(sum(1 for v in r.values() if v == "test") for r in split.values())
    for s, roles in split.items():
        c = {r: sum(1 for v in roles.values() if v == r) for r in ("train", "val", "test")}
        log(f"{s}: train {c['train']}, val {c['val']}, test {c['test']}")
    assert n_test == spec.SAMPLE, n_test
    (run_dir / "plan.json").write_text(json.dumps(dict(tag=a.tag, sets=sets, candidates=candidates, devices=a.device_list, budget_min=a.budget_min,
                                                       camera=a.camera, seed=a.seed, started=datetime.datetime.now().isoformat()), indent=1))

    # --- cache: every set's tables as one .npz, read once here (threaded parquet reads), so the units start in seconds
    if a.cache:
        for s in sets:
            t0 = time.time()
            sd = data.SetData(s, data.set_dir(a.data, s), cache_dir=a.cache, state_key=a.state_key or None, action_key=a.action_key or None)
            log(f"{s}: {len(sd.episodes)} episodes, {sum(len(v['state']) for v in sd.episodes.values())} frames, state {sd.state_dim}, action "
                f"{sd.action_dim}, {sd.n_tasks} tasks; cache {sd.cache_path(a.cache).name} ({time.time() - t0:.0f} s)")
            del sd

    # --- train: learned candidates round-robin over the GPU workers, ridge on a CPU worker (or with the rest without GPUs)
    gpus = [d for d in a.device_list if d.startswith("cuda")]
    units = []
    learned = [c for c in candidates if c != "ridge"]
    i = 0
    for c in learned:
        for s in sets:
            units.append(((gpus or ["cpu"])[i % max(1, len(gpus))], s, c))
            i += 1
    if "ridge" in candidates:
        units += [("cpu" if gpus else "cpu", s, "ridge") for s in sets]
    budget_s = max(60.0, (a.budget_min - a.reserve_min) * 60 - (time.time() - started))
    log(f"{len(units)} training units on {a.device_list}, {budget_s:.0f} s of training budget per worker")
    results = run_units(units, a, run_dir, split_file, budget_s)
    failed = [k for k, v in results.items() if v not in (0, "cached")]
    if failed:
        log(f"{len(failed)} units failed: {failed}")

    # --- select
    sel = select(run_dir, sets, candidates, a, split)
    if len(sel) != len(sets):
        raise SystemExit(f"selection covers {list(sel)} of {sets}: training failed somewhere (see {run_dir / 'logs'})")
    for s, info in sel.items():
        log(f"{s}: selected {info['candidate']} (val {info['val']['successes']}/{info['val']['episodes']}, mean skill {info['val']['mean_skill']:.3f}); "
            + ", ".join(f"{c} {v['successes']}/{v['episodes']}" for c, v in info["candidates"].items()))
    (run_dir / "selection.json").write_text(json.dumps(sel, indent=1))
    diag = diagnose(a, run_dir, split, sel)

    # --- test
    records, refs, summ = [], [], None
    if a.test:
        records, refs = test(a, run_dir, split, sel, logs_dir)
        summ = report(a, run_dir, split, sel, records, refs, started, a.budget_min, diag)
        with open(history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(number=a.test_number, tag=a.tag, date=datetime.datetime.now().isoformat(timespec="minutes"), sets=sets,
                                    candidates=candidates, camera=a.camera, seed=a.seed, successes=summ["successes"], episodes=summ["episodes"],
                                    mean_skill=round(summ["mean_skill"], 4), selected={s: v["candidate"] for s, v in sel.items()},
                                    contract=dict(history_s=spec.HISTORY_S, horizon_s=spec.HORIZON_S, skill_pass=spec.SKILL_PASS, salt=spec.SALT))) + "\n")
        log(f"TEST {summ['successes']}/{summ['episodes']} = {summ['rate']:.0%} (Wilson 95 % {summ['wilson95']}), mean skill {summ['mean_skill']:.3f}; "
            f"records {logs_dir / f'test-{a.tag}.jsonl'}, report {run_dir / 'report.md'}")
        if not a.no_check:
            r = subprocess.run([PY, str(REPO / "scripts" / "check_openhlm.py"), "--offline", str(logs_dir)], cwd=str(REPO))
            log(f"check_openhlm.py rc={r.returncode}")
    else:
        report(a, run_dir, split, sel, records, refs, started, a.budget_min, diag)
        log(f"validation only (no --test); report {run_dir / 'report.md'}")
    log(f"PIPELINE-DONE tag={a.tag} seconds={time.time() - started:.0f} test={'yes' if a.test else 'no'}"
        + (f" successes={summ['successes']}/{summ['episodes']}" if summ else ""))
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
