"""Download OpenHLM-data (Hugging Face `OpenHLM/OpenHLM-data`) into the Hugging Face cache and describe it: per set the
episodes, frames, frame rate, features, cameras, the contract's split, and whether the tables load. Run on a machine
with network access (the cluster login node, HF_HOME=$SHARE/hf_cache); the GPU job reads the cache offline.

  python -m openhlm.fetch                         tables and metadata of all five sets, then the table cache and the
                                                  frame cache of the three cameras (openhlm/frames.py: 96 px, every
                                                  3rd frame; from the PNG rows) that the vision candidates read
  python -m openhlm.fetch --sets a,b              some sets
  python -m openhlm.fetch --videos ""             tables only (no videos, no frames; the job then runs state-only)
  python -m openhlm.fetch --videos head_image_left   one camera only
  python -m openhlm.fetch --describe              no download: describe what the cache (or --data DIR) holds
  python -m openhlm.fetch --revision <sha>        pin a dataset revision (the default is the main branch)
The table cache (runs/openhlm/cache/<set>-<id>.npz, --cache) lets the GPU job's units start in a second instead of
reading hundreds of parquet files (the 2026-11-04 run lost half of every unit to that); the frame cache
(runs/openhlm/frames/<set>/<camera>/, --frames-out, --size, --stride, --workers) is what openhlm/run.sbatch detects
to switch the vision candidates on.
Prints one FETCH-DONE line; exits non-zero when a wanted set is missing or a table does not load.
"""
import argparse
import json

import numpy as np
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from openhlm import data, spec  # noqa: E402


def download(sets, videos, revision, local_dir=None):
    from huggingface_hub import snapshot_download
    patterns = []
    for s in sets:
        patterns += [f"{s}/meta/*", f"{s}/meta/**", f"{s}/data/**"]
        for cam in videos:
            patterns.append(f"{s}/videos/*/{cam}/*")  # the camera is its feature key (head_image_left, observation.images.x)
    kw = dict(repo_id=spec.REPO_ID, repo_type="dataset", allow_patterns=patterns, max_workers=8)
    if revision:
        kw["revision"] = revision
    if local_dir:
        kw["local_dir"] = local_dir
    return pathlib.Path(snapshot_download(**kw))


def describe(root, sets, verify=True):
    """Per set: episodes, frames, fps, features, cameras; the split; whether a table loads. Returns the problems."""
    problems, found = [], {}
    for s in sets:
        d = root / s
        if not (d / "meta" / "info.json").exists():
            problems.append(f"{s}: not found under {root}")
            continue
        info = json.loads((d / "meta" / "info.json").read_text(encoding="utf-8"))
        meta = data.episode_meta(d)
        feats = info["features"]
        cams = data.cameras_of(info)
        try:
            sk, ak = data.resolve_keys(info)
        except KeyError as ex:
            problems.append(f"{s}: {ex}")
            sk = ak = None
        lengths = [m["length"] for m in meta.values()]
        print(f"== {s}: codebase {info.get('codebase_version')}, robot {info.get('robot_type')}, fps {info.get('fps')}, "
              f"{len(meta)} episodes, {sum(lengths)} frames (episode length {min(lengths) if lengths else 0}-{max(lengths) if lengths else 0}), "
              f"{info.get('total_tasks', '?')} tasks")
        if sk and ak:
            sn, an = data.feature_names(info, sk), data.feature_names(info, ak)
            print(f"   state = {sk!r} {feats[sk].get('shape')} {'named' if sn else 'unnamed'}; action = {ak!r} {feats[ak].get('shape')} "
                  f"{'named' if an else 'unnamed'}; cameras {dict((c, feats[c].get('dtype')) for c in cams)} "
                  f"({'frames in the parquet rows' if any(feats[c].get('dtype') == 'image' for c in cams) else 'mp4 files: ' + str(info.get('video_path'))}); "
                  f"other features {[k for k in feats if k not in (sk, ak) and k not in cams]}")
            if sn and an:
                shared = sum(n in sn for n in an)
                print(f"   action dims held from a state dim of the same name: {shared}/{len(an)}" + ("" if shared == len(an) else f"; unmatched: {[n for n in an if n not in sn][:6]}"))
                if shared == 0:
                    print(f"   NOTE: no action name is a state name (state e.g. {sn[:3]}, action e.g. {an[:3]}): the hold map will be inferred from the data")
            else:
                print("   no joint names: the hold map is inferred from the training data (data.infer_hold_map); see 'tables load' below")
        else:
            print(f"   features {list(feats)}")
        tp = d / "meta" / "tasks.jsonl"
        if tp.exists():
            tasks = [json.loads(l).get("task", "") for l in tp.read_text(encoding="utf-8").splitlines() if l.strip()]
            print(f"   {len(tasks)} tasks, e.g. {tasks[:4]}")
        H, K = spec.history(float(info.get("fps", 30))), spec.horizon(float(info.get("fps", 30)))
        short = [e for e, m in meta.items() if m["length"] < H + K]
        if short:
            print(f"   {len(short)} episodes shorter than history + horizon ({H + K} frames): {short[:8]}")
        found[s] = sorted(meta)
        if verify:
            try:
                t0 = time.time()
                sd = data.SetData(s, d, episodes=sorted(meta)[:8])
                e = next(iter(sd.episodes))
                print(f"   tables load: episode {e} state {sd.episodes[e]['state'].shape} action {sd.episodes[e]['action'].shape} "
                      f"task {sd.episodes[e]['task']!r} ({time.time() - t0:.1f} s); on these {len(sd.episodes)} episodes: "
                      + data.describe_hold_map(sd, sd.hold_map()))
                st = sd.stats(list(sd.episodes))
                print(f"   action std per dim: {np.round(st['action_std'], 3).tolist()}")
            except Exception as ex:  # noqa: BLE001
                problems.append(f"{s}: tables do not load: {type(ex).__name__}: {ex}")
    if found:
        try:
            split = spec.split_episodes(found)
            print("== split (openhlm/spec.py):")
            for s, roles in split.items():
                c = {r: sum(1 for v in roles.values() if v == r) for r in ("train", "val", "test")}
                print(f"   {s}: train {c['train']}, val {c['val']}, test {c['test']} {sorted(e for e, r in roles.items() if r == 'test')}")
        except ValueError as ex:
            problems.append(str(ex))
    return problems


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--sets", default=",".join(spec.SETS))
    p.add_argument("--videos", default="head_image_left,left_wrist_image,right_wrist_image",
                   help="comma-separated cameras to fetch (if mp4 `video` features) and decode; '' for none")
    p.add_argument("--frames-out", default=str(pathlib.Path(__file__).resolve().parents[1] / "runs" / "openhlm" / "frames"))
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--workers", type=int, default=0, help="decoding processes (default: half the CPUs)")
    p.add_argument("--no-frames", action="store_true", help="fetch the videos but do not decode them")
    p.add_argument("--revision", default="")
    p.add_argument("--data", default="", help="describe this folder (DIR/<set>/meta/info.json) instead of the cache")
    p.add_argument("--local-dir", default="", help="download into this folder instead of the cache")
    p.add_argument("--describe", action="store_true", help="no download")
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--cache", default=str(pathlib.Path(__file__).resolve().parents[1] / "runs" / "openhlm" / "cache"),
                   help="build each set's table cache here (what the pipeline's units read; '' to skip)")
    a = p.parse_args(argv)
    sets = [s for s in a.sets.split(",") if s]
    unknown = [s for s in sets if s not in spec.SETS]
    if unknown:
        raise SystemExit(f"unknown sets {unknown}; the OpenHLM sets are {list(spec.SETS)}")
    if a.describe or a.data:
        root = pathlib.Path(a.data) if a.data else data.hub_root()
        if root is None:
            raise SystemExit("nothing in the Hugging Face cache; run without --describe to download")
    else:
        t0 = time.time()
        root = download(sets, [c for c in a.videos.split(",") if c], a.revision, a.local_dir or None)
        print(f"downloaded to {root} in {time.time() - t0:.0f} s")
    problems = describe(root, sets, verify=not a.no_verify)
    if a.describe:  # the job's own check: describe only, no caches (the pipeline builds the table cache when missing)
        a.cache, a.no_frames = "", True
    if a.cache and not problems:  # the tables once into one .npz per set, so the GPU job's units start in seconds
        for s in sets:
            t0 = time.time()
            sd = data.SetData(s, root / s, cache_dir=a.cache)
            print(f"   cache: {sd.cache_path(a.cache)} ({sum(len(v['state']) for v in sd.episodes.values())} frames, {time.time() - t0:.0f} s)")
            del sd
    cams = [c for c in a.videos.split(",") if c]
    if cams and not a.no_frames and not problems:  # the frame cache of each fetched camera (openhlm/frames.py in-process)
        import multiprocessing
        import shutil
        from openhlm import frames
        total = sum(sum(m["length"] for m in data.episode_meta(root / s).values()) for s in sets)
        size = a.size
        need = len(cams) * (total // a.stride) * size * size * 3
        pathlib.Path(a.frames_out).mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(a.frames_out).free
        if need * 1.2 > free and size > 64:
            size = 64
            print(f"   frame cache at {a.size} px would need {need / 2 ** 30:.1f} GB of {free / 2 ** 30:.1f} GB free: using {size} px")
        print(f"   decoding {len(cams)} camera(s) x {total} frames, every {a.stride}rd at {size} px: about {len(cams) * (total // a.stride) * size * size * 3 / 2 ** 30:.1f} GB", flush=True)
        workers = a.workers or max(2, min(8, multiprocessing.cpu_count() - 1))  # each worker holds one row batch of PNGs
        for cam in cams:
            t0 = time.time()
            argv = ["--camera", cam, "--size", str(size), "--stride", str(a.stride), "--sets", ",".join(sets), "--data", str(root),
                    "--out", a.frames_out, "--cache", a.cache or "", "--workers", str(workers)]
            rc = frames.main(argv)
            print(f"   frames of {cam}: {'ok' if rc == 0 else 'ERRORS'} ({time.time() - t0:.0f} s) -> {a.frames_out}")
            if rc:  # not a PROBLEM: the job runs state only for sets without a complete frame cache
                print(f"   NOTE: frames of {cam} incomplete; openhlm/run.sbatch falls back to state only where they are missing")
    for pr in problems:
        print("PROBLEM:", pr)
    print(f"FETCH-DONE root={root} sets={len(sets) - sum(p.split(':')[0] in sets and 'not found' in p for p in problems)}/{len(sets)} problems={len(problems)}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
