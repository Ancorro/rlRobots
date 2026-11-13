"""Reading the OpenHLM sets (LeRobot v2.0 / v2.1 layout; v3.0 tables are read the same way) into arrays, the training
statistics, the split, and the chunk windows the policy and the baselines are scored on.

A set: meta/info.json (fps, features with shapes and names), meta/episodes.jsonl (episode_index, length, tasks),
data/chunk-*/episode_*.parquet with the state and action list columns (named "state" / "actions" in OpenHLM-data,
"observation.state" / "action" in other LeRobot sets: resolve_keys) and an episode_index column. Videos are not read
here; openhlm/frames.py turns the chosen camera's mp4s into a per-episode uint8 cache that `SetData` picks up.

Where the data is: --data DIR with DIR/<set>/meta/info.json, or the Hugging Face cache (openhlm/fetch.py puts the
snapshot at $HF_HOME/hub/datasets--OpenHLM--OpenHLM-data/snapshots/<rev>/<set>).
"""
import glob
import hashlib
import json
import os
import pathlib

import numpy as np

from openhlm import spec


def hub_root():
    """The downloaded snapshot folder in the Hugging Face cache, or None."""
    home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    snaps = sorted(glob.glob(os.path.join(home, "hub", "datasets--" + spec.REPO_ID.replace("/", "--"), "snapshots", "*")),
                   key=os.path.getmtime)
    return pathlib.Path(snaps[-1]) if snaps else None


def set_dir(data, name):
    """Folder of one set (has meta/info.json), under --data or the hub cache."""
    for root in ([pathlib.Path(data)] if data else []) + [hub_root()]:
        if root and (root / name / "meta" / "info.json").exists():
            return root / name
    raise FileNotFoundError(f"{name}: no meta/info.json under {data or '(no --data)'} or the Hugging Face cache; run openhlm/fetch.py")


def available_sets(data, names=spec.SETS):
    return [n for n in names if any(r and (r / n / "meta" / "info.json").exists() for r in ([pathlib.Path(data)] if data else []) + [hub_root()])]


def resolve_keys(info, state_key=None, action_key=None):
    """The state and action feature keys of a set: the ones given, else the first of spec.STATE_KEYS / ACTION_KEYS
    present, else a numeric feature whose key contains 'state' / 'action'."""
    feats = info["features"]
    numeric = [k for k, f in feats.items() if f.get("dtype") not in ("video", "image", "string")]

    def find(given, candidates, word):
        if given:
            if given not in feats:
                raise KeyError(f"feature {given!r} not in meta/info.json (features: {list(feats)})")
            return given
        for c in candidates:
            if c in feats:
                return c
        hits = [k for k in numeric if word in k.lower()]
        if hits:
            return hits[0]
        raise KeyError(f"no {word} feature among {list(feats)}; give --state-key / --action-key")
    return find(state_key, spec.STATE_KEYS, "state"), find(action_key, spec.ACTION_KEYS, "action")


def cameras_of(info):
    """The video / image feature keys of a set (OpenHLM-data: head_image_left, left_wrist_image, right_wrist_image,
    `image` features: PNG bytes in the parquet rows; `video` features are mp4 files)."""
    return [k for k, f in info["features"].items() if f.get("dtype") in ("video", "image")]


def camera_kind(info, camera):
    return info["features"][camera].get("dtype") if camera in info["features"] else None


def infer_hold_map(S, A, match=spec.HOLD_MATCH, max_lag=15, max_rows=8_000):
    """Without joint names: for each action dimension the state dimension that tracks it, or -1. Tracks = at the best
    lag of 0..max_lag frames (a position-controlled joint follows its command a little late) the median over frames
    of |a_d(t) - s_j(t + lag)| is under `match` x std(a_d); each state dimension is used once (best match first)."""
    n = len(A)
    max_lag = max(0, min(max_lag, n - 2))
    rows = np.linspace(0, max(0, n - 1 - max_lag), min(n, max_rows)).astype(int)
    std = A.std(0).astype(np.float64)
    best = np.full((A.shape[1], S.shape[1]), np.inf)
    Ar = A[rows].astype(np.float64)
    for lag in range(max_lag + 1):
        Sr = S[rows + lag].astype(np.float64)
        for d in range(A.shape[1]):
            if std[d] < spec.STD_FLOOR:
                continue
            r = np.median(np.abs(Ar[:, d:d + 1] - Sr), 0) / std[d]
            best[d] = np.minimum(best[d], r)
    hmap = np.full(A.shape[1], -1, np.int64)
    scores = sorted((float(best[d, j]), d, int(j)) for d in range(A.shape[1]) for j in [int(np.argmin(best[d]))] if best[d, j] < match)
    used = set()
    for _, d, j in scores:
        if j not in used:
            hmap[d], used = j, used | {j}
    return hmap


def feature_names(info, key):
    """The feature's dimension names as a flat list, or None (Unitree-style nested [[...]] lists are flattened)."""
    f = info["features"].get(key) or {}
    names = f.get("names")
    while isinstance(names, list) and len(names) == 1 and isinstance(names[0], list):
        names = names[0]
    if isinstance(names, dict):  # {"motors": [...]} style
        names = next(iter(names.values()), None)
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        return None
    dim = int(np.prod(f.get("shape", [len(names)])))
    return list(names) if len(names) == dim else None


def column(table, key):
    """A list column (fixed or variable size) as a (rows x dim) float32 array."""
    col = table.column(key).combine_chunks()
    if hasattr(col, "flatten"):
        flat = col.flatten().to_numpy(zero_copy_only=False)
        return np.ascontiguousarray(flat.reshape(table.num_rows, -1), dtype=np.float32)
    return np.asarray(col.to_numpy(zero_copy_only=False), np.float32).reshape(table.num_rows, -1)


def episode_meta(d):
    """{episode_index: {"length": int, "tasks": [...]}} from meta/episodes.jsonl (v2) or meta/episodes/*.parquet (v3)."""
    out = {}
    p = d / "meta" / "episodes.jsonl"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                out[int(e["episode_index"])] = dict(length=int(e.get("length", 0)), tasks=e.get("tasks", []))
        return out
    files = sorted(glob.glob(str(d / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))
    if files:
        import pyarrow.parquet as pq
        for f in files:
            t = pq.read_table(f).to_pylist()
            for e in t:
                out[int(e["episode_index"])] = dict(length=int(e.get("length", 0)), tasks=e.get("tasks", []))
    return out


class SetData:
    """One set in memory: per-episode state / action arrays, names, fps, and (optional) cached frames."""

    def __init__(self, name, d, state_key=None, action_key=None, episodes=None, cameras=None, frames_dir=None, cache_dir=None, workers=8):
        self.name, self.dir = name, pathlib.Path(d)
        self.info = json.loads((self.dir / "meta" / "info.json").read_text(encoding="utf-8"))
        self.fps = float(self.info["fps"])
        state_key, action_key = resolve_keys(self.info, state_key, action_key)
        self.state_key, self.action_key = state_key, action_key
        self.hold_source = "names"
        self.state_names = feature_names(self.info, state_key)
        self.action_names = feature_names(self.info, action_key)
        self.meta = episode_meta(self.dir)
        self.tasks = {}
        tp = self.dir / "meta" / "tasks.jsonl"
        if tp.exists():
            for line in tp.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    t = json.loads(line)
                    self.tasks[int(t["task_index"])] = t["task"]
        self.episodes = {}
        want = None if episodes is None else set(int(e) for e in episodes)
        cache = self.cache_path(cache_dir) if cache_dir else None
        if cache and cache.exists():
            self.load_cache(cache, want)
        else:
            self.read_tables(want, workers)
            if cache and want is None:
                self.save_cache(cache)
        if not self.episodes:
            raise ValueError(f"{name}: none of the wanted episodes found")
        # tasks: the instruction is known to a policy at run time, so its index conditions the policies (one-hot)
        self.n_tasks = max(len(self.tasks), 1 + max(v["task_index"] for v in self.episodes.values()), 1)
        self.state_dim = next(iter(self.episodes.values()))["state"].shape[1]
        self.action_dim = next(iter(self.episodes.values()))["action"].shape[1]
        self.horizon, self.history = spec.horizon(self.fps), spec.history(self.fps)
        self.cameras = list(cameras or [])
        self.frames_dir = pathlib.Path(frames_dir) if frames_dir else None
        self.frames, self.frame_stride = {}, 1
        if self.cameras:
            self.load_frames()

    # --- tables: parquet (threaded) or one cached .npz per set (the 2026-11-04 run spent ~60 s of every 120 s training
    # unit reading 405 parquet files from network storage; the cache reads in about a second) ---
    def read_tables(self, want, workers=8):
        import pyarrow.parquet as pq
        from concurrent.futures import ThreadPoolExecutor
        files = self.episode_files(want)
        if not files:
            raise FileNotFoundError(f"{self.name}: no parquet files under {self.dir / 'data'}")
        # only the needed columns: OpenHLM-data keeps its camera frames (PNG bytes, `image` features) in the same rows, a
        # whole table is hundreds of MB per episode and reading all of them killed a 32 GB job (2026-11-04)
        have = set(pq.read_schema(files[0]).names)
        cols = [c for c in (self.state_key, self.action_key, "episode_index", "frame_index", "task_index") if c in have]
        with ThreadPoolExecutor(max(1, workers)) as pool:
            tables = list(pool.map(lambda f: pq.read_table(f, columns=cols), files))
        for f, t in zip(files, tables):
            ep = t.column("episode_index").to_numpy()
            if want is not None and not (set(np.unique(ep).tolist()) & want):
                continue
            S, A = column(t, self.state_key), column(t, self.action_key)
            bad = int(np.isnan(S).sum() + np.isnan(A).sum())
            if bad:
                print(f"{self.name}: {bad} NaN values in {pathlib.Path(f).name} replaced by 0", flush=True)
                S, A = np.nan_to_num(S), np.nan_to_num(A)
            task = t.column("task_index").to_numpy() if "task_index" in t.column_names else np.zeros(len(ep), np.int64)
            order = np.argsort(t.column("frame_index").to_numpy(), kind="stable") if "frame_index" in t.column_names else None
            for e in np.unique(ep):
                e = int(e)
                if want is not None and e not in want:
                    continue
                rows = np.flatnonzero(ep == e)
                if order is not None:
                    rows = rows[np.argsort(order[rows], kind="stable")]
                self.episodes[e] = dict(state=S[rows], action=A[rows], task_index=int(task[rows[0]]), task=self.tasks.get(int(task[rows[0]]), ""))

    def cache_path(self, cache_dir):
        tag = hashlib.sha1(f"{self.dir.resolve()}|{self.info.get('total_frames')}|{self.info.get('total_episodes')}|{self.state_key}|{self.action_key}".encode()).hexdigest()[:10]
        return pathlib.Path(cache_dir) / f"{self.name}-{tag}.npz"

    def save_cache(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        eps = sorted(self.episodes)
        np.savez(path, episodes=np.array(eps, np.int64), lengths=np.array([len(self.episodes[e]["state"]) for e in eps], np.int64),
                 task_index=np.array([self.episodes[e]["task_index"] for e in eps], np.int64),
                 state=np.concatenate([self.episodes[e]["state"] for e in eps]), action=np.concatenate([self.episodes[e]["action"] for e in eps]))

    def load_cache(self, path, want):
        z = np.load(path)
        S, A = z["state"], z["action"]
        off = 0
        for e, L, ti in zip(z["episodes"].tolist(), z["lengths"].tolist(), z["task_index"].tolist()):
            if want is None or e in want:
                self.episodes[e] = dict(state=S[off:off + L], action=A[off:off + L], task_index=ti, task=self.tasks.get(ti, ""))
            off += L

    def episode_file(self, e):
        """The parquet file of one episode (LeRobot v2: from the set's data_path template)."""
        tmpl = self.info.get("data_path", "")
        if "{episode_index" not in tmpl:
            raise FileNotFoundError(f"{self.name}: no per-episode data_path template in meta/info.json")
        f = self.dir / tmpl.format(episode_chunk=e // int(self.info.get("chunks_size", 1000)), episode_index=e)
        if not f.exists():
            raise FileNotFoundError(f"{self.name}: {f} missing")
        return f

    def episode_files(self, want):
        """The parquet files to read: the wanted episodes' own files when the set's data_path template names one file
        per episode (LeRobot v2), else every file under data/."""
        tmpl = self.info.get("data_path", "")
        if want is not None and "{episode_index" in tmpl:
            chunk = int(self.info.get("chunks_size", 1000))
            files = [self.dir / tmpl.format(episode_chunk=e // chunk, episode_index=e) for e in sorted(want)]
            if all(f.exists() for f in files):
                return [str(f) for f in files]
        return sorted(glob.glob(str(self.dir / "data" / "**" / "*.parquet"), recursive=True))

    # --- frames (openhlm/frames.py cache) ---
    def frame_path(self, camera, episode):
        return self.frames_dir / self.name / camera / f"episode_{episode:06d}.npy"

    def load_frames(self):
        """The cameras' cached frames of every loaded episode: self.frames[e] (cached frames x cams x h x w x 3) with one
        cached frame per `frame_stride` table rows (the cameras must share a stride); padded by repeating the last
        frame when a video is shorter than its table, so lengths and chunk starts stay the table's."""
        strides = []
        for cam in self.cameras:
            m = self.frames_dir / self.name / cam / "meta.json"
            strides.append(int(json.load(open(m)).get("stride", 1)) if m.exists() else 1)
        if len(set(strides)) > 1:
            raise ValueError(f"{self.name}: cameras with different frame strides {dict(zip(self.cameras, strides))}")
        self.frame_stride = strides[0] if strides else 1
        missing, short = [], 0
        for e in self.episodes:
            need = -(-len(self.episodes[e]["state"]) // self.frame_stride)  # ceil
            per_cam = []
            for cam in self.cameras:
                p = self.frame_path(cam, e)
                if not p.exists():
                    missing.append(str(p))
                    continue
                fr = np.asarray(np.load(p, mmap_mode="r")[:need])
                if len(fr) < need:
                    short += 1
                    fr = np.concatenate([fr, np.repeat(fr[-1:], need - len(fr), 0)]) if len(fr) else np.zeros((need,) + fr.shape[1:], np.uint8)
                per_cam.append(fr)
            if len(per_cam) == len(self.cameras) and per_cam:
                self.frames[e] = np.stack(per_cam, 1)
        if missing:  # a few missing videos: those episodes get black frames; many: the cache is not usable
            if len(missing) > 0.2 * len(self.episodes) * len(self.cameras) or not self.frames:
                raise FileNotFoundError(f"{self.name}: {len(missing)} cached frame files missing, e.g. {missing[0]}; run openhlm/frames.py")
            shape = next(iter(self.frames.values())).shape[1:]
            for e in self.episodes:
                if e not in self.frames:
                    self.frames[e] = np.zeros((-(-len(self.episodes[e]["state"]) // self.frame_stride),) + shape, np.uint8)
            print(f"{self.name}: {len(missing)} cached frame files missing (e.g. {missing[0]}): black frames for those episodes", flush=True)
        if short:
            print(f"{self.name}: {short} videos shorter than their tables; last frame repeated", flush=True)

    # --- names and the hold map ---
    def hold_map(self, episodes=None):
        """For each action dimension, the state dimension that holds it, or -1 (then the last recorded action is held).
        By name when both features are named and at least one name matches; else inferred from the data of `episodes`
        (default: all loaded; training episodes in the trainer) by infer_hold_map. Sets self.hold_source."""
        if self.state_names and self.action_names:
            idx = {n: i for i, n in enumerate(self.state_names)}
            m = np.array([idx.get(n, -1) for n in self.action_names], np.int64)
            if (m >= 0).any():
                self.hold_source = "names"
                return m
        eps = list(self.episodes) if episodes is None else list(episodes)
        S = np.concatenate([self.episodes[e]["state"] for e in eps])
        A = np.concatenate([self.episodes[e]["action"] for e in eps])
        self.hold_source = "inferred"
        return infer_hold_map(S, A, max_lag=int(round(0.5 * self.fps)))

    def lengths(self):
        return {e: len(v["state"]) for e, v in self.episodes.items()}

    def usable(self, episodes=None):
        """Episodes long enough for one chunk (history + horizon)."""
        eps = self.episodes if episodes is None else [e for e in episodes if e in self.episodes]
        return [e for e in eps if len(self.episodes[e]["state"]) >= self.history + self.horizon]

    def stats(self, episodes):
        """Per-dimension mean and std of state and action over the given (training) episodes."""
        S = np.concatenate([self.episodes[e]["state"] for e in episodes])
        A = np.concatenate([self.episodes[e]["action"] for e in episodes])
        return dict(state_mean=S.mean(0), state_std=S.std(0), action_mean=A.mean(0), action_std=A.std(0))

    def chunk_starts(self, e):
        """Evaluation chunk starts of an episode: from the first frame with a full history, every horizon frames,
        while a full horizon remains (the last partial chunk is not scored)."""
        L = len(self.episodes[e]["state"])
        return list(range(self.history - 1, L - self.horizon + 1, self.horizon))

    def train_starts(self, e, stride=1):
        L = len(self.episodes[e]["state"])
        return list(range(self.history - 1, L - self.horizon + 1, stride))


def load_sets(data, names, episodes_by_set=None, **kw):
    """{name: SetData}; episodes_by_set limits what is read (e.g. only the split's episodes)."""
    out = {}
    for n in names:
        out[n] = SetData(n, set_dir(data, n), episodes=(episodes_by_set or {}).get(n), **kw)
    return out


def describe_hold_map(sd, hmap):
    """One line: how many action dimensions are held from a state dimension and how, for logs and reports."""
    n = int((hmap >= 0).sum())
    src = getattr(sd, "hold_source", "names")
    return f"{n}/{len(hmap)} action dims held from a state dim ({src}" + (f": {hmap.tolist()}" if src == "inferred" else "") + ")"


def split_sets(data, names, canonical=False):
    """The contract's split from each set's episode list (meta/episodes.jsonl, without reading the tables). Episodes
    shorter than one chunk (history + horizon frames) cannot be scored by any policy and are left out before the split."""
    eps = {}
    for n in names:
        d = set_dir(data, n)
        fps = float(json.loads((d / "meta" / "info.json").read_text(encoding="utf-8"))["fps"])
        need = spec.history(fps) + spec.horizon(fps)
        meta = episode_meta(d)
        if not meta:  # no meta: read episode ids and lengths from the tables
            meta = {e: dict(length=len(v["state"])) for e, v in SetData(n, d).episodes.items()}
        eps[n] = sorted(e for e, m in meta.items() if m["length"] >= need or m["length"] == 0)
    return spec.split_episodes(eps, canonical=canonical)


def hold_chunk(sd, e, t, hmap):
    """The hold baseline's actions for the chunk starting at t: s_t for mapped dimensions, a_{t-1} for the rest."""
    ep = sd.episodes[e]
    K = sd.horizon
    base = np.empty(sd.action_dim, np.float32)
    last = ep["action"][t - 1] if t > 0 else ep["action"][0]
    for d, j in enumerate(hmap):
        base[d] = ep["state"][t, j] if j >= 0 else last[d]
    return np.repeat(base[None], K, 0)


def nmae(pred, true, std):
    """Mean over steps and dimensions of |pred - true| / std, constant dimensions (spec.scored_dims) left out."""
    w = np.asarray(spec.scored_dims(std), np.float64)
    if w.sum() == 0:
        return 0.0
    err = np.abs(pred.astype(np.float64) - true.astype(np.float64)) / np.maximum(std, spec.STD_FLOOR)
    return float((err * w).sum() / (w.sum() * err.shape[0]))
