"""A stand-in for OpenHLM-data in the LeRobot v2.1 layout, for the tests and for dry runs without the download:
per set meta/info.json, meta/episodes.jsonl, meta/tasks.jsonl, data/chunk-000/episode_*.parquet and, with --videos,
small camera frames. Two schemas: "openhlm" (the real dataset's: columns `state` and `actions` without joint names,
cameras `head_image_left`, `left_wrist_image` as `image` features, i.e. PNG bytes in the parquet rows, which is how
OpenHLM-data ships its frames; here the state gets 3 extra sensor dimensions and the action 2 command dimensions, so
the spaces differ as they may in the data) and "lerobot" (`observation.state` / `action` with joint names,
`observation.images.<camera>` as `video` features, one mp4 per episode). The motions are smooth random joint trajectories whose commanded action leads the
measured state by a few frames (as on a position-controlled robot), so a policy that reads the history can predict the
next second better than holding still, and the hold baseline is a real baseline.

  python -m openhlm.fake_data OUT_DIR [--sets=2] [--episodes=12] [--fps=30] [--videos] [--seed=0] [--dims=28] [--schema=openhlm|lerobot]
"""
import argparse
import json
import pathlib

import numpy as np

from openhlm import spec

JOINTS = [f"{side}_{part}" for side in ("left", "right") for part in
          ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")] + \
         [f"{side}_hand_{f}_{i}" for side in ("left", "right") for f, i in
          (("thumb", 0), ("thumb", 1), ("thumb", 2), ("middle", 0), ("middle", 1), ("index", 0), ("index", 1))]


def poses(rng, dims, n=6):
    """A set's pose library: the places the teleoperator's arms go (shared by its episodes, like a task's waypoints)."""
    return rng.uniform(-0.8, 0.8, (n, dims)).astype(np.float32)


def trajectory(rng, length, dims, fps, lib, lead=4):
    """Actions (length x dims) and states (length x dims): minimum-jerk moves between poses of the library (with a
    little per-episode offset), pauses between them, the hands closing after one move and opening after a later one;
    the state is the action delayed `lead` frames through a first-order lag with measurement noise."""
    a = np.zeros((length, dims), np.float32)
    grip = np.zeros(length, np.float32)
    order = np.arange(len(lib))  # the task's fixed sequence of waypoints
    cur = lib[order[0]] + rng.normal(0, 0.03, dims).astype(np.float32)
    i, k = 0, 1
    arrivals = []
    while i < length:
        pause = int(rng.uniform(0.2, 0.6) * fps)
        a[i:i + pause] = cur
        i += pause
        if i >= length:
            break
        tgt = lib[order[k % len(lib)]] + rng.normal(0, 0.03, dims).astype(np.float32)
        dur = max(2, int(rng.uniform(0.8, 2.0) * fps))
        s = np.linspace(0, 1, dur)
        prof = (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5)[:, None]
        a[i:i + dur] = (cur + (tgt - cur) * prof)[:length - i]
        i += dur
        arrivals.append(min(i, length))
        cur = tgt
        k += 1
    ramp = max(1, int(0.3 * fps))
    for j, level in ((1, 0.6), (3, 0.0)):  # close after the second move, open after the fourth
        if len(arrivals) > j:
            t0 = arrivals[j]
            n = min(ramp, length - t0)
            grip[t0:t0 + n] = grip[t0 - 1] + (level - grip[t0 - 1]) * np.linspace(0, 1, n)
            grip[t0 + n:] = level
    a[:, 14:] += grip[:, None]
    s = np.empty_like(a)
    s[0] = a[0]
    alpha = 1 - np.exp(-1.0 / lead)
    for j in range(1, length):
        s[j] = s[j - 1] + alpha * (a[max(0, j - lead)] - s[j - 1])
    s += rng.normal(0, 0.002, s.shape).astype(np.float32)
    return a, s


def png_bytes(frame):
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="PNG")
    return buf.getvalue()


def write_video(path, frames, fps):
    import av
    path.parent.mkdir(parents=True, exist_ok=True)
    out = av.open(str(path), "w")
    st = out.add_stream("libx264", rate=int(fps))
    st.width, st.height, st.pix_fmt = frames.shape[2], frames.shape[1], "yuv420p"
    st.options = {"crf": "30", "preset": "ultrafast"}
    for fr in frames:
        for pkt in st.encode(av.VideoFrame.from_ndarray(fr, format="rgb24")):
            out.mux(pkt)
    for pkt in st.encode():
        out.mux(pkt)
    out.close()


SCHEMAS = {  # state key, action key, named joints, camera keys, extra state / action dims
    "openhlm": dict(state="state", action="actions", named=False, cameras=("head_image_left", "left_wrist_image"), extra_state=3, extra_action=2, images="image"),
    "lerobot": dict(state="observation.state", action="action", named=True, cameras=("observation.images.cam_left_high",), extra_state=0, extra_action=0, images="video"),
}


def make_set(root, name, episodes, fps, dims, rng, videos=False, cameras=None, size=48, min_len=90, max_len=200,
             extra_state=None, extra_action=None, n_tasks=3, schema="openhlm"):
    """Write one set; returns its episode lengths. extra_state: state-only dimensions (like an IMU reading);
    extra_action: action-only dimensions (like a base velocity command), so the state and action spaces differ.
    n_tasks: each task has its own waypoints; an episode's task is drawn at random and recorded as task_index."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    sc = SCHEMAS[schema]
    cameras = sc["cameras"] if cameras is None else cameras
    extra_state = sc["extra_state"] if extra_state is None else extra_state
    extra_action = sc["extra_action"] if extra_action is None else extra_action
    d = pathlib.Path(root) / name
    (d / "meta").mkdir(parents=True, exist_ok=True)
    (d / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    names = (JOINTS * (dims // len(JOINTS) + 1))[:dims]
    names = [n if names.count(n) == 1 else f"{n}_{i}" for i, n in enumerate(names)]
    state_names = names + [f"imu_{i}" for i in range(extra_state)]
    action_names = names + [f"base_cmd_{i}" for i in range(extra_action)]
    lengths, index = [], 0
    task_names = [f"task {k}: pick the fruit and place it in basket {k}" for k in range(n_tasks)]
    libs = [poses(rng, dims) for _ in range(n_tasks)]
    with open(d / "meta" / "episodes.jsonl", "w") as ej:
        for e in range(episodes):
            length = int(rng.integers(min_len, max_len + 1))
            task = int(rng.integers(n_tasks))
            a, s = trajectory(rng, length, dims, fps, libs[task])
            if extra_state:
                s = np.concatenate([s, rng.normal(0, 0.1, (length, extra_state)).astype(np.float32)], 1)
            if extra_action:
                t = np.arange(length)[:, None] / fps
                a = np.concatenate([a, (0.3 * np.sin(0.5 * t + rng.uniform(0, 6, extra_action))).astype(np.float32)], 1)
            cols = {
                sc["state"]: pa.array(s.tolist(), pa.list_(pa.float32(), s.shape[1])),
                sc["action"]: pa.array(a.tolist(), pa.list_(pa.float32(), a.shape[1])),
                "timestamp": pa.array((np.arange(length) / fps).astype(np.float32)),
                "frame_index": pa.array(np.arange(length), pa.int64()),
                "episode_index": pa.array(np.full(length, e), pa.int64()),
                "index": pa.array(np.arange(index, index + length), pa.int64()),
                "task_index": pa.array(np.full(length, task, np.int64)),
            }
            if videos:
                for cam in cameras:
                    # a bright square that follows joint 0 (so an image carries some of the state) over noise
                    frames = rng.integers(0, 60, (length, size, size, 3), dtype=np.uint8)
                    x = ((s[:, 0] + 1) / 2 * (size - 8)).clip(0, size - 8).astype(int)
                    for i in range(length):
                        frames[i, size // 2 - 4:size // 2 + 4, x[i]:x[i] + 8] = 230
                    if sc["images"] == "video":
                        write_video(d / "videos" / "chunk-000" / cam / f"episode_{e:06d}.mp4", frames, fps)
                    else:  # PNG bytes per row, as Hugging Face datasets' Image feature stores them
                        cols[cam] = pa.array([dict(bytes=png_bytes(fr), path=None) for fr in frames],
                                             pa.struct([("bytes", pa.binary()), ("path", pa.string())]))
            pq.write_table(pa.table(cols), d / "data" / "chunk-000" / f"episode_{e:06d}.parquet")
            ej.write(json.dumps(dict(episode_index=e, tasks=[task_names[task]], length=length)) + "\n")
            lengths.append(length)
            index += length
    feats = {
        sc["state"]: dict(dtype="float32", shape=[len(state_names)], names=[state_names] if sc["named"] else None),
        sc["action"]: dict(dtype="float32", shape=[len(action_names)], names=[action_names] if sc["named"] else None),
        "timestamp": dict(dtype="float32", shape=[1], names=None),
        "frame_index": dict(dtype="int64", shape=[1], names=None),
        "episode_index": dict(dtype="int64", shape=[1], names=None),
        "index": dict(dtype="int64", shape=[1], names=None),
        "task_index": dict(dtype="int64", shape=[1], names=None),
    }
    if videos:
        for cam in cameras:
            feats[cam] = dict(dtype=sc["images"], shape=[size, size, 3], names=["height", "width", "channel"],
                              info=({"video.fps": fps, "video.codec": "h264"} if sc["images"] == "video" else None))
    info = dict(codebase_version="v2.1", robot_type="unitree_g1", total_episodes=episodes, total_frames=int(sum(lengths)),
                total_tasks=n_tasks, total_videos=(episodes * len(cameras) if videos else 0), total_chunks=1, chunks_size=1000,
                fps=fps, splits={"train": f"0:{episodes}"},
                data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4", features=feats)
    (d / "meta" / "info.json").write_text(json.dumps(info, indent=1))
    (d / "meta" / "tasks.jsonl").write_text("".join(json.dumps(dict(task_index=k, task=t)) + "\n" for k, t in enumerate(task_names)))
    return lengths


def make(root, sets=2, episodes=12, fps=30, dims=28, videos=False, seed=0, schema="openhlm", **kw):
    rng = np.random.default_rng(seed)
    out = {}
    for name in spec.SETS[:sets]:
        out[name] = make_set(root, name, episodes, fps, dims, rng, videos=videos, schema=schema, **kw)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("out")
    p.add_argument("--sets", type=int, default=2)
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--dims", type=int, default=28)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--videos", action="store_true")
    p.add_argument("--schema", default="openhlm", choices=sorted(SCHEMAS))
    a = p.parse_args(argv)
    lengths = make(a.out, a.sets, a.episodes, a.fps, a.dims, a.videos, a.seed, schema=a.schema)
    for name, l in lengths.items():
        print(f"{name}: {len(l)} episodes, {sum(l)} frames")


if __name__ == "__main__":
    main()
