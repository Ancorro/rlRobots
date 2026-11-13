"""Decode one camera of the OpenHLM sets into a per-episode frame cache for the vision candidates of the chunk policy:
<frames>/<set>/<camera>/episode_XXXXXX.npy, uint8 (frames x size x size x 3), every --stride-th frame of the episode
(frame i of the cache is row i * stride of the table), and <frames>/<set>/<camera>/meta.json with the size and
stride. The policy sees the latest cached frame at or before the chunk start (at most stride - 1 frames old: 67 ms at
the default 3 and 30 fps; the contract's chunk starts are multiples of 3, so they are exact).

Two sources, by the camera feature's dtype in meta/info.json: `image` (OpenHLM-data: PNG bytes in the episode's
parquet rows, read in small row batches so a worker never holds a whole table) and `video` (one mp4 per camera and
episode at the table's frame rate, decoded with PyAV).

  python -m openhlm.frames --camera head_image_left [--size 96] [--stride 3] [--sets a,b] [--data DIR]
                           [--out runs/openhlm/frames] [--workers 8] [--episodes-file split.json]   (CPU only)

A camera is its feature key in meta/info.json (OpenHLM-data: head_image_left, left_wrist_image, right_wrist_image;
other LeRobot sets: observation.images.<name>); `--list` prints them. Needs the videos fetched (openhlm/fetch.py
--videos <camera>). Decoding uses PyAV; the resize is the decoder's own (swscale, bilinear).
"""
import argparse
import json
import multiprocessing
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from openhlm import data, spec  # noqa: E402


cameras_of = data.cameras_of


def video_path(sd, camera, episode):
    chunk = episode // int(sd.info.get("chunks_size", 1000))
    rel = sd.info.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
    return sd.dir / rel.format(episode_chunk=chunk, video_key=camera, episode_index=episode)


def decode_video(src, size, length, stride):
    import av
    frames = []
    with av.open(str(src)) as c:
        stream = c.streams.video[0]
        stream.thread_type = "AUTO"
        for i, fr in enumerate(c.decode(stream)):
            if length and i >= length:
                break
            if i % stride == 0:
                frames.append(fr.reformat(width=size, height=size, format="rgb24").to_ndarray())
    return frames


def decode_parquet(src, camera, size, length, stride):
    """PNG / JPEG bytes of an `image` feature, every stride-th row in table order (frame_index), resized with PIL."""
    import io
    import pyarrow.parquet as pq
    from PIL import Image
    pf = pq.ParquetFile(str(src))
    cols = [camera] + (["frame_index"] if "frame_index" in pf.schema_arrow.names else [])
    frames, i = {}, 0
    for batch in pf.iter_batches(batch_size=64, columns=cols):
        col = batch.column(0)
        idx = batch.column(1).to_pylist() if len(cols) > 1 else list(range(i, i + len(col)))
        for j, fi in enumerate(idx):
            if fi % stride == 0 and (not length or fi < length):
                v = col[j].as_py()
                raw = v.get("bytes") if isinstance(v, dict) else v
                if raw is None and isinstance(v, dict) and v.get("path"):
                    raw = open(pathlib.Path(src).parent / v["path"], "rb").read()
                im = Image.open(io.BytesIO(raw)).convert("RGB").resize((size, size), Image.BILINEAR)
                frames[fi // stride] = np.asarray(im, np.uint8)
        i += len(col)
    return [frames[k] for k in sorted(frames)]


def decode(args):
    """(source path, camera or None for a video, out path, size, length, stride) -> frames written, or an error string."""
    src, camera, dst, size, length, stride = args
    try:
        frames = decode_video(src, size, length, stride) if camera is None else decode_parquet(src, camera, size, length, stride)
        arr = np.stack(frames) if frames else np.zeros((0, size, size, 3), np.uint8)
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.save(dst, arr)
        return len(arr)
    except Exception as e:  # noqa: BLE001
        return f"{src.name}: {type(e).__name__}: {e}"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--camera", default="")
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--stride", type=int, default=3, help="keep every stride-th frame")
    p.add_argument("--sets", default="", help="comma-separated; default: every set found")
    p.add_argument("--data", default="")
    p.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parents[1] / "runs" / "openhlm" / "frames"))
    p.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() // 2))
    p.add_argument("--episodes-file", default="", help="split.json: only its episodes (all roles)")
    p.add_argument("--list", action="store_true", help="print each set's cameras and stop")
    p.add_argument("--force", action="store_true", help="decode again even when the cache file exists")
    p.add_argument("--cache", default=str(pathlib.Path(__file__).resolve().parents[1] / "runs" / "openhlm" / "cache"), help="table cache (data.SetData)")
    a = p.parse_args(argv)
    sets = [s for s in a.sets.split(",") if s] or data.available_sets(a.data)
    if not sets:
        raise SystemExit("no sets found; run openhlm/fetch.py or give --data")
    if a.list or not a.camera:
        for s in sets:
            d = data.set_dir(a.data, s)
            info = json.loads((d / "meta" / "info.json").read_text())
            print(f"{s}: cameras {cameras_of(info)}; fps {info['fps']}; features {[k for k in info['features'] if k not in cameras_of(info)]}")
        return 0
    wanted = json.load(open(a.episodes_file)) if a.episodes_file else None
    jobs, out, errors_missing = [], pathlib.Path(a.out), []
    for s in sets:
        sd = data.SetData(s, data.set_dir(a.data, s), episodes=(list(map(int, wanted[s])) if wanted else None), cache_dir=a.cache or None)
        if a.camera not in cameras_of(sd.info):
            raise SystemExit(f"{s}: camera {a.camera!r} not in {cameras_of(sd.info)}")
        meta = out / s / a.camera / "meta.json"
        if meta.exists():
            m = json.load(open(meta))
            if (m.get("size"), m.get("stride")) != (a.size, a.stride) and not a.force:
                raise SystemExit(f"{meta} has size {m.get('size')} stride {m.get('stride')}; asked {a.size}/{a.stride}: --force rebuilds")
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_text(json.dumps(dict(size=a.size, stride=a.stride, camera=a.camera)))
        kind = data.camera_kind(sd.info, a.camera)
        for e, ep in sd.episodes.items():
            dst = out / s / a.camera / f"episode_{e:06d}.npy"
            if dst.exists() and not a.force:
                continue
            if kind == "image":  # frames in the parquet rows
                jobs.append((sd.episode_file(e), a.camera, dst, a.size, len(ep["state"]), a.stride))
                continue
            src = video_path(sd, a.camera, e)
            if not src.exists():
                errors_missing.append(f"{s}: {src.name} missing (openhlm/fetch.py --videos {a.camera})")
                continue
            jobs.append((src, None, dst, a.size, len(ep["state"]), a.stride))
    print(f"{len(jobs)} videos to decode at {a.size}x{a.size}, stride {a.stride}, with {a.workers} workers -> {out}"
          + (f"; {len(errors_missing)} videos missing" if errors_missing else ""), flush=True)
    errors, n = list(errors_missing), 0
    with multiprocessing.Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(decode, jobs), 1):
            if isinstance(r, str):
                errors.append(r)
            else:
                n += r
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} videos, {n} frames, {len(errors)} errors", flush=True)
    for e in errors[:10]:
        print("  error:", e)
    print(f"FRAMES-DONE videos={len(jobs) - len(errors)} frames={n} errors={len(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
