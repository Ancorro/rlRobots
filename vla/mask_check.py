"""Outline-model check: the same Grounding DINO + SigLIP box, outlined by each masker, vs the simulator's true mask.

python vla/mask_check.py [--scenes=10] [--set=primitives|scanned] [--maskers=facebook/sam2.1-hiera-tiny,facebook/sam3]
Prints per masker: IoU (intersection over union) with the true object mask, pixels outside / missed, and seconds per mask.
"""
import sys
import time

import mujoco
import numpy as np
import torch

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
MASKERS = ARGS.get("maskers", "facebook/sam2.1-hiera-tiny,facebook/sam3").split(",")


def main():
    from so101.objects import body_name, build, layout, random_objects, scanned_objects
    from so101.see import Eyes
    rng = np.random.default_rng(int(ARGS.get("seed", 3)))
    scenes = []
    while len(scenes) < int(ARGS.get("scenes", 10)):
        specs = random_objects(rng, 5) if ARGS.get("set", "primitives") == "primitives" else scanned_objects(rng, 3)
        try:  # some draws do not fit on the table together: draw again (as the benchmark does)
            scenes.append((specs, layout(rng, specs, tries=200, max_r=0.23)))
        except RuntimeError:
            pass
    stats = {m: [] for m in MASKERS}
    for m in MASKERS:
        eyes = None
        for specs, poses in scenes:
            model = build(specs, poses)
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            if eyes is None:
                eyes = Eyes(model, data, cams=("cam_front",), masker=m)
            eyes.attach(model, data)
            cams = eyes.src
            cams.seg_r.update_scene(data, camera=cams.names[0])
            seg = cams.seg_r.render()
            img = cams.render_rgb(0)
            for spec in specs:
                body = model.body(body_name(spec)).id
                geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body]
                truth = np.isin(seg[..., 0], geoms) & (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)
                if truth.sum() < 50:
                    continue
                boxes, _ = eyes.boxes(img, spec["name"])
                if not len(boxes):
                    continue
                vs, us = np.nonzero(truth)
                tb = np.array([us.min(), vs.min(), us.max(), vs.max()])
                ious = [_box_iou(b, tb) for b in boxes]
                box = boxes[int(np.argmax(ious))]  # the detector box on the right object (isolates the outline step)
                if max(ious) < 0.5:
                    continue
                torch.cuda.synchronize()
                t = time.time()
                mask = eyes.mask(img, box)
                torch.cuda.synchronize()
                dt = time.time() - t
                inter, union = (mask & truth).sum(), (mask | truth).sum()
                stats[m].append((inter / union, (mask & ~truth).sum() / truth.sum(), (~mask & truth).sum() / truth.sum(), dt))
        a = np.array(stats[m])
        print(f"{m}: {len(a)} objects, IoU median {np.median(a[:, 0]):.3f} p10 {np.percentile(a[:, 0], 10):.3f}, "
              f"extra pixels {np.median(a[:, 1]) * 100:.1f}%, missed {np.median(a[:, 2]) * 100:.1f}%, "
              f"{np.median(a[1:, 3]) * 1000:.0f} ms per mask, GPU peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
        del eyes
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


if __name__ == "__main__":
    main()
