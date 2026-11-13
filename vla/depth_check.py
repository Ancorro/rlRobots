"""Depth from one colour camera vs the simulator's true depth (cam_front, table-plane fit as in MonoDepth).

Per model, over N scenes of each object set (scanned household objects, random primitives):
  object   error at object pixels (median, p90), mm
  smear    background pixels within 3 px outside an object that the model pulls toward the camera by > 1 cm
           (the 'objects look 2-3 cm too big' failure), share
  table    error on bare table, mm; ms per image
Usage (cluster): sbatch vla/run.sbatch vla/depth_check.py small base depthpro da3-mono moge2-l [--scenes=20]
(run.sbatch uses .venv-vla; the DA3 / MoGe models need PY=.venv-depth/bin/python, see RUNBOOK)
"""
import sys
import time

sys.path.insert(0, ".")
import cv2
import mujoco
import numpy as np

from so101.camera import MonoDepth, SimCameras
from so101.objects import body_name, build, layout, random_objects, scanned_objects

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
MODELS = [a for a in sys.argv[1:] if not a.startswith("--")]
N = int(ARGS.get("scenes", 20))


def scenes(kind, rng):
    for _ in range(N):
        for _ in range(50):
            specs = scanned_objects(rng, 3) if kind == "scanned" else random_objects(rng, 5)
            try:
                poses = layout(rng, specs, tries=200, max_r=0.23)
                break
            except RuntimeError:
                pass
        model = build(specs, poses)
        data = mujoco.MjData(model)
        data.qpos[:6] = [0, -1.6, 1.4, 1.2, 0, 0]  # arm up and back, mostly out of view
        data.ctrl[:] = data.qpos[:6]
        mujoco.mj_step(model, data, nstep=300)  # objects settle
        objs = {model.body(body_name(s)).id for s in specs}
        yield model, data, np.array([model.body_rootid[model.geom_bodyid[g]] in objs for g in range(model.ngeom)] + [False])


def main():
    rows = {m: {"scanned": [], "primitives": []} for m in MODELS}
    nets = {}
    for kind in ("scanned", "primitives"):
        for model, data, obj_geom in scenes(kind, np.random.default_rng(11)):
            sim = SimCameras(model, data, ["cam_front"])
            truth = sim.grab(0).depth
            obj = sim.known_px(0, obj_geom)
            ring = cv2.dilate(obj.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & ~obj
            for name in MODELS:
                if name not in nets:
                    nets[name] = MonoDepth(sim, name)
                md = nets[name]
                md.inner, md.names, md.known = sim, sim.names, np.zeros(model.ngeom + 1, bool)
                t = time.time()
                f = md.grab(0)
                ms = (time.time() - t) * 1000
                e = np.abs(f.depth - truth)
                tab = ~obj & ~ring & ~np.isnan(md.table_depth(f))
                rows[name][kind].append((np.median(e[obj]), np.percentile(e[obj], 90),
                                         np.mean(truth[ring] - f.depth[ring] > 0.01), np.median(e[tab]), ms))
            sim.close()
    print(f"{'model':12s} {'set':10s} obj med mm  obj p90 mm  smear %  table mm   ms")
    for name in MODELS:
        for kind, r in rows[name].items():
            a = np.median(np.array(r), 0)
            print(f"{name:12s} {kind:10s} {a[0]*1000:9.1f} {a[1]*1000:10.1f} {a[2]*100:8.1f} {a[3]*1000:8.1f} {a[4]:5.0f}")


if __name__ == "__main__":
    main()
